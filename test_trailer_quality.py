"""Offline regressions for Stage 14 selections -> Stage 15 timing -> Stage 9 video."""

import importlib.util
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock
import uuid

import pytest

from pipeline_common import load_config, setup_directories
from trailer_generator.assembly.video_assembler import VideoAssembler


# The deterministic constructor does not need the narrative package's Azure SDK.
spec = importlib.util.spec_from_file_location(
    'quality_timeline_constructor',
    Path(__file__).parent / 'trailer_generator' / 'narrative' / 'timeline_constructor.py'
)
timeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timeline_module)
TimelineConstructor = timeline_module.TimelineConstructor


def test_unused_title_text_generation_is_disabled_by_default():
    settings = Path(__file__).parent / 'trailer_generator' / 'config' / 'settings.yaml'
    assert load_config(str(settings))['video']['ai_title_generation'] is False


@pytest.fixture
def workspace():
    path = Path(__file__).parent / f'.pytest-trailer-quality-{uuid.uuid4().hex}'
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def candidate(workspace, shot_id, duration=10.0, **overrides):
    path = workspace / f'shot_{shot_id:04d}.mp4'
    path.write_bytes(b'placeholder')
    return {
        'shot_id': shot_id, 'duration': duration,
        'start_time': shot_id * 20.0, 'end_time': shot_id * 20.0 + duration,
        'shot_path': str(path), 'score': 1 - shot_id / 100, 'caption': 'A scene',
        **overrides,
    }


def assert_contiguous(timeline):
    current = 0.0
    for shot in timeline['shots']:
        assert shot['timeline_start'] == pytest.approx(current)
        assert shot['timeline_duration'] > 0
        assert shot['use_start_offset'] >= 0
        assert shot['use_end_offset'] <= shot['source_duration'] + 1e-8
        assert shot['use_end_offset'] - shot['use_start_offset'] == pytest.approx(
            shot['timeline_duration'])
        current += shot['timeline_duration']
        assert shot['timeline_end'] == pytest.approx(current)
    assert timeline['actual_duration'] == pytest.approx(current)
    assert timeline['metadata']['pacing_profile']['shots_per_minute'] == pytest.approx(
        len(timeline['shots']) * 60 / current)


@pytest.mark.parametrize('duration', [0, -1, None, True, '90', float('nan'), float('inf')])
def test_invalid_target_duration(duration):
    with pytest.raises(ValueError, match='Target duration'):
        TimelineConstructor(duration)


@pytest.mark.parametrize('beats,scenes', [
    ([], {}), ([{}], {}), ([{'id': 'a'}, {'id': 'a'}], {}),
    ([{'id': None}], {}), ([{'id': 'a'}], {}),
    ([{'id': 'a'}], {'a': [None, {}, 'invalid']}),
    (None, {}), ([{'id': 'a'}], {'a': 1}),
])
def test_empty_or_malformed_selection_fails_explicitly(beats, scenes):
    with pytest.raises(ValueError):
        TimelineConstructor().construct_timeline(beats, scenes)


@pytest.mark.parametrize('overrides', [
    {'shot_path': ''}, {'shot_path': None}, {'shot_path': 'missing-shot.mp4'},
    {'shot_id': None}, {'shot_id': True}, {'shot_id': '1'},
    {'duration': 0}, {'duration': -1}, {'duration': float('nan')},
    {'duration': '5'}, {'duration': float('inf')}, {'start_time': -1},
    {'end_time': 1}, {'end_time': float('inf')}, {'start_time': None},
])
def test_bad_candidates_cannot_produce_timeline(workspace, overrides):
    bad = {**candidate(workspace, 1), **overrides}
    with pytest.raises(ValueError, match='No usable shots'):
        TimelineConstructor().construct_timeline([{'id': 'a'}], {'a': [bad]})


def test_short_sources_have_no_phantom_time_or_duplicate_content(workspace):
    first = candidate(workspace, 1, duration=0.5)
    second = candidate(workspace, 2, duration=0.7)
    alias = {**first, 'shot_id': 3}
    beats = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}, {'id': 'd'}]
    timeline = TimelineConstructor(90).construct_timeline(
        beats, {'a': [first], 'b': [first, alias], 'c': [second, second], 'd': []})
    assert [s['shot_id'] for s in timeline['shots']] == [1, 2]
    assert timeline['actual_duration'] == pytest.approx(1.2)
    assert timeline['metadata']['represented_beat_count'] == 2
    assert timeline['metadata']['duration_shortfall'] == pytest.approx(88.8)
    assert_contiguous(timeline)


def test_duration_is_capped_by_declared_source_range(workspace):
    shot = candidate(workspace, 1, duration=20, start_time=100, end_time=102)
    timeline = TimelineConstructor(10).construct_timeline([{'id': 'a'}], {'a': [shot]})
    assert timeline['actual_duration'] == 2
    assert timeline['shots'][0]['source_shot_start'] == 100
    assert timeline['shots'][0]['use_start_offset'] == 0
    assert_contiguous(timeline)


def test_empty_files_and_copied_source_ranges_are_not_reused(workspace):
    first = candidate(workspace, 1, duration=1)
    duplicate = candidate(workspace, 2, duration=1,
                          start_time=first['start_time'], end_time=first['end_time'])
    empty = candidate(workspace, 3)
    Path(empty['shot_path']).write_bytes(b'')
    result = TimelineConstructor(10).construct_timeline(
        [{'id': 'a'}, {'id': 'b'}], {'a': [first], 'b': [empty, duplicate]})
    assert result['total_shots'] == 1
    assert result['actual_duration'] == 1


def test_invalid_candidates_are_reported_even_when_other_candidates_work(workspace, caplog):
    good = candidate(workspace, 1)
    result = TimelineConstructor(5).construct_timeline(
        [{'id': 'opening'}], {'opening': [{'shot_id': 2}, good]})
    assert result['total_shots'] == 1
    assert 'skipped 1 candidates with invalid timing or missing/empty files' in caplog.text


def test_clipped_allocations_are_redistributed_without_repeating_shots(workspace):
    shots = [candidate(workspace, 1, duration=1),
             candidate(workspace, 2, duration=10),
             candidate(workspace, 3, duration=10)]
    used = set()
    selected = TimelineConstructor()._select_shots_for_beat(
        {'id': 'setup'}, [shots[0], shots[0], *shots[1:]],
        {'shot_count': 3, 'duration': 9, 'position': 0.25}, 0, 1, used)
    assert len(selected) == 3
    assert selected[0]['timeline_duration'] == 1
    assert sum(s['timeline_duration'] for s in selected) == pytest.approx(9)
    assert {s['shot_id'] for s in selected} == {1, 2, 3}
    assert {1, 2, 3} <= used


@pytest.mark.parametrize('position', [0, 0.2, 0.5, 0.8, 0.95])
def test_pacing_allocations_are_positive_and_complete(position):
    durations = TimelineConstructor()._allocate_shot_durations(10, 20, position)
    assert len(durations) == 10
    assert all(d > 0 for d in durations)
    assert sum(durations) == pytest.approx(20)


def test_realistic_stage14_file_builds_deterministic_multibeat_timeline(workspace):
    beats = [{'id': f'beat_{i}', 'voiceover': f'Line {i}'} for i in range(10)]
    scenes = {}
    previous = []
    for i, beat in enumerate(beats):
        current = [candidate(workspace, 1 + i * 5 + j, duration=15) for j in range(5)]
        scenes[beat['id']] = previous[:1] + current
        previous = current
    selected_path = workspace / 'selected_scenes.json'
    selected_path.write_text(json.dumps({'beats': beats, 'selected_scenes': scenes}))
    output_path = workspace / 'trailer_timeline.json'
    result = timeline_module.construct_timeline(selected_path, output_path, 60, 'drama')
    assert result == json.loads(output_path.read_text())
    assert result == TimelineConstructor(60).construct_timeline(beats, scenes, 'drama')
    assert result['actual_duration'] == pytest.approx(60)
    assert len({s['shot_id'] for s in result['shots']}) == result['total_shots']
    assert result['metadata']['represented_beat_count'] == 10
    assert any(s['transition_out'] == 'dissolve' for s in result['shots'])
    assert any(s['use_start_offset'] > 0 for s in result['shots'])
    for beat in beats:
        assert sum(s['voiceover'] is not None for s in result['shots']
                   if s['beat_id'] == beat['id']) == 1
    assert_contiguous(result)


@pytest.fixture
def assembler(workspace):
    return VideoAssembler(
        {'video': {'resolution': '160x90', 'fps': 24, 'codec': 'libx264',
                   'bitrate': '300k', 'preset': 'ultrafast'}},
        {'color_grade': {'filter': 'eq=saturation=0.8'}}, workspace,
        enable_color_grading=False, enable_transitions=True
    )


def render_shot(workspace, shot_id, duration=1.0, offset=1.0, available=4.0):
    source = candidate(workspace, shot_id, available)
    return {
        'shot_id': shot_id, 'shot_path': source['shot_path'],
        'timeline_duration': duration, 'use_start_offset': offset,
        'use_end_offset': offset + duration, 'use_duration': duration,
        'source_shot_start': source['start_time'], 'source_shot_end': source['end_time'],
        'source_duration': available, 'transition_out': 'dissolve',
    }


@pytest.mark.parametrize('patch', [
    {'timeline_duration': 0}, {'timeline_duration': -1},
    {'timeline_duration': 0.01},
    {'timeline_duration': None}, {'timeline_duration': '1'},
    {'timeline_duration': float('nan')}, {'timeline_duration': float('inf')},
    {'timeline_duration': True}, {'use_start_offset': -1},
    {'use_start_offset': float('inf')}, {'use_end_offset': 3},
    {'use_duration': 2}, {'timeline_start': 1}, {'timeline_end': 10},
    {'source_duration': 1}, {'source_shot_start': -1},
    {'source_shot_end': None}, {'shot_id': None}, {'shot_id': True},
])
def test_invalid_timing_is_rejected_before_ffmpeg(assembler, workspace, monkeypatch, patch):
    shot = {**render_shot(workspace, 1), **patch}
    run = Mock()
    monkeypatch.setattr('trailer_generator.assembly.video_assembler.subprocess.run', run)
    with pytest.raises(ValueError):
        assembler.assemble_video({'shots': [shot]}, workspace, workspace / 'out.mp4')
    run.assert_not_called()


def test_empty_or_inconsistent_timelines_never_start_processes(assembler, workspace, monkeypatch):
    run = Mock()
    monkeypatch.setattr('trailer_generator.assembly.video_assembler.subprocess.run', run)
    shot = render_shot(workspace, 1)
    for timeline in [{'shots': []}, {'shots': [shot], 'actual_duration': 10},
                     {'shots': [shot, shot]}, {'shots': [None]}]:
        with pytest.raises(ValueError):
            assembler.assemble_video(timeline, workspace, workspace / 'out.mp4')
    run.assert_not_called()


def test_missing_explicit_path_does_not_render_different_canonical_file(
        assembler, workspace, monkeypatch):
    shot = render_shot(workspace, 1)
    shot['shot_path'] = str(workspace / 'missing.mp4')
    run = Mock()
    monkeypatch.setattr('trailer_generator.assembly.video_assembler.subprocess.run', run)
    with pytest.raises(FileNotFoundError):
        assembler.assemble_video({'shots': [shot]}, workspace, workspace / 'out.mp4')
    run.assert_not_called()


def test_probe_prevents_stale_metadata_truncation(assembler, workspace, monkeypatch):
    shot = render_shot(workspace, 1)
    monkeypatch.setattr(assembler, '_probe_media', lambda path: {
        'duration': 1.2, 'has_audio': True})
    run = Mock()
    monkeypatch.setattr('trailer_generator.assembly.video_assembler.subprocess.run', run)
    with pytest.raises(ValueError, match='exceeds video duration'):
        assembler.assemble_video({'shots': [shot]}, workspace, workspace / 'out.mp4')
    run.assert_not_called()


@pytest.mark.parametrize('enable_transitions', [False, True])
def test_render_uses_source_offsets_paths_and_bounded_matching_audio(
        assembler, workspace, monkeypatch, enable_transitions):
    assembler.enable_transitions = enable_transitions
    shots = [render_shot(workspace, 1), render_shot(workspace, 2),
             render_shot(workspace, 3)]
    custom = workspace / 'chosen-scene.mp4'
    custom.write_bytes(b'placeholder')
    shots[0]['shot_path'] = str(custom)
    monkeypatch.setattr(assembler, '_probe_media', lambda path: {
        'duration': 4, 'has_audio': path != Path(shots[1]['shot_path'])})

    def fake_render(command, **kwargs):
        Path(command[-1]).write_bytes(b'rendered')
        return subprocess.CompletedProcess(command, 0, '', '')

    run = Mock(side_effect=fake_render)
    monkeypatch.setattr('trailer_generator.assembly.video_assembler.subprocess.run', run)
    assembler.assemble_video({'shots': shots, 'actual_duration': 3},
                             workspace, workspace / 'out.mp4')
    command = run.call_args.args[0]
    assert str(custom) in command
    assert str(workspace / 'shot_0001.mp4') not in command
    filters = command[command.index('-filter_complex') + 1]
    assert 'trim=start=1.0:duration=' in filters
    assert 'atrim=start=1.0:duration=' in filters
    assert '[1:a]' not in filters
    assert 'anullsrc=r=48000:cl=stereo' in filters
    assert 'scale=160:90' in filters
    assert 'fps=24' in filters
    if enable_transitions:
        assert 'duration=0.5:offset=1.0' in filters
        assert 'duration=0.5:offset=2.0' in filters
        assert filters.count('acrossfade=d=0.5') == 2
        assert '[0:v]trim=start=1.0:duration=1.5' in filters
    else:
        assert 'xfade=' not in filters
        assert 'concat=n=3:v=1:a=1' in filters
        assert '[0:v]trim=start=1.0:duration=1.0' in filters


def test_transitions_without_source_handles_preserve_hard_cuts(assembler, workspace):
    shots = [render_shot(workspace, 1, offset=0, available=1),
             render_shot(workspace, 2, offset=0, available=1)]
    filters = assembler._build_filter_complex(
        shots, workspace, [{'type': 'fade', 'duration': 0.5}])
    assert 'xfade=' not in filters
    assert 'concat=n=2:v=1:a=1' in filters


def test_partial_transition_list_keeps_all_shots_and_ignores_untrusted_offsets(
        assembler, workspace):
    shots = [render_shot(workspace, i) for i in range(1, 5)]
    filters = assembler._build_filter_complex(
        shots, workspace, [{'shot_index': 1, 'type': 'fade', 'duration': 10, 'offset': 900}])
    assert 'duration=0.5:offset=2.0' in filters
    assert '900' not in filters
    assert '[joinedv1][v3]concat=' in filters
    assert '[joineda1][a3]concat=' in filters
    assert 'concat=n=2:v=1:a=0,fps=24,settb=AVTB' in filters


@pytest.mark.parametrize('transition', [
    {'shot_index': 10, 'type': 'fade', 'duration': 0.5},
    {'type': 'unknown', 'duration': 0.5},
    {'type': 'fade', 'duration': -1},
    {'type': 'fade', 'duration': float('nan')},
])
def test_invalid_transition_is_rejected(assembler, workspace, transition):
    shots = [render_shot(workspace, 1), render_shot(workspace, 2)]
    with pytest.raises(ValueError):
        assembler._build_filter_complex(shots, workspace, [transition])


def test_ai_transition_selector_receives_stage15_shots(assembler, workspace, monkeypatch):
    shots = [render_shot(workspace, 1), render_shot(workspace, 2)]
    assembler.video_config['ai_transition_selection'] = True
    monkeypatch.setattr(assembler, '_probe_media', lambda path: {
        'duration': 4, 'has_audio': True})
    selector = Mock(return_value=[])
    monkeypatch.setattr(
        'trailer_generator.assembly.video_assembler.TransitionSelector.select_transitions',
        selector)

    def fake_render(command, **kwargs):
        Path(command[-1]).write_bytes(b'rendered')
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr('trailer_generator.assembly.video_assembler.subprocess.run', fake_render)
    assembler.assemble_video({'shots': shots}, workspace, workspace / 'out.mp4',
                             shot_metadata=[{'id': 1}], azure_client=Mock())
    selection_input = selector.call_args.args[0]['timeline']
    assert [s['shot_id'] for s in selection_input] == [1, 2]
    assert [s['duration'] for s in selection_input] == [1, 1]


@pytest.mark.parametrize('profile,expected', [
    ({'name': 'kdrama'}, 'kdrama'),
    ({'genre': 'romance'}, 'romance'),
    ({}, 'comedy'),
])
def test_titles_use_profile_name_or_actual_timeline_genre(
        assembler, workspace, monkeypatch, profile, expected):
    assembler.genre_profile = profile
    assembler.video_config['ai_title_generation'] = True
    monkeypatch.setattr(assembler, '_probe_media', lambda path: {
        'duration': 4, 'has_audio': True})
    output = workspace / 'out.mp4'
    output.write_bytes(b'rendered')
    monkeypatch.setattr(assembler, '_assemble_simple', Mock(return_value=output))
    title_generator = Mock()
    title_generator.return_value.generate_titles.return_value = []
    monkeypatch.setattr('trailer_generator.assembly.video_assembler.TitleGenerator',
                        title_generator)
    client = Mock()
    timeline = {'shots': [render_shot(workspace, 1)], 'target_genre': 'comedy'}
    assembler.assemble_video(timeline, workspace, output, azure_client=client)
    title_generator.assert_called_once_with(client, expected)
    title_generator.return_value.generate_titles.assert_called_once_with(timeline)


def test_legacy_duration_and_canonical_filename_remain_supported(assembler, workspace, monkeypatch):
    candidate(workspace, 1)
    monkeypatch.setattr(assembler, '_probe_media', lambda path: {
        'duration': 10, 'has_audio': True})
    assembler._validate_timeline({'shots': [{'shot_id': 1, 'duration': 2}]}, workspace)


@pytest.mark.parametrize('valid', [False, True])
def test_stage9_consumes_stage15_artifact_and_only_checkpoints_success(
        assembler, workspace, monkeypatch, valid):
    stage_spec = importlib.util.spec_from_file_location(
        'quality_video_stage', Path(__file__).parent / '9_video_assembly.py')
    stage = importlib.util.module_from_spec(stage_spec)
    stage_spec.loader.exec_module(stage)
    dirs = setup_directories(workspace, 'thriller')
    genre_dir = dirs['genre_output']
    selected_path = genre_dir / 'selected_scenes.json'
    selected_path.write_text(json.dumps({
        'beats': [{'id': 'opening'}, {'id': 'reveal'}],
        'selected_scenes': {
            'opening': [candidate(workspace, 1, 6)],
            'reveal': [candidate(workspace, 2, 6)],
        },
    }))
    timeline_path = genre_dir / 'trailer_timeline.json'
    timeline = timeline_module.construct_timeline(selected_path, timeline_path, 4)
    if not valid:
        timeline['shots'] = []
        timeline_path.write_text(json.dumps(timeline))
    checkpoint = Mock()
    checkpoint.is_stage_completed.side_effect = lambda name, *args: name != 'video_assembly'
    checkpoint.get_stats.return_value = {
        'completed_stages': 6, 'total_stages': 15, 'progress_percent': 40,
        'completed_list': [],
    }
    monkeypatch.setattr(stage, 'initialize_stage', lambda *args: (
        workspace, dirs, checkpoint, logging.getLogger('quality-stage9')))
    monkeypatch.setattr(stage, 'load_config', lambda *args: assembler.config)
    monkeypatch.setattr(stage, 'load_genre_profile', lambda *args: assembler.genre_profile)
    monkeypatch.setattr(stage, 'load_shots_from_metadata', lambda *args: [])
    monkeypatch.setattr(VideoAssembler, '_probe_media', staticmethod(lambda path: {
        'duration': 6, 'has_audio': True}))
    assembler_factory = Mock(wraps=VideoAssembler)
    monkeypatch.setattr(stage, 'VideoAssembler', assembler_factory)
    monkeypatch.setattr(sys, 'argv', [
        '9_video_assembly.py', '--input', 'movie.mp4', '--genre', 'thriller',
        '--no-ai-titles', '--no-ai-transitions', '--no-color-grade',
    ])

    def fake_render(command, **kwargs):
        Path(command[-1]).write_bytes(b'rendered')
        return subprocess.CompletedProcess(command, 0, '', '')

    run = Mock(side_effect=fake_render)
    monkeypatch.setattr('trailer_generator.assembly.video_assembler.subprocess.run', run)
    if valid:
        stage.main()
        checkpoint.mark_stage_completed.assert_called_once()
        details = checkpoint.mark_stage_completed.call_args.args[1]
        assert details['duration'] == pytest.approx(4)
        assert details['shots_count'] == 2
        assert Path(details['output_file']).is_file()
        assert Path(details['output_file']).parent == genre_dir
        assert assembler_factory.call_args.kwargs['output_dir'] == genre_dir
        filters = run.call_args.args[0]
        filters = filters[filters.index('-filter_complex') + 1]
        for shot in timeline['shots']:
            assert f"trim=start={shot['use_start_offset']}:duration=" in filters
    else:
        with pytest.raises(SystemExit) as error:
            stage.main()
        assert error.value.code == 1
        run.assert_not_called()
        checkpoint.mark_stage_completed.assert_not_called()


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'),
                    reason='FFmpeg and FFprobe are required for synthetic media smoke tests')
@pytest.mark.parametrize('enable_transitions', [False, True])
def test_ffmpeg_synthetic_source_offset_and_duration(assembler, workspace, enable_transitions):
    shots = [render_shot(workspace, i, offset=1.5, available=3) for i in range(1, 4)]
    # The selected interval is blue, not the red beginning of each source.
    for i, shot in enumerate(shots):
        command = [
            'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
            'color=c=red:s=160x90:r=24:d=1',
            '-f', 'lavfi', '-i', 'color=c=blue:s=160x90:r=24:d=2',
        ]
        if i != 1:
            command += ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=3']
        command += ['-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[v]',
                    '-map', '[v]']
        if i != 1:
            command += ['-map', '2:a', '-c:a', 'aac']
        command += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', shot['shot_path']]
        subprocess.run(command, check=True, capture_output=True)
    assembler.enable_transitions = enable_transitions
    output = workspace / 'rendered.mp4'
    assembler.assemble_video({'shots': shots, 'actual_duration': 3},
                             workspace, output)
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(output)],
        check=True, capture_output=True, text=True)
    media = json.loads(result.stdout)
    assert float(media['format']['duration']) == pytest.approx(3, abs=0.1)
    for stream in media['streams']:
        assert float(stream['duration']) == pytest.approx(3, abs=0.1)
    frame = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(output), '-frames:v', '1',
         '-vf', 'scale=1:1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        check=True, capture_output=True).stdout
    assert len(frame) == 3
    assert frame[2] > 150 and frame[0] < 50
