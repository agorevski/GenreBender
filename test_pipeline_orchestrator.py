"""Offline CLI and orchestration regressions; no media or cloud clients are used."""

import argparse
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
import yaml

import run_multi_genre_pipeline as pipeline


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / 'repository'
    root.mkdir()
    inputs = tmp_path / 'movie inputs'
    inputs.mkdir()
    for filename in ('movie.mp4', 'movie.srt', 'synopsis.txt'):
        (inputs / filename).write_text('fixture content', encoding='utf-8')
    movie = {
        'movie_name': 'Example Movie',
        'video': 'movie.mp4',
        'subtitles': 'movie.srt',
        'synopsis': 'synopsis.txt',
        'genres': ['comedy', 'thriller'],
    }
    config = inputs / 'pipeline.yaml'
    config.write_text(yaml.safe_dump({
        'settings': {'parallel_workers': 2},
        'movies': {'example': movie},
    }), encoding='utf-8')
    settings = root / 'settings.yaml'
    settings.write_text(yaml.safe_dump({
        'azure_openai': {
            'endpoint': 'https://example.openai.azure.com/',
            'api_key': 'offline-test-key',
            'deployment_name': 'test-deployment',
            'api_version': '2025-01-01-preview',
        },
    }), encoding='utf-8')
    plans = pipeline._phase1_plan('video', 'movie', 'synopsis', 'subtitles')
    plans += pipeline._genre_plan('comedy', 'video', 'movie')
    for script, _ in plans:
        (root / script).write_text('# Not executable in these tests\n', encoding='utf-8')
    monkeypatch.setattr(pipeline, 'PROJECT_ROOT', root)
    monkeypatch.setattr(pipeline, 'SETTINGS_PATH', settings)
    monkeypatch.setattr(pipeline.shutil, 'which', lambda name: str(root / f'{name}.exe'))
    monkeypatch.delenv('AZURE_OPENAI_ENDPOINT', raising=False)
    monkeypatch.delenv('AZURE_OPENAI_KEY', raising=False)
    return root, config


def arguments(config, **overrides):
    values = dict(
        config=str(config), config_key='example', genres=None,
        parallel_workers=None, skip_phase1=False, dry_run=False,
        force=False, sequential=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def update_config(config, change):
    value = yaml.safe_load(config.read_text(encoding='utf-8'))
    change(value)
    config.write_text(yaml.safe_dump(value), encoding='utf-8')


def invoke(config, *args):
    with pytest.raises(SystemExit) as result:
        pipeline.main(argv=['example', '--config', str(config), *args])
    return result.value.code


def test_config_relative_paths_and_normalized_deduplicated_genres(project, monkeypatch):
    root, config = project
    update_config(config, lambda value: value['movies']['example'].update(
        genres=[' Comedy ', 'THRILLER', 'comedy', ' WESTERN '],
    ))
    monkeypatch.chdir(root)
    movie, genres, workers = pipeline._load_request(arguments(config))
    assert genres == ['comedy', 'thriller', 'western']
    assert workers == 2
    assert Path(movie['video']) == config.parent / 'movie.mp4'
    assert pipeline._output_base(movie['video']) == root / 'outputs' / 'movie'


def test_cli_genres_and_workers_override_configuration(project):
    _, config = project
    _, genres, workers = pipeline._load_request(arguments(
        config, genres=' HORROR ,comedy,Horror', parallel_workers=7,
    ))
    assert genres == ['horror', 'comedy']
    assert workers == 7


def test_all_profile_genres_are_supported(project):
    _, config = project
    _, genres, _ = pipeline._load_request(arguments(config, genres=','.join(pipeline.ALL_GENRES)))
    assert genres == pipeline.ALL_GENRES


@pytest.mark.parametrize('genres', [[], '', 'comedy', None, ['typo'], [''], [1], ['comedy', None]])
def test_invalid_configured_genres_are_rejected(project, genres):
    _, config = project
    update_config(config, lambda value: value['movies']['example'].update(genres=genres))
    with pytest.raises(ValueError, match='[Gg]enre'):
        pipeline._load_request(arguments(config))


@pytest.mark.parametrize('genres', ['', 'comedy,', ',thriller', 'comedy,typo'])
def test_invalid_cli_genres_are_rejected(project, genres):
    _, config = project
    with pytest.raises(ValueError, match='[Gg]enre'):
        pipeline._load_request(arguments(config, genres=genres))


@pytest.mark.parametrize('workers', [0, -1, True, '2', 1.5, None])
def test_invalid_config_workers_are_rejected(project, workers):
    _, config = project
    update_config(config, lambda value: value['settings'].update(parallel_workers=workers))
    with pytest.raises(ValueError, match='positive integer'):
        pipeline._load_request(arguments(config))


@pytest.mark.parametrize('workers', ['0', '-2'])
def test_cli_workers_must_be_positive(project, workers, capsys):
    _, config = project
    assert invoke(config, '--parallel-workers', workers, '--dry-run') == 2
    assert 'positive integer' in capsys.readouterr().err


def test_workers_fall_back_to_four(project):
    _, config = project
    update_config(config, lambda value: value.pop('settings'))
    assert pipeline._load_request(arguments(config))[2] == 4


@pytest.mark.parametrize('contents,message', [
    ('', 'non-empty YAML mapping'),
    ('[]', 'non-empty YAML mapping'),
    ('movies: [broken', 'Invalid YAML'),
    ('movies: []', "'movies' mapping"),
    ('movies: {example: null}', 'must be a mapping'),
    ('movies: {other: {}}', 'Available movie keys: other'),
    ('movies: {example: {}}', "'movie_name' string"),
])
def test_malformed_config_reports_actionable_error(project, contents, message, capsys):
    _, config = project
    config.write_text(contents, encoding='utf-8')
    assert invoke(config, '--dry-run') == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert 'Traceback' not in captured.err


def test_missing_config_is_not_reloaded_in_error_handler(project, capsys):
    _, config = project
    config.unlink()
    assert invoke(config, '--dry-run') == 2
    assert 'Cannot read Pipeline config' in capsys.readouterr().err


@pytest.mark.parametrize('kind', ['missing', 'directory', 'empty'])
def test_invalid_input_file_is_rejected(project, kind, capsys):
    _, config = project
    video = config.parent / 'movie.mp4'
    video.unlink()
    if kind == 'directory':
        video.mkdir()
    elif kind == 'empty':
        video.touch()
    assert invoke(config, '--dry-run') == 2
    assert 'Video file' in capsys.readouterr().err


def test_non_mapping_global_settings_are_rejected(project):
    _, config = project
    update_config(config, lambda value: value.update(settings=[]))
    with pytest.raises(ValueError, match="'settings' must be a mapping"):
        pipeline._load_request(arguments(config))


def test_missing_stage_script_is_detected_before_generation(project, capsys):
    root, config = project
    (root / '14_scene_retrieval.py').unlink()
    assert invoke(config, '--dry-run') == 2
    assert '14_scene_retrieval.py' in capsys.readouterr().err


def test_dry_run_is_side_effect_free_and_includes_exact_plan(project, monkeypatch, capsys):
    root, config = project
    forbidden = Mock(side_effect=AssertionError('dry run must not execute'))
    monkeypatch.setattr(pipeline.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(pipeline, 'run_phase1_stages', forbidden)
    monkeypatch.setattr(pipeline, 'run_phase2_parallel', forbidden)
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    assert invoke(config, '--dry-run', '--force', '--genres', ' Comedy,comedy,HORROR ') == 0
    after = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    assert before == after
    assert not (root / 'outputs').exists()
    output = capsys.readouterr().out
    assert 'DRY RUN' in output
    assert 'Genres: comedy, horror' in output
    assert 'workers: 2' in output
    assert output.count('--force') == 18
    assert 'trailer_comedy_final.mp4' in output
    assert str(root / '1_shot_detection.py') in output
    assert str(config.parent / 'movie.mp4') in output
    assert 'cloud connectivity was not tested' in output
    forbidden.assert_not_called()


def test_dry_run_reports_missing_runtime_prerequisites_without_executing(project, monkeypatch, capsys):
    _, config = project
    monkeypatch.setattr(pipeline.shutil, 'which', lambda name: None)
    settings = yaml.safe_load(pipeline.SETTINGS_PATH.read_text(encoding='utf-8'))
    settings['azure_openai']['api_key'] = 'your_key'
    pipeline.SETTINGS_PATH.write_text(yaml.safe_dump(settings), encoding='utf-8')
    forbidden = Mock(side_effect=AssertionError('must not execute'))
    monkeypatch.setattr(pipeline.subprocess, 'Popen', forbidden)
    assert invoke(config, '--dry-run', '--skip-phase1') == 0
    output = capsys.readouterr().out
    assert 'ffmpeg is not on PATH' in output
    assert 'ffprobe is not on PATH' in output
    assert 'AZURE_OPENAI_KEY' in output
    assert '--skip-phase1' in output
    assert 'Generation is NOT ready' in output
    assert '1_shot_detection.py' not in output
    assert 'your_key' not in output
    forbidden.assert_not_called()


def test_actual_generation_is_blocked_before_expensive_work(project, monkeypatch):
    _, config = project
    monkeypatch.setattr(pipeline.shutil, 'which', lambda name: None)
    forbidden = Mock(side_effect=AssertionError('must not execute'))
    monkeypatch.setattr(pipeline, 'run_phase1_stages', forbidden)
    assert invoke(config) == 1
    forbidden.assert_not_called()


def test_environment_credentials_override_placeholders_without_exposure(project, monkeypatch):
    _, config = project
    pipeline.SETTINGS_PATH.write_text(yaml.safe_dump({
        'azure_openai': {
            'endpoint': 'https://yourendpoint.openai.azure.com/',
            'api_key': '${UNSET_TEST_SECRET}',
            'deployment_name': 'test-deployment',
            'api_version': '2025-01-01-preview',
        },
    }), encoding='utf-8')
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://example.openai.azure.com/')
    monkeypatch.setenv('AZURE_OPENAI_KEY', 'offline-test-env-key')
    movie, _, _ = pipeline._load_request(arguments(config))
    assert pipeline._readiness_issues(movie, False) == []


def test_malformed_runtime_settings_are_reported_safely(project, capsys):
    _, config = project
    pipeline.SETTINGS_PATH.write_text('azure_openai: [private-test-value', encoding='utf-8')
    assert invoke(config, '--dry-run') == 2
    error = capsys.readouterr().err
    assert 'Invalid YAML in Runtime settings' in error
    assert 'private-test-value' not in error


def create_resume_artifacts(movie):
    output = pipeline._output_base(movie['video'])
    files = {
        pipeline.PROJECT_ROOT / pipeline.get_story_graph_dir(movie['movie_name']) / 'story_graph.json':
            {'title': 'Example Movie'},
        output / 'shots' / 'shot_metadata.json': {'shots': []},
        output / 'checkpoint.json': {
            'stages': {stage: {'completed': True} for stage in pipeline.PHASE1_CHECKPOINT_STAGES},
        },
    }
    for path, value in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
    return output


def test_resume_checks_artifacts_without_mutating_checkpoint(project):
    _, config = project
    movie, _, _ = pipeline._load_request(arguments(config))
    output = create_resume_artifacts(movie)
    checkpoint = output / 'checkpoint.json'
    before = checkpoint.read_bytes()
    assert pipeline._resume_issues(movie['video'], movie['movie_name']) == []
    assert checkpoint.read_bytes() == before
    checkpoint.write_text('{"stages": {}}', encoding='utf-8')
    assert 'incomplete shared stages' in pipeline._resume_issues(movie['video'], movie['movie_name'])[0]
    checkpoint.write_text('broken json', encoding='utf-8')
    assert '--skip-phase1' in pipeline._resume_issues(movie['video'], movie['movie_name'])[0]


@pytest.mark.parametrize('sequential', [False, True])
def test_main_propagates_workers_force_and_failure_summary(project, monkeypatch, capsys, sequential):
    _, config = project
    phase1 = Mock(return_value=True)

    def phase2(genres, video, movie, *workers, force=False, errors=None):
        assert genres == ['comedy', 'thriller']
        assert force is True
        assert workers == (() if sequential else (2,))
        errors['comedy'] = '12_beat_sheet_generator.py: quota unavailable'
        return {'comedy': False, 'thriller': True}

    monkeypatch.setattr(pipeline, 'run_phase1_stages', phase1)
    monkeypatch.setattr(pipeline, 'run_phase2_sequential' if sequential else 'run_phase2_parallel', phase2)
    options = ['--force'] + (['--sequential'] if sequential else [])
    assert invoke(config, *options) == 1
    assert phase1.call_args.args[-1] is True
    output = capsys.readouterr().out
    assert 'Completed: 1/2 genres' in output
    assert 'quota unavailable' in output
    assert '--skip-phase1 --genres comedy' in output
    assert 'OK thriller' in output


def test_main_reports_all_genres_not_run_after_shared_failure(project, monkeypatch, capsys):
    _, config = project
    monkeypatch.setattr(pipeline, 'run_phase1_stages', lambda *args: False)
    forbidden = Mock(side_effect=AssertionError('Phase 2 must not execute'))
    monkeypatch.setattr(pipeline, 'run_phase2_parallel', forbidden)
    assert invoke(config) == 1
    output = capsys.readouterr().out
    assert 'NOT RUN comedy' in output
    assert 'NOT RUN thriller' in output
    forbidden.assert_not_called()


def test_main_success_exit_and_cli_worker_override(project, monkeypatch, capsys):
    _, config = project
    movie, genres, _ = pipeline._load_request(arguments(config))
    create_resume_artifacts(movie)
    forbidden = Mock(side_effect=AssertionError('Phase 1 must be skipped'))
    monkeypatch.setattr(pipeline, 'run_phase1_stages', forbidden)
    phase2 = Mock(return_value={genre: True for genre in genres})
    monkeypatch.setattr(pipeline, 'run_phase2_parallel', phase2)
    assert invoke(config, '--skip-phase1', '--parallel-workers', '3') == 0
    assert phase2.call_args.args[-1] == 3
    assert 'Completed: 2/2 genres' in capsys.readouterr().out
    forbidden.assert_not_called()


@pytest.mark.parametrize('runner', [pipeline.run_phase2_sequential, pipeline.run_phase2_parallel])
@pytest.mark.parametrize('unexpected_error', [False, True])
def test_every_genre_finishes_after_independent_failure(monkeypatch, runner, unexpected_error):
    calls = []

    def run(genre, video, movie, **kwargs):
        calls.append(genre)
        assert kwargs['force'] is True
        if genre == 'comedy':
            if unexpected_error:
                raise RuntimeError('worker diagnostic')
            return genre, False, '12_beat_sheet_generator.py: worker diagnostic'
        return genre, True, 'ready'

    monkeypatch.setattr(pipeline, 'run_genre_pipeline', run)
    errors = {}
    genres = ['comedy', 'thriller', 'horror']
    results = runner(genres, 'movie.mp4', 'Movie', force=True, errors=errors)
    assert results == {'comedy': False, 'thriller': True, 'horror': True}
    assert sorted(calls) == sorted(genres)
    assert 'worker diagnostic' in errors['comedy']
    assert len(errors) == 1


@pytest.mark.parametrize('force', [False, True])
def test_force_propagates_to_every_stage_without_unsupported_flags(project, monkeypatch, force):
    _, config = project
    movie, _, _ = pipeline._load_request(arguments(config))
    final_path = pipeline._trailer_path(movie['video'], 'comedy')
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b'offline-test-artifact')
    stage = Mock(return_value=(True, 'ok'))
    monkeypatch.setattr(pipeline, 'run_stage_script', stage)
    assert pipeline.run_phase1_stages(
        movie['video'], movie['movie_name'], movie['synopsis'], movie['subtitles'], force,
    )
    assert pipeline.run_genre_pipeline(
        'comedy', movie['video'], movie['movie_name'], force=force,
    )[1] is True
    assert stage.call_count == 12
    allowed = {'--input', '--genre', '--movie-name', '--synopsis', '--srt-file', '--force'}
    for call in stage.call_args_list:
        script, args = call.args[:2]
        assert ('--force' in args) is force
        assert all(arg in allowed for arg in args if arg.startswith('--'))
        if script in ('12_beat_sheet_generator.py', '13_embedding_generator.py', '14_scene_retrieval.py'):
            assert '--movie-name' in args
        if script == '15_timeline_constructor.py':
            assert '--movie-name' not in args


def test_genre_stops_its_dependent_stages_and_preserves_diagnostic(monkeypatch):
    stage = Mock(side_effect=[(True, 'ok'), (False, '13_embedding_generator.py: embedding quota exhausted')])
    monkeypatch.setattr(pipeline, 'run_stage_script', stage)
    genre, success, message = pipeline.run_genre_pipeline('comedy', 'video.mp4', 'Movie')
    assert genre == 'comedy'
    assert success is False
    assert 'embedding quota exhausted' in message
    assert stage.call_count == 2


def test_successful_exit_without_final_artifact_is_not_reported_as_success(project, monkeypatch):
    _, config = project
    movie, _, _ = pipeline._load_request(arguments(config))
    monkeypatch.setattr(pipeline, 'run_stage_script', Mock(return_value=(True, 'ok')))
    _, success, message = pipeline.run_genre_pipeline('comedy', movie['video'], movie['movie_name'])
    assert success is False
    assert 'Final trailer file not found' in message


class FakeProcess:
    def __init__(self, output='', returncode=0, timeout=False):
        self.stdout = StringIO(output)
        self.returncode = returncode
        self.timeout = timeout
        self.killed = False

    def wait(self, timeout=None):
        if timeout is not None and self.timeout:
            raise subprocess.TimeoutExpired('stage', timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


@pytest.mark.parametrize('stream', [False, True])
def test_subprocess_absolute_paths_bounded_combined_diagnostics(project, monkeypatch, capsys, stream):
    root, _ = project
    output = 'old context to discard\n' + ('x' * 10000 + '\n') * 40 + 'last stdout error\n'
    process = FakeProcess(output=output, returncode=3)
    popen = Mock(return_value=process)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', popen)
    success, message = pipeline.run_stage_script('12_beat_sheet_generator.py', ['--genre', 'comedy'],
                                                 '[COMEDY] ', stream_output=stream)
    assert success is False
    assert 'last stdout error' in message
    assert 'old context to discard' not in message
    assert 'exit code 3' in message
    assert len(message) < pipeline.DIAGNOSTIC_LIMIT + 150
    command = popen.call_args.args[0]
    assert command[:2] == [sys.executable, str(root / '12_beat_sheet_generator.py')]
    assert popen.call_args.kwargs['cwd'] == str(root)
    assert popen.call_args.kwargs['stderr'] == subprocess.STDOUT
    assert popen.call_args.kwargs['encoding'] == 'utf-8'
    captured = capsys.readouterr().out
    assert ('last stdout error' in captured) is stream
    assert process.stdout.closed


@pytest.mark.parametrize('stream', [False, True])
def test_stage_timeout_terminates_process_and_keeps_output(project, monkeypatch, stream):
    process = FakeProcess('last diagnostic before timeout\n', timeout=True)
    monkeypatch.setattr(pipeline.subprocess, 'Popen', Mock(return_value=process))
    success, message = pipeline.run_stage_script(
        '5_remote_analysis.py', [], stream_output=stream, timeout_seconds=0.01,
    )
    assert success is False
    assert 'timed out after 0.01 seconds' in message
    assert 'last diagnostic before timeout' in message
    assert process.killed
    assert process.stdout.closed


def test_process_start_failure_is_actionable(project, monkeypatch):
    monkeypatch.setattr(pipeline.subprocess, 'Popen', Mock(side_effect=OSError('permission denied')))
    success, message = pipeline.run_stage_script('1_shot_detection.py', [])
    assert success is False
    assert '1_shot_detection.py: could not start' in message
    assert 'permission denied' in message


def test_successful_stage_result(project, monkeypatch):
    monkeypatch.setattr(pipeline.subprocess, 'Popen', Mock(return_value=FakeProcess('ok\n')))
    assert pipeline.run_stage_script('1_shot_detection.py', [], stream_output=False)[0] is True
