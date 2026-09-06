"""Offline regressions for per-genre finalization and audio preservation."""

import importlib.util
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import struct
import sys
from unittest.mock import Mock

import pytest

from pipeline_common import setup_directories
from trailer_generator.audio.audio_mixer import AudioMixer


@pytest.fixture
def audio_config(tmp_path):
    return {'audio': {
        'music_library_path': str(tmp_path / 'music'),
        'sample_rate': 48000,
        'bitrate': '192k',
        'output_format': 'aac',
        'ducking_threshold': -20,
        'ducking_ratio': 4,
        'normalization_target': -14,
        'ai_music_selection': False,
    }}


@pytest.fixture
def mixer(tmp_path, audio_config):
    return AudioMixer(audio_config, {'music_tags': []}, tmp_path)


@pytest.mark.parametrize('ducking', [True, False])
@pytest.mark.parametrize('has_dialogue', [True, False])
def test_music_loops_only_to_actual_trailer_duration(
        mixer, tmp_path, monkeypatch, ducking, has_dialogue):
    run = Mock()
    monkeypatch.setattr('trailer_generator.audio.audio_mixer.subprocess.run', run)
    dialogue = tmp_path / 'dialogue.wav'
    if has_dialogue:
        dialogue.touch()
    method = mixer._mix_with_ducking if ducking else mixer._mix_simple
    method('short-music.wav', dialogue if has_dialogue else None,
           {'actual_duration': 12.5, 'target_duration': 90})
    command = run.call_args.args[0]
    assert command[command.index('-stream_loop') + 1] == '-1'
    assert command[command.index('-t') + 1] == '12.5'
    if has_dialogue:
        filters = command[command.index('-filter_complex') + 1]
        assert 'amix=inputs=2' in filters
        assert 'normalize=0' in filters
        if ducking:
            assert 'asplit=2[sidechain][dialogue]' in filters
            assert '[music][sidechain]sidechaincompress=' in filters
            assert '[dialogue][ducked]amix=' in filters


@pytest.mark.parametrize('duration', [None, 0, -1, float('nan'), float('inf'), True, '90'])
def test_invalid_audio_duration_does_not_start_ffmpeg(mixer, monkeypatch, duration):
    run = Mock()
    monkeypatch.setattr('trailer_generator.audio.audio_mixer.subprocess.run', run)
    with pytest.raises(ValueError, match='timeline duration'):
        mixer._mix_simple('music.wav', None, {'actual_duration': duration})
    run.assert_not_called()


def test_legacy_timeline_duration_supported(mixer):
    assert mixer._timeline_duration({'total_duration': 30}) == 30


def test_short_audio_cannot_truncate_video(mixer, tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr('trailer_generator.audio.audio_mixer.subprocess.run', run)
    mixer._mux_audio_video(tmp_path / 'video.mp4', tmp_path / 'audio.wav',
                           tmp_path / 'final.mp4')
    command = run.call_args.args[0]
    assert command[command.index('-af') + 1] == 'apad'
    assert '-shortest' in command


@pytest.fixture
def audio_stage(tmp_path, monkeypatch, audio_config):
    path = Path(__file__).parent / '10_audio_mixing.py'
    spec = importlib.util.spec_from_file_location('audio_stage', path)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    dirs = setup_directories(tmp_path / 'outputs', 'comedy')
    checkpoint = Mock()
    checkpoint.is_stage_completed.side_effect = lambda name, *args: name != 'audio_mixing'
    checkpoint.get_stats.return_value = {
        'completed_stages': 6, 'total_stages': 15, 'progress_percent': 40,
        'completed_list': [],
    }
    monkeypatch.setattr(stage, 'initialize_stage', lambda *args: (
        dirs['base'], dirs, checkpoint, logging.getLogger('audio-stage-test')))
    monkeypatch.setattr(stage, 'load_config', lambda *args: audio_config)
    monkeypatch.setattr(sys, 'argv', [str(path), '--input', 'movie.mp4', '--genre', 'comedy'])
    return stage, dirs, checkpoint


def test_finalization_uses_genre_artifacts_and_isolates_temp_files(audio_stage, monkeypatch):
    stage, dirs, checkpoint = audio_stage
    genre_dir = dirs['genre_output']
    timeline = {'actual_duration': 12.5, 'timeline': []}
    (genre_dir / 'trailer_timeline.json').write_text(json.dumps(timeline), encoding='utf-8')
    assembled = genre_dir / 'trailer_comedy_assembled.mp4'
    assembled.touch()
    final = genre_dir / 'trailer_comedy_final.mp4'
    mixer = Mock()

    def render(**kwargs):
        assert kwargs['timeline'] == timeline
        assert kwargs['video_path'] == assembled
        assert kwargs['output_path'] == final
        final.write_bytes(b'generated trailer')
        return str(final)

    mixer.mix_audio.side_effect = render
    factory = Mock(return_value=mixer)
    monkeypatch.setattr(stage, 'AudioMixer', factory)
    stage.main()
    assert factory.call_args.kwargs['output_dir'] == genre_dir
    assert checkpoint.mark_stage_completed.call_args.args[1]['duration'] == 12.5
    assert checkpoint.mark_stage_completed.call_args.kwargs['genre'] == 'comedy'


def test_shared_timeline_is_not_used_for_another_genre(audio_stage, monkeypatch):
    stage, dirs, checkpoint = audio_stage
    (dirs['output'] / 'trailer_timeline.json').write_text('{}', encoding='utf-8')
    factory = Mock()
    monkeypatch.setattr(stage, 'AudioMixer', factory)
    with pytest.raises(SystemExit) as error:
        stage.main()
    assert error.value.code == 1
    factory.assert_not_called()
    checkpoint.mark_stage_completed.assert_not_called()


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'),
                    reason='FFmpeg and FFprobe are required for the media smoke test')
@pytest.mark.parametrize('ducking', [True, False])
def test_short_music_preserves_rendered_video_duration(mixer, tmp_path, ducking):
    video = tmp_path / 'source.mp4'
    music = tmp_path / 'music.wav'
    final = tmp_path / 'final.mp4'
    subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=160x90:r=24',
        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
        '-t', '2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-y', str(video),
    ], check=True, capture_output=True)
    subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
        'sine=frequency=880:sample_rate=48000', '-t', '0.25', '-y', str(music),
    ], check=True, capture_output=True)
    mixer.enable_ducking = ducking
    mixer.mix_audio({'actual_duration': 2}, video, final, str(music))
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json',
        str(final),
    ], check=True, capture_output=True, text=True)
    metadata = json.loads(result.stdout)
    assert float(metadata['format']['duration']) == pytest.approx(2, abs=0.1)
    assert {stream['codec_type'] for stream in metadata['streams']} == {'video', 'audio'}
    decoded = subprocess.run([
        'ffmpeg', '-v', 'error', '-i', str(final), '-ss', '0.5', '-t', '1',
        '-vn', '-ac', '1', '-ar', '8000', '-f', 's16le', '-',
    ], check=True, capture_output=True).stdout
    samples = [sample[0] for sample in struct.iter_unpack('<h', decoded)]
    assert len(samples) >= 7900

    def amplitude(frequency):
        angle = 2 * math.pi * frequency / 8000
        return math.hypot(
            sum(value * math.cos(angle * index) for index, value in enumerate(samples)),
            sum(value * math.sin(angle * index) for index, value in enumerate(samples)),
        ) / len(samples)

    assert amplitude(440) > amplitude(880)
