"""Checkpoint transactions must preserve progress across independent workers."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import threading
import time

import pytest

from trailer_generator import checkpoint as checkpoint_module
from trailer_generator.checkpoint import CheckpointFormatError, CheckpointManager


def _process_writer(path, worker, ready):
    """Top-level target so this exercises Windows' spawn, not inherited memory."""
    manager = CheckpointManager(Path(path))
    original_write = manager._write

    def delayed_write(data):
        time.sleep(0.02)
        original_write(data)

    manager._write = delayed_write
    ready.wait(timeout=30)
    genre = f'genre-{worker}'
    for stage in manager.GENRE_DEPENDENT_STAGES:
        manager.mark_stage_completed(stage, {'worker': worker}, genre=genre)
    manager.mark_stage_completed(manager.GENRE_AGNOSTIC_STAGES[worker])


def test_independent_process_transactions_preserve_all_progress(tmp_path):
    path = tmp_path / 'checkpoint.json'
    seed = CheckpointManager(path)
    seed.set_metadata('film-\u00e9.mp4', 'thriller')
    seed.mark_stage_completed('audio_mixing', {'duration': 90}, genre='existing')
    seed.data['extension'] = {'source': 'preserve me'}
    seed.save()
    created_at = seed.data['created_at']
    context = multiprocessing.get_context('spawn')
    ready = context.Barrier(5)
    workers = [
        context.Process(target=_process_writer, args=(str(path), index, ready))
        for index in range(4)
    ]
    try:
        for process in workers:
            process.start()
        ready.wait(timeout=30)
        for process in workers:
            process.join(timeout=30)
        assert [process.exitcode for process in workers] == [0] * 4
    finally:
        for process in workers:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=10)

    result = CheckpointManager(path)
    assert result.data['input_file'] == 'film-\u00e9.mp4'
    assert result.data['genre'] == 'thriller'
    assert result.data['created_at'] == created_at
    assert result.data['extension'] == {'source': 'preserve me'}
    assert result.is_stage_completed('audio_mixing', 'existing')
    for worker in range(4):
        assert result.is_stage_completed(result.GENRE_AGNOSTIC_STAGES[worker])
        for stage in result.GENRE_DEPENDENT_STAGES:
            assert result.is_stage_completed(stage, f'genre-{worker}')
            assert result.data['genre_stages'][f'genre-{worker}'][stage]['worker'] == worker
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('shared_manager', [False, True])
def test_thread_transactions_with_shared_or_separate_managers(tmp_path, shared_manager):
    path = tmp_path / 'checkpoint.json'
    seed = CheckpointManager(path)
    seed.set_metadata('film.mp4')
    managers = [seed if shared_manager else CheckpointManager(path) for _ in range(8)]
    ready = threading.Barrier(len(managers))

    def write(worker):
        ready.wait(timeout=10)
        managers[worker].mark_stage_completed('scene_retrieval', genre=f'genre-{worker}')

    with ThreadPoolExecutor(max_workers=len(managers)) as pool:
        list(pool.map(write, range(len(managers))))
    result = CheckpointManager(path)
    assert set(result.get_completed_genres('scene_retrieval')) == {
        f'genre-{worker}' for worker in range(len(managers))
    }
    assert result.data['input_file'] == 'film.mp4'


def test_stale_managers_preserve_shared_genre_and_input_metadata(tmp_path):
    path = tmp_path / 'checkpoint.json'
    first = CheckpointManager(path)
    stale = CheckpointManager(path)
    first.set_metadata('film.mp4', 'Comedy')
    first.mark_stage_completed('shot_detection', {'shots_count': 12})
    stale.mark_stage_completed('scene_retrieval', {'scenes': [1, 2]}, genre='THRILLER')
    first.mark_stage_completed('scene_retrieval', {'scenes': [3]}, genre='Comedy')
    stale.set_metadata('film.mp4')
    assert stale.data['input_file'] == 'film.mp4'
    assert stale.data['genre'] == 'Comedy'
    assert stale.data['stages']['shot_detection']['shots_count'] == 12
    assert stale.is_stage_completed('scene_retrieval', 'comedy')
    assert stale.is_stage_completed('scene_retrieval', 'thriller')
    assert stale.get_last_completed_stage() == 'shot_detection'
    assert stale.get_incomplete_genres('scene_retrieval', ['COMEDY', 'Thriller', 'horror']) == ['horror']


def test_save_merges_only_local_edits_and_deletions(tmp_path):
    path = tmp_path / 'checkpoint.json'
    first = CheckpointManager(path)
    first.set_metadata('film.mp4', 'comedy')
    first.mark_stage_completed('shot_detection', {'old': True})
    stale = CheckpointManager(path)
    first.mark_stage_completed('shot_detection', {'new': True})
    first.mark_stage_completed('scene_retrieval', genre='thriller')
    first.set_metadata('replacement.mp4', 'thriller')
    del stale.data['stages']['shot_detection']['old']
    stale.data['stages']['shot_detection']['local'] = 'edit'
    stale.save()
    result = CheckpointManager(path)
    assert result.data == stale.data
    assert result.data['input_file'] == 'replacement.mp4'
    assert result.data['genre'] == 'thriller'
    assert result.is_stage_completed('scene_retrieval', 'thriller')
    assert result.data['stages']['shot_detection']['new'] is True
    assert result.data['stages']['shot_detection']['local'] == 'edit'
    assert 'old' not in result.data['stages']['shot_detection']


def test_save_merges_concurrently_created_genre_sections(tmp_path):
    path = tmp_path / 'checkpoint.json'
    local = CheckpointManager(path)
    other = CheckpointManager(path)
    local.data['genre_stages']['comedy'] = {'scene_retrieval': {'completed': True}}
    other.mark_stage_completed('embedding_generation', genre='comedy')
    local.save()
    assert local.is_stage_completed('scene_retrieval', 'comedy')
    assert local.is_stage_completed('embedding_generation', 'comedy')


def test_reset_genre_uses_latest_state_and_preserves_other_work(tmp_path):
    path = tmp_path / 'checkpoint.json'
    writer = CheckpointManager(path)
    writer.set_metadata('film.mp4', 'thriller')
    writer.mark_stage_completed('shot_detection')
    stale = CheckpointManager(path)
    writer.mark_stage_completed('scene_retrieval', genre='comedy')
    writer.mark_stage_completed('scene_retrieval', genre='thriller')
    stale.reset_genre('COMEDY')
    assert 'comedy' not in CheckpointManager(path).data['genre_stages']
    writer.mark_stage_completed('audio_mixing', genre='thriller')
    result = CheckpointManager(path)
    assert not result.is_stage_completed('scene_retrieval', 'comedy')
    assert result.is_stage_completed('scene_retrieval', 'thriller')
    assert result.is_stage_completed('audio_mixing', 'thriller')
    assert result.is_stage_completed('shot_detection')
    assert result.data['input_file'] == 'film.mp4'
    assert result.data['genre'] == 'thriller'
    result.reset_genre('missing')
    assert result.is_stage_completed('scene_retrieval', 'thriller')


def test_full_reset_clears_metadata_without_stale_writer_resurrection(tmp_path):
    path = tmp_path / 'checkpoint.json'
    writer = CheckpointManager(path)
    writer.set_metadata('film.mp4', 'comedy')
    writer.mark_stage_completed('shot_detection')
    writer.mark_stage_completed('scene_retrieval', genre='comedy')
    stale = CheckpointManager(path)
    writer.reset()
    result = CheckpointManager(path)
    assert result.data['input_file'] is None
    assert result.data['genre'] is None
    assert result.data['last_completed_stage'] is None
    assert result.data['genre_stages'] == {}
    assert all(record == {'completed': False} for record in result.data['stages'].values())
    stale.save()
    stale.mark_stage_completed('scene_retrieval', genre='thriller')
    result.reload()
    assert not result.is_stage_completed('shot_detection')
    assert not result.is_stage_completed('scene_retrieval', 'comedy')
    assert result.is_stage_completed('scene_retrieval', 'thriller')
    assert result.data['input_file'] is None


def test_reset_of_absent_genre_does_not_write(tmp_path):
    path = tmp_path / 'checkpoint.json'
    manager = CheckpointManager(path)
    manager.reset_genre('missing')
    assert not path.exists()
    manager.save()
    original_bytes = path.read_bytes()
    manager.reset_genre('missing')
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize('version', [None, '1.0'])
def test_legacy_migration_preserves_data_and_adds_new_stages(tmp_path, version):
    path = tmp_path / 'checkpoint.json'
    legacy = {
        'created_at': '2020-01-01T00:00:00',
        'input_file': 'legacy.mp4',
        'genre': 'comedy',
        'last_completed_stage': 'shot_detection',
        'stages': {'shot_detection': {'completed': True, 'shots_count': 23}},
        'extension': {'retained': True},
    }
    if version is not None:
        legacy['version'] = version
    original = json.dumps(legacy)
    path.write_text(original, encoding='utf-8')
    manager = CheckpointManager(path)
    assert manager.data['version'] == '2.0'
    assert manager.is_stage_completed('shot_detection')
    assert not manager.is_stage_completed('story_graph_generation')
    assert path.read_text(encoding='utf-8') == original
    manager.mark_stage_completed('story_graph_generation')
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['version'] == '2.0'
    assert result['genre_stages'] == {}
    for field in ('created_at', 'input_file', 'genre', 'extension'):
        assert result[field] == legacy[field]
    assert result['stages']['shot_detection'] == legacy['stages']['shot_detection']


@pytest.mark.parametrize('contents', [
    b'',
    b'{"version": "2.0",',
    b'\xff',
    b'[]',
    b'null',
    b'{}',
    b'{"version": "3.0", "stages": {}, "genre_stages": {}}',
    b'{"version": "2.0", "stages": {}}',
    b'{"stages": []}',
    b'{"stages": {"shot_detection": true}}',
    b'{"stages": {"shot_detection": {"completed": "false"}}}',
    b'{"stages": {}, "genre_stages": {"comedy": []}}',
    b'{"stages": {}, "input_file": []}',
])
def test_corrupt_checkpoints_raise_and_preserve_original_bytes(tmp_path, contents):
    path = tmp_path / 'checkpoint.json'
    path.write_bytes(contents)
    with pytest.raises(CheckpointFormatError, match='Invalid checkpoint'):
        CheckpointManager(path)
    assert path.read_bytes() == contents


@pytest.mark.parametrize('operation', [
    lambda manager: manager.mark_stage_completed('shot_detection'),
    lambda manager: manager.set_metadata('film.mp4'),
    lambda manager: manager.save(),
    lambda manager: manager.reset(),
    lambda manager: manager.reset_genre('comedy'),
    lambda manager: manager.reload(),
])
def test_existing_manager_never_overwrites_later_corruption(tmp_path, operation):
    path = tmp_path / 'checkpoint.json'
    manager = CheckpointManager(path)
    before = deepcopy(manager.data)
    path.write_bytes(b'corrupted during another run')
    with pytest.raises(CheckpointFormatError):
        operation(manager)
    assert path.read_bytes() == b'corrupted during another run'
    assert manager.data == before


@pytest.mark.parametrize('failure_point', ['replace', 'fsync', 'dump'])
def test_write_failures_preserve_previous_state_and_release_locks(tmp_path, monkeypatch, failure_point):
    path = tmp_path / 'checkpoint.json'
    manager = CheckpointManager(path)
    manager.set_metadata('film.mp4')
    original_bytes = path.read_bytes()
    original_data = deepcopy(manager.data)

    def fail(*args, **kwargs):
        if failure_point == 'dump':
            args[1].write('{"partial":')
        raise OSError('simulated disk failure')

    with monkeypatch.context() as patch:
        target = checkpoint_module.json if failure_point == 'dump' else checkpoint_module.os
        patch.setattr(target, failure_point, fail)
        with pytest.raises(OSError, match='simulated disk failure'):
            manager.mark_stage_completed('shot_detection')
    assert path.read_bytes() == original_bytes
    assert manager.data == original_data
    assert not list(tmp_path.glob('*.tmp'))
    manager.mark_stage_completed('shot_detection')
    assert CheckpointManager(path).is_stage_completed('shot_detection')


def test_unserializable_metadata_does_not_truncate_checkpoint(tmp_path):
    path = tmp_path / 'checkpoint.json'
    manager = CheckpointManager(path)
    manager.save()
    original_bytes = path.read_bytes()
    with pytest.raises(TypeError):
        manager.mark_stage_completed('shot_detection', {'unsupported': object()})
    assert path.read_bytes() == original_bytes
    assert not manager.is_stage_completed('shot_detection')
    assert not list(tmp_path.glob('*.tmp'))


def test_replacement_uses_complete_fsynced_sibling_file(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.json'
    manager = CheckpointManager(path)
    manager.save()
    original_bytes = path.read_bytes()
    real_replace = checkpoint_module.os.replace
    real_fsync = checkpoint_module.os.fsync
    flushed = []
    replacements = []

    def record_fsync(fd):
        real_fsync(fd)
        flushed.append(fd)

    def inspect_replace(source, destination):
        assert flushed
        assert Path(source).parent == path.parent
        assert Path(destination) == path
        assert path.read_bytes() == original_bytes
        pending = json.loads(Path(source).read_text(encoding='utf-8'))
        assert pending['stages']['shot_detection']['completed']
        replacements.append(source)
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint_module.os, 'fsync', record_fsync)
    monkeypatch.setattr(checkpoint_module.os, 'replace', inspect_replace)
    manager.mark_stage_completed('shot_detection')
    assert len(replacements) == 1
    assert not Path(replacements[0]).exists()


def test_read_errors_are_not_treated_as_missing_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.json'
    manager = CheckpointManager(path)
    manager.save()
    original_bytes = path.read_bytes()
    original_open = Path.open

    def unreadable(self, *args, **kwargs):
        if self == path:
            raise PermissionError('checkpoint unreadable')
        return original_open(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, 'open', unreadable)
        with pytest.raises(PermissionError, match='checkpoint unreadable'):
            manager.mark_stage_completed('shot_detection')
    assert path.read_bytes() == original_bytes


def test_legacy_stage_and_cached_read_api_remain_available(tmp_path):
    path = tmp_path / 'checkpoint.json'
    manager = CheckpointManager(path)
    observer = CheckpointManager(path)
    manager.mark_stage_completed('genre_scoring', {'scores': [1]})
    manager.mark_stage_completed('video_assembly')
    manager.mark_stage_completed('not_a_stage')
    assert manager.is_stage_completed('genre_scoring')
    assert manager.is_stage_completed('video_assembly')
    assert not manager.is_stage_completed('video_assembly', 'comedy')
    assert not manager.is_stage_completed('not_a_stage')
    assert manager.should_skip_stage('genre_scoring')
    assert not manager.should_skip_stage('genre_scoring', force=True)
    assert manager.get_resume_stage('shot_detection') == 'shot_detection'
    assert not observer.is_stage_completed('genre_scoring')
    observer.reload()
    assert observer.is_stage_completed('genre_scoring')
