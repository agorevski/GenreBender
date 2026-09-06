#!/usr/bin/env python3
"""Run or preview the shared preprocessing and per-genre trailer pipeline.

Input paths are relative to the pipeline configuration file. Stage scripts,
runtime settings, assets and generated outputs are rooted at this repository.
Use --dry-run to preview the plan without media processing or network calls.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Thread
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml

from pipeline_common import ALL_GENRES, get_output_base_dir, get_story_graph_dir


PROJECT_ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_ROOT / 'trailer_generator' / 'config' / 'settings.yaml'
DIAGNOSTIC_LIMIT = 8192
PHASE1_CHECKPOINT_STAGES = (
    'shot_detection', 'keyframe_extraction', 'audio_extraction',
    'subtitle_management', 'remote_analysis',
)


def _phase1_plan(video_path, movie_name, synopsis_path, srt_path, force=False):
    common = ['--input', video_path]
    stages = [
        ('1_shot_detection.py', common.copy()),
        ('2_keyframe_extraction.py', common.copy()),
        ('3_audio_extraction.py', common.copy()),
        ('4_subtitle_management.py', common + ['--srt-file', srt_path]),
        ('5_remote_analysis.py', common.copy()),
        ('11_story_graph_generator.py', [
            '--movie-name', movie_name, '--synopsis', synopsis_path,
            '--srt-file', srt_path,
        ]),
    ]
    return [(script, args + (['--force'] if force else [])) for script, args in stages]


def _genre_plan(genre, video_path, movie_name, force=False):
    stages = [
        ('12_beat_sheet_generator.py', ['--movie-name', movie_name, '--genre', genre]),
        ('13_embedding_generator.py', [
            '--input', video_path, '--genre', genre, '--movie-name', movie_name,
        ]),
        ('14_scene_retrieval.py', [
            '--input', video_path, '--genre', genre, '--movie-name', movie_name,
        ]),
        ('15_timeline_constructor.py', ['--input', video_path, '--genre', genre]),
        ('9_video_assembly.py', ['--input', video_path, '--genre', genre]),
        ('10_audio_mixing.py', ['--input', video_path, '--genre', genre]),
    ]
    return [(script, args + (['--force'] if force else [])) for script, args in stages]


def _output_base(video_path):
    return PROJECT_ROOT / get_output_base_dir(video_path)


def _trailer_path(video_path, genre):
    return _output_base(video_path) / 'trailers' / genre / f'trailer_{genre}_final.mp4'


def run_stage_script(
    script_name: str,
    args: List[str],
    log_prefix: str = "",
    stream_output: bool = True,
    *,
    timeout_seconds: float = 7200,
) -> Tuple[bool, str]:
    """Run a stage with a bounded combined-output tail and a timeout in both modes."""
    script_path = PROJECT_ROOT / script_name
    command = [sys.executable, str(script_path), *args]
    output_tail = deque(maxlen=20)
    read_errors = []
    label = f"{log_prefix}{script_path.name}"

    try:
        process = subprocess.Popen(
            command, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8',
            errors='replace', bufsize=1,
            env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'},
        )
    except OSError as exc:
        return False, f"{label}: could not start ({exc})"

    def read_output():
        try:
            # Limit individual reads too: media tools may emit enormous lines.
            for chunk in iter(lambda: process.stdout.readline(4096), ''):
                output_tail.append(chunk)
                if stream_output:
                    print(f"{log_prefix}{chunk}", end='', flush=True)
        except (OSError, UnicodeError) as exc:
            read_errors.append(exc)

    reader = Thread(target=read_output, daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait()
    except KeyboardInterrupt:
        process.kill()
        process.wait()
        raise
    finally:
        reader.join(timeout=5)
        if not reader.is_alive():
            process.stdout.close()

    context = ''.join(list(output_tail))[-DIAGNOSTIC_LIMIT:].strip() or 'No output captured.'
    if timed_out:
        return False, f"{label}: timed out after {timeout_seconds:g} seconds\n{context}"
    if read_errors:
        return False, f"{label}: could not read output ({read_errors[0]})\n{context}"
    if reader.is_alive():
        return False, f"{label}: output pipe did not close after process exit\n{context}"
    if process.returncode != 0:
        return False, f"{label}: failed (exit code {process.returncode})\n{context}"
    return True, f"{label}: completed successfully"


def run_genre_pipeline(
    genre: str,
    video_path: str,
    movie_name: str,
    config_path: str = 'trailer_generator/config/settings.yaml',
    stream_output: bool = True,
    *,
    force: bool = False,
) -> Tuple[str, bool, str]:
    """Run Phase 2 for one genre; config_path is retained for caller compatibility.

    Semantic stages use repository settings and do not accept a --config flag.
    """
    prefix = f"[{genre.upper()}] "
    for script, args in _genre_plan(genre, video_path, movie_name, force):
        if stream_output:
            print(f"{prefix}Running {script}...")
        success, message = run_stage_script(
            script, args, prefix, stream_output=stream_output,
        )
        if stream_output:
            print(message)
        if not success:
            return genre, False, message

    trailer_path = _trailer_path(video_path, genre)
    try:
        _require_file(trailer_path, 'Final trailer')
    except ValueError as exc:
        return genre, False, f"{prefix}Stages exited successfully, but {exc}"
    return genre, True, f"{prefix}Trailer ready: {trailer_path}"


def run_phase1_stages(
    video_path: str,
    movie_name: str,
    synopsis_path: str,
    srt_path: str,
    force: bool = False,
) -> bool:
    """Run shared stages sequentially, stopping when a prerequisite fails."""
    print("\nPHASE 1: Shared preprocessing")
    for script, args in _phase1_plan(video_path, movie_name, synopsis_path, srt_path, force):
        print(f"\nRunning {script}...")
        success, message = run_stage_script(script, args)
        print(message)
        if not success:
            return False
    return True


def _record_result(results, errors, genre, success, message):
    results[genre] = success
    if not success and errors is not None:
        errors[genre] = message
    print(f"\n{'OK' if success else 'FAILED'} {genre.upper()}: {message}")


def run_phase2_sequential(
    genres: List[str],
    video_path: str,
    movie_name: str,
    *,
    force: bool = False,
    errors: Optional[Dict[str, str]] = None,
) -> Dict[str, bool]:
    """Process every genre, retaining failures without blocking unrelated genres."""
    print(f"\nPHASE 2: Sequential processing ({len(genres)} genres)")
    return _run_genres(
        genres, video_path, movie_name, 1, True, force, errors,
    )


def run_phase2_parallel(
    genres: List[str],
    video_path: str,
    movie_name: str,
    parallel_workers: int = 4,
    *,
    force: bool = False,
    errors: Optional[Dict[str, str]] = None,
) -> Dict[str, bool]:
    """Run isolated stage subprocesses concurrently and collect every outcome."""
    _validate_workers(parallel_workers)
    print(f"\nPHASE 2: Parallel processing ({len(genres)} genres, {parallel_workers} workers)")
    return _run_genres(
        genres, video_path, movie_name, parallel_workers, False, force, errors,
    )


def _run_genres(genres, video_path, movie_name, workers, stream_output, force, errors):
    results = {}
    # Work is already isolated in subprocesses; threads avoid another process layer.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_genre_pipeline, genre, video_path, movie_name,
                stream_output=stream_output, force=force,
            ): genre
            for genre in genres
        }
        for future in as_completed(futures):
            genre = futures[future]
            error = future.exception()
            if error is not None:
                success = False
                message = f"Worker {type(error).__name__}: {str(error)[-DIAGNOSTIC_LIMIT:]}"
            else:
                _, success, message = future.result()
            _record_result(results, errors, genre, success, message)
    return {genre: results[genre] for genre in genres}


def _read_mapping(path, label):
    try:
        with path.open(encoding='utf-8-sig') as stream:
            value = yaml.safe_load(stream)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read {label} {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, 'problem_mark', None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ''
        raise ValueError(f"Invalid YAML in {label} {path}{location}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty YAML mapping: {path}")
    return value


def _require_file(path, label):
    if not path.is_file():
        raise ValueError(f"{label} file not found or not a regular file: {path}")
    try:
        with path.open('rb') as stream:
            if not stream.read(1):
                raise ValueError(f"{label} file is empty: {path}")
    except OSError as exc:
        raise ValueError(f"Cannot read {label} file {path}: {exc}") from exc


def _validate_workers(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("parallel_workers must be a positive integer")
    return value


def _normalize_genres(values):
    if not isinstance(values, list) or not values:
        raise ValueError("Genres must be a non-empty list (or comma-separated --genres)")
    genres = []
    for value in values:
        if not isinstance(value, str) or value.strip().lower() not in ALL_GENRES:
            raise ValueError(f"Invalid genre {value!r}. Valid genres: {', '.join(ALL_GENRES)}")
        genre = value.strip().lower()
        if genre not in genres:
            genres.append(genre)
    return genres


def _load_request(args):
    config_path = Path(args.config).expanduser().resolve()
    config = _read_mapping(config_path, 'Pipeline config')
    movies = config.get('movies')
    if not isinstance(movies, dict) or not movies:
        raise ValueError("Pipeline config must contain a non-empty 'movies' mapping")
    if args.config_key not in movies:
        available = ', '.join(str(key) for key in movies)
        raise ValueError(f"Unknown movie key {args.config_key!r}. Available movie keys: {available}")
    movie = movies[args.config_key]
    if not isinstance(movie, dict):
        raise ValueError(f"Movie {args.config_key!r} must be a mapping")
    movie = movie.copy()
    for field in ('movie_name', 'video', 'synopsis', 'subtitles'):
        if not isinstance(movie.get(field), str) or not movie[field].strip():
            raise ValueError(f"Movie {args.config_key!r} requires a non-empty '{field}' string")
    movie['movie_name'] = movie['movie_name'].strip()
    for field in ('video', 'synopsis', 'subtitles'):
        path = (config_path.parent / Path(movie[field]).expanduser()).resolve()
        _require_file(path, field.capitalize())
        movie[field] = str(path)
    settings = config.get('settings', {})
    if not isinstance(settings, dict):
        raise ValueError("Pipeline 'settings' must be a mapping")
    configured_workers = _validate_workers(settings.get('parallel_workers', 4))
    workers = _validate_workers(
        args.parallel_workers if args.parallel_workers is not None else configured_workers,
    )
    genres = _normalize_genres(
        args.genres.split(',') if args.genres is not None else movie.get('genres', ['thriller']),
    )
    return movie, genres, workers


def _resume_issues(video_path, movie_name):
    issues = []
    paths = [
        (PROJECT_ROOT / get_story_graph_dir(movie_name) / 'story_graph.json', 'Story graph'),
        (_output_base(video_path) / 'shots' / 'shot_metadata.json', 'Shot metadata'),
        (_output_base(video_path) / 'checkpoint.json', 'Checkpoint'),
    ]
    for path, label in paths:
        try:
            _require_file(path, label)
            with path.open(encoding='utf-8-sig') as stream:
                data = json.load(stream)
            if not isinstance(data, (dict, list)) or not data:
                raise ValueError(f"{label} must contain non-empty JSON data: {path}")
            if label == 'Checkpoint':
                stages = data.get('stages', {}) if isinstance(data, dict) else {}
                missing = [
                    stage for stage in PHASE1_CHECKPOINT_STAGES
                    if not isinstance(stages, dict)
                    or not isinstance(stages.get(stage), dict)
                    or stages[stage].get('completed') is not True
                ]
                if missing:
                    issues.append(f"Checkpoint has incomplete shared stages: {', '.join(missing)}")
        except (ValueError, OSError) as exc:
            issues.append(f"--skip-phase1: {exc}. Run without --skip-phase1 to rebuild prerequisites.")
    return issues


def _readiness_issues(movie, skip_phase1):
    """Check only local prerequisites; never initialize clients or execute tools."""
    issues = []
    for tool in ('ffmpeg', 'ffprobe'):
        if shutil.which(tool) is None:
            issues.append(f"{tool} is not on PATH; install FFmpeg before generating trailers.")
    settings = _read_mapping(SETTINGS_PATH, 'Runtime settings')
    azure = settings.get('azure_openai')
    if not isinstance(azure, dict):
        issues.append("Runtime settings require an 'azure_openai' mapping.")
    else:
        azure = azure.copy()
        for key, env_name in (('endpoint', 'AZURE_OPENAI_ENDPOINT'), ('api_key', 'AZURE_OPENAI_KEY')):
            if os.getenv(env_name):
                azure[key] = os.environ[env_name]
        key = azure.get('api_key')
        if isinstance(key, str) and key.startswith('${') and key.endswith('}'):
            azure['api_key'] = os.getenv(key[2:-1])
        for field in ('endpoint', 'api_key', 'deployment_name', 'api_version'):
            value = azure.get(field)
            if (not isinstance(value, str) or not value.strip()
                    or value.strip().lower().startswith(('your', '${', 'changeme'))):
                issues.append(f"Configure azure_openai.{field} in runtime settings"
                              + (" or AZURE_OPENAI_KEY." if field == 'api_key'
                                 else " or AZURE_OPENAI_ENDPOINT." if field == 'endpoint' else "."))
        endpoint = azure.get('endpoint')
        if isinstance(endpoint, str) and endpoint:
            parsed = urlparse(endpoint)
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                    or 'yourendpoint' in parsed.hostname.lower()):
                issues.append("azure_openai.endpoint must be a configured HTTP(S) endpoint.")
    if skip_phase1:
        issues.extend(_resume_issues(movie['video'], movie['movie_name']))
    return issues


def _print_plan(movie, genres, workers, args):
    print("\nGenreBender: Multi-Genre Trailer Pipeline")
    print(f"Movie: {movie['movie_name']}")
    print(f"Video: {movie['video']}")
    print(f"Genres: {', '.join(genres)}")
    print(f"Mode: {'sequential' if args.sequential else 'parallel'}; workers: "
          f"{1 if args.sequential else workers}")
    print(f"Force re-run: {args.force}")
    print(f"Stage working directory: {PROJECT_ROOT}")
    print(f"Runtime settings: {SETTINGS_PATH}")
    print(f"Output location: {_output_base(movie['video'])}")
    if not args.dry_run:
        return
    print("\nDRY RUN: no stages, media tools, API calls or output writes.")
    phase1 = [] if args.skip_phase1 else _phase1_plan(
        movie['video'], movie['movie_name'], movie['synopsis'], movie['subtitles'], args.force,
    )
    print("\nPhase 1: " + ('SKIPPED (existing prerequisites required)' if args.skip_phase1
                            else 'shared preprocessing'))
    for script, command_args in phase1:
        print("  " + subprocess.list2cmdline([sys.executable, str(PROJECT_ROOT / script), *command_args]))
    for genre in genres:
        print(f"\nPhase 2: {genre}")
        for script, command_args in _genre_plan(genre, movie['video'], movie['movie_name'], args.force):
            print("  " + subprocess.list2cmdline([sys.executable, str(PROJECT_ROOT / script), *command_args]))
        print(f"  Expected trailer: {_trailer_path(movie['video'], genre)}")


def main(*, argv=None) -> None:
    """Validate, preview or execute the pipeline; exit nonzero for any failed genre."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config_key', help='Movie key from the pipeline config')
    parser.add_argument('--parallel-workers', type=int, default=None,
                        help='Positive worker count (default: settings.parallel_workers, then 4)')
    parser.add_argument('--sequential', action='store_true', help='Process genres one at a time with streamed logs')
    parser.add_argument('--skip-phase1', action='store_true', help='Reuse existing shared preprocessing artifacts')
    parser.add_argument('--force', action='store_true', help='Force re-run of every selected stage')
    parser.add_argument('--genres', help='Comma-separated genres overriding the movie config')
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'config.yaml'),
                        help='Pipeline YAML; input paths are relative to this file (default: repository config.yaml)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview commands and outputs offline; report readiness warnings without generating')
    args = parser.parse_args(argv)
    try:
        movie, genres, workers = _load_request(args)
        plans = _genre_plan(genres[0], movie['video'], movie['movie_name'])
        if not args.skip_phase1:
            plans += _phase1_plan(
                movie['video'], movie['movie_name'], movie['synopsis'], movie['subtitles'],
            )
        for script, _ in plans:
            _require_file(PROJECT_ROOT / script, 'Stage script')
        issues = _readiness_issues(movie, args.skip_phase1)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    _print_plan(movie, genres, workers, args)
    if issues:
        print("\nReadiness issues (resolve before generation; cloud connectivity was not tested):")
        for issue in issues:
            print(f"  - {issue}")
    if args.dry_run:
        print("\nPreview complete. " + ("Generation is NOT ready." if issues
                                       else "Local checks passed; cloud connectivity was not tested."))
        sys.exit(0)
    if issues:
        sys.exit(1)

    start_time = datetime.now()
    if not args.skip_phase1 and not run_phase1_stages(
        movie['video'], movie['movie_name'], movie['synopsis'], movie['subtitles'], args.force,
    ):
        print("\nPIPELINE SUMMARY: shared preprocessing failed.")
        for genre in genres:
            print(f"  NOT RUN {genre}: Phase 1 prerequisite failed")
        sys.exit(1)

    errors = {}
    if args.sequential:
        results = run_phase2_sequential(
            genres, movie['video'], movie['movie_name'], force=args.force, errors=errors,
        )
    else:
        results = run_phase2_parallel(
            genres, movie['video'], movie['movie_name'], workers, force=args.force, errors=errors,
        )
    print(f"\nPIPELINE SUMMARY ({(datetime.now() - start_time).total_seconds() / 60:.1f} minutes)")
    for genre in genres:
        if results.get(genre):
            print(f"  OK {genre}: {_trailer_path(movie['video'], genre)}")
        else:
            print(f"  FAILED {genre}: {errors.get(genre, 'No successful result was returned')}")
    completed = sum(bool(results.get(genre)) for genre in genres)
    print(f"\nCompleted: {completed}/{len(genres)} genres")
    if completed != len(genres):
        failed = ','.join(genre for genre in genres if not results.get(genre))
        print(f"After fixing errors, retry only failed genres with --skip-phase1 --genres {failed}")
    sys.exit(0 if completed == len(genres) else 1)


if __name__ == '__main__':
    main()
