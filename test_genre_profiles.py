"""Regression coverage for the product's shared genre registry."""

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

from pipeline_common import (
    ALL_GENRES,
    GENRE_PROFILES_PATH,
    add_genre_arguments,
    load_genre_profile,
    normalize_genre,
)


def test_advertised_genres_match_profiles():
    with GENRE_PROFILES_PATH.open(encoding='utf-8') as stream:
        profiles = yaml.safe_load(stream)
    assert set(ALL_GENRES) == set(profiles)
    assert len(ALL_GENRES) == 27


@pytest.mark.parametrize('genre', ALL_GENRES)
def test_shared_cli_accepts_every_profile(genre):
    parser = argparse.ArgumentParser()
    add_genre_arguments(parser)
    args = parser.parse_args(['--input', 'movie.mp4', '--genre', f' {genre.upper()} '])
    assert args.genre == genre
    assert load_genre_profile(genre)['name'] == genre


@pytest.mark.parametrize('genre', ['', 'typo', '../thriller'])
def test_unknown_genres_fail_instead_of_falling_back(genre):
    with pytest.raises(ValueError, match='Unknown genre'):
        normalize_genre(genre)
    with pytest.raises(ValueError, match='Unknown genre'):
        load_genre_profile(genre)


def test_profiles_load_independently_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_genre_profile(' Western ')['name'] == 'western'


def test_loading_profile_does_not_share_mutable_data():
    profile = load_genre_profile('comedy')
    profile['music_tags'].clear()
    assert load_genre_profile('comedy')['music_tags']


@pytest.mark.parametrize('contents', ['', '[]', 'comedy: null', 'Comedy: {pacing: fast}'])
def test_invalid_profile_registry_is_rejected(tmp_path, contents):
    path = tmp_path / 'profiles.yaml'
    path.write_text(contents, encoding='utf-8')
    with pytest.raises(ValueError, match='[Gg]enre profile'):
        load_genre_profile('comedy', str(path))


@pytest.mark.parametrize(
    'script,parser_name,required',
    [
        ('12_beat_sheet_generator.py', 'parse_arguments', ['--movie-name', 'Movie']),
        ('13_embedding_generator.py', 'parse_args', ['--input', 'movie.mp4']),
        ('14_scene_retrieval.py', 'parse_args', ['--input', 'movie.mp4']),
        ('15_timeline_constructor.py', 'parse_args', ['--input', 'movie.mp4']),
    ],
)
def test_semantic_stage_parsers_share_registry(script, parser_name, required, monkeypatch):
    path = Path(__file__).parent / script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = getattr(module, parser_name)
    for genre in ALL_GENRES:
        monkeypatch.setattr(sys, 'argv', [script, *required, '--genre', genre.upper()])
        assert parser().genre == genre
    monkeypatch.setattr(sys, 'argv', [script, *required, '--genre', 'typo'])
    with pytest.raises(SystemExit) as error:
        parser()
    assert error.value.code == 2
