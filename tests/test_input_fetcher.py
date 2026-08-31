"""Tests de `InputFetcher`."""

from __future__ import annotations

import numpy as np
import pytest

from tf_vis.input_fetcher import InputFetcher


def test_rejects_unknown_source():
    with pytest.raises(ValueError, match='Fuente de entrada no válida'):
        InputFetcher(input_source='ftp://algo')


def test_directory_source_lists_images(image_dir):
    fetcher = InputFetcher(input_source=f'directory:{image_dir}', target_size=(16, 16))
    assert len(fetcher) == 3
    assert fetcher.is_live is False


def test_directory_source_requires_existing_path(tmp_path):
    with pytest.raises(ValueError, match='no existe'):
        InputFetcher(input_source=f'directory:{tmp_path / "nope"}')


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match='No se encontraron imágenes'):
        InputFetcher(input_source=f'directory:{tmp_path}')


def test_resizes_to_target_size_width_height(image_dir):
    # target_size es (ancho, alto); la imagen resultante debe ser (alto, ancho, 3).
    fetcher = InputFetcher(input_source=f'directory:{image_dir}', target_size=(24, 12))
    image = fetcher.get_current_image()
    assert image.shape == (12, 24, 3)


def test_iterates_and_wraps_around(image_dir):
    fetcher = InputFetcher(input_source=f'directory:{image_dir}', target_size=(8, 8))
    seen = [int(fetcher.get_next_image().mean()) for _ in range(4)]
    # Tres imágenes distintas y luego vuelta a empezar.
    assert seen[0] != seen[1] != seen[2]
    assert seen[3] == seen[0]


def test_previous_image_goes_backwards(image_dir):
    fetcher = InputFetcher(input_source=f'directory:{image_dir}', target_size=(8, 8))
    first = fetcher.get_specific_image(1)
    fetcher.get_next_image()
    assert np.array_equal(fetcher.get_previous_image(), first)


def test_out_of_range_index_raises(image_dir):
    fetcher = InputFetcher(input_source=f'directory:{image_dir}', target_size=(8, 8))
    with pytest.raises(IndexError):
        fetcher.get_specific_image(99)


def test_preprocessing_is_not_applied_twice(image_dir):
    calls = []

    def preprocess(img):
        calls.append(img.mean())
        return img - 100.0

    fetcher = InputFetcher(input_source=f'directory:{image_dir}',
                           preprocessing_function=preprocess, target_size=(8, 8))

    first = fetcher.get_specific_image(0)
    second = fetcher.get_specific_image(0)

    # Reprocesar la misma imagen no debe acumular la transformación.
    assert np.array_equal(first, second)
    assert calls[0] == pytest.approx(calls[1])


def test_raw_image_is_kept_unpreprocessed(image_dir):
    fetcher = InputFetcher(input_source=f'directory:{image_dir}',
                           preprocessing_function=lambda img: img - 1000.0,
                           target_size=(8, 8))
    processed = fetcher.get_specific_image(0)

    assert processed.min() < 0
    assert fetcher.current_raw_image.dtype == np.uint8
    assert fetcher.current_raw_image.min() >= 0


def test_file_source(image_dir):
    path = sorted(image_dir.glob('*.jpg'))[0]
    fetcher = InputFetcher(input_source=f'file:{path}', target_size=(8, 8))
    assert len(fetcher) == 1
    assert fetcher.get_next_image().shape == (8, 8, 3)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match='no existe'):
        InputFetcher(input_source=f'file:{tmp_path / "nope.jpg"}')


def test_synthetic_source_is_deterministic():
    first = InputFetcher(input_source='synthetic', target_size=(16, 16)).get_current_image()
    second = InputFetcher(input_source='synthetic', target_size=(16, 16)).get_current_image()
    assert np.array_equal(first, second)
    assert first.shape == (16, 16, 3)


def test_webcam_falls_back_to_synthetic_when_unavailable():
    # No hay cámara en CI, así que la fuente debe degradarse en lugar de fallar.
    fetcher = InputFetcher(input_source='webcam:99', target_size=(16, 16))
    assert fetcher.is_live is False
    assert fetcher.get_next_image().shape == (16, 16, 3)


def test_context_manager_closes(image_dir):
    with InputFetcher(input_source=f'directory:{image_dir}', target_size=(8, 8)) as fetcher:
        fetcher.get_next_image()
    assert fetcher.webcam is None
