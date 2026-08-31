"""Tests del punto de entrada de línea de comandos."""

from __future__ import annotations

import pytest

from tf_vis.app import main, parse_args


def test_defaults():
    args = parse_args([])
    assert args.model == 'vgg16'
    assert args.input_source == 'webcam'
    assert args.include_top is True
    assert args.input_size is None
    assert args.gpu is False


def test_no_top_flag():
    assert parse_args(['--no-top']).include_top is False


def test_input_size_is_parsed_as_width_height():
    assert parse_args(['--input-size', '320', '240']).input_size == [320, 240]


def test_list_models_prints_registry_and_exits(capsys):
    assert main(['--list-models']) == 0

    output = capsys.readouterr().out
    assert 'vgg16' in output
    assert 'inception_v3' in output
    assert '(299, 299, 3)' in output


def test_unknown_model_reports_available_options(capsys):
    assert main(['--model', 'alexnet']) == 1
    assert 'Modelos disponibles' in capsys.readouterr().out


def test_missing_model_file_is_reported(capsys, tmp_path):
    assert main(['--model-file', str(tmp_path / 'nope.keras')]) == 1
    assert 'Error al cargar el modelo' in capsys.readouterr().out


def test_gpu_flag_controls_device_visibility(monkeypatch):
    from tf_vis.app import _configure_devices

    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    _configure_devices(use_gpu=False)

    import os

    # Sin --gpu se fuerza la CPU aunque el entorno ya declare una GPU visible.
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'


def test_invalid_input_source_is_reported(capsys, monkeypatch):
    import keras

    import tf_vis.utils.misc as misc

    def fake_spec(name):
        inputs = keras.Input((8, 8, 3))
        outputs = keras.layers.Conv2D(2, 3)(inputs)
        return misc.ModelSpec(
            name='fake',
            constructor=lambda **kwargs: keras.Model(inputs, outputs),
            preprocess=lambda img: img,
            input_shape=(8, 8, 3),
        )

    monkeypatch.setattr(misc, 'get_model_spec', fake_spec)

    assert main(['--input-source', 'ftp://algo']) == 1
    assert 'Error al configurar la fuente de entrada' in capsys.readouterr().out


@pytest.mark.parametrize('weights', ['none', 'NONE'])
def test_weights_none_is_case_insensitive(weights):
    # Solo se comprueba el parseo; cargar pesos reales descargaría cientos de MB.
    assert parse_args(['--weights', weights]).weights == weights
