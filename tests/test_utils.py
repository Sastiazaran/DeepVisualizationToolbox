"""Tests de las utilidades de introspección, imagen y registro de modelos."""

from __future__ import annotations

import numpy as np
import pytest

from tf_vis.utils.image_utils import (
    create_grid_of_images,
    deprocess_image,
    load_image,
    preprocess_image_for_model,
)
from tf_vis.utils.layers import describe_layer, is_spatial_shape, layer_num_units
from tf_vis.utils.misc import (
    count_params,
    create_model_summary,
    get_available_models,
    get_model_spec,
    get_model_specs,
    load_model,
    save_visualizations,
    time_execution,
)


# ----------------------------------------------------------------------
# Registro de modelos
# ----------------------------------------------------------------------
def test_registry_contains_expected_models():
    specs = get_model_specs()
    assert {'vgg16', 'resnet50', 'inception_v3', 'efficientnetv2_b0', 'convnext_tiny'} <= set(specs)


def test_model_spec_lookup_is_case_and_dash_insensitive():
    assert get_model_spec('MobileNet-V2').name == 'mobilenet_v2'


def test_unknown_model_raises():
    with pytest.raises(ValueError, match='no disponible'):
        get_model_spec('alexnet')


def test_load_model_rejects_unknown_name():
    with pytest.raises(ValueError, match='no disponible'):
        load_model('alexnet')


def test_legacy_dict_view_still_works():
    models = get_available_models()
    assert callable(models['vgg16']['preprocess'])
    assert models['vgg16']['input_shape'] == (224, 224, 3)


def test_inception_declares_its_native_input_size():
    assert get_model_spec('inception_v3').input_shape == (299, 299, 3)


# ----------------------------------------------------------------------
# Introspección de capas
# ----------------------------------------------------------------------
def test_describe_layer(toy_model):
    described = describe_layer(toy_model.get_layer('conv2'))
    assert described['name'] == 'conv2'
    assert described['type'] == 'Conv2D'
    assert described['units'] == 16
    assert described['params'] > 0


def test_layer_num_units_of_dense(toy_model):
    assert layer_num_units(toy_model.get_layer('predictions')) == 10


def test_is_spatial_shape():
    assert is_spatial_shape((None, 8, 8, 3)) is True
    assert is_spatial_shape((None, 10)) is False
    assert is_spatial_shape(None) is False


def test_create_model_summary(toy_model):
    summary = create_model_summary(toy_model)
    assert summary['name'] == 'toy_cnn'
    assert summary['total_params'] == toy_model.count_params()
    assert summary['trainable_params'] > 0
    conv1 = next(layer for layer in summary['layers'] if layer['name'] == 'conv1')
    assert conv1['filters'] == 8
    assert conv1['activation'] == 'relu'


def test_count_params(toy_model):
    assert count_params(toy_model.trainable_weights) == toy_model.count_params()


# ----------------------------------------------------------------------
# Utilidades de imagen
# ----------------------------------------------------------------------
def test_load_image_returns_rgb(image_dir):
    path = str(sorted(image_dir.glob('*.jpg'))[0])
    img = load_image(path, target_size=(16, 8))
    assert img.shape == (8, 16, 3)


def test_load_image_missing_file(tmp_path):
    with pytest.raises(ValueError, match='No se pudo cargar'):
        load_image(str(tmp_path / 'nope.jpg'))


def test_preprocess_expands_grayscale():
    gray = np.full((8, 8), 128, dtype=np.uint8)
    assert preprocess_image_for_model(gray, 'vgg16').shape == (8, 8, 3)


def test_preprocess_does_not_mutate_input():
    img = np.full((4, 4, 3), 200, dtype=np.float32)
    original = img.copy()
    preprocess_image_for_model(img, 'vgg16')
    assert np.array_equal(img, original)


def test_preprocess_unknown_model_scales_to_unit_range():
    img = np.full((4, 4, 3), 255, dtype=np.uint8)
    assert preprocess_image_for_model(img, 'desconocido').max() == pytest.approx(1.0)


def test_deprocess_roundtrip_for_caffe_models():
    img = np.full((4, 4, 3), 128, dtype=np.float32)
    restored = deprocess_image(preprocess_image_for_model(img, 'vgg16'), 'vgg16')
    assert np.allclose(restored, 128 / 255.0, atol=1e-4)


def test_deprocess_roundtrip_for_scaled_models():
    img = np.full((4, 4, 3), 200, dtype=np.float32)
    restored = deprocess_image(preprocess_image_for_model(img, 'mobilenet'), 'mobilenet')
    assert np.allclose(restored, 200 / 255.0, atol=1e-4)


def test_create_grid_of_images():
    images = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(4)]
    grid = create_grid_of_images(images, padding=1)
    assert grid.shape == (9, 9, 3)


def test_create_grid_requires_images():
    with pytest.raises(ValueError, match='al menos una imagen'):
        create_grid_of_images([])


# ----------------------------------------------------------------------
# Varios
# ----------------------------------------------------------------------
def test_time_execution():
    result, elapsed = time_execution(sum, [1, 2, 3])
    assert result == 6
    assert elapsed >= 0


def test_save_visualizations(tmp_path):
    written = save_visualizations(
        {'a': np.zeros((4, 4, 3), dtype=np.float32)}, str(tmp_path), prefix='run'
    )
    assert len(written) == 1
    assert (tmp_path / 'run_a.png').exists()
