"""Tests de las utilidades de introspección, imagen y registro de modelos."""

from __future__ import annotations

import numpy as np
import pytest

from tf_vis.utils.image_utils import (
    apply_gradcam,
    create_grid_of_images,
    deprocess_image,
    load_image,
    overlay_gradcam,
    preprocess_image_for_model,
)
from tf_vis.utils.layers import (
    describe_layer,
    is_spatial_shape,
    layer_num_units,
    layer_output_shape,
)
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


def test_layer_output_shape_falls_back_to_legacy_attributes():
    class LegacyLayer:
        """Capa al estilo de Keras 2, sin tensor `output` construido."""

        output = None
        batch_input_shape = (None, 8, 8, 3)

    assert layer_output_shape(LegacyLayer()) == (None, 8, 8, 3)


def test_layer_output_shape_returns_none_when_unknown():
    class Opaque:
        output = None

    assert layer_output_shape(Opaque()) is None
    assert layer_num_units(Opaque()) == 0


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


def test_apply_gradcam(toy_model, sample_image):
    heatmap = apply_gradcam(toy_model, sample_image, 'conv2', class_idx=1)
    assert heatmap.shape == (16, 16)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


def test_apply_gradcam_defaults_to_the_predicted_class(toy_model, sample_image):
    predicted = int(np.argmax(toy_model.predict(sample_image[None], verbose=0)[0]))
    assert np.allclose(
        apply_gradcam(toy_model, sample_image, 'conv2'),
        apply_gradcam(toy_model, sample_image, 'conv2', class_idx=predicted),
    )


def test_apply_gradcam_never_returns_nan(toy_model):
    # Un mapa de calor íntegramente nulo dividía por cero y producía NaN.
    heatmap = apply_gradcam(toy_model, np.zeros((32, 32, 3), dtype=np.float32), 'conv1', 0)
    assert np.isfinite(heatmap).all()


def test_overlay_gradcam(sample_image):
    heatmap = np.linspace(0, 1, 256).reshape(16, 16).astype(np.float32)
    overlay = overlay_gradcam(sample_image, heatmap)
    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8


def test_imagenet_labels_use_the_keras_class_order(tmp_path, monkeypatch):
    """
    El archivo `ImageNetLabels.txt` de TensorFlow tiene 1001 entradas por incluir
    una clase de fondo, así que usarlo desplazaba todas las etiquetas un índice.
    """
    import json

    import keras

    import tf_vis.utils.misc as misc

    index_file = tmp_path / 'imagenet_class_index.json'
    index_file.write_text(json.dumps({
        '0': ['n01440764', 'tench'],
        '1': ['n01443537', 'goldfish'],
        '2': ['n01484850', 'great_white_shark'],
    }))

    monkeypatch.setattr(misc, '_IMAGENET_LABELS', None)
    monkeypatch.setattr(keras.utils, 'get_file', lambda *args, **kwargs: str(index_file))

    labels = misc.get_imagenet_labels()

    assert labels == ['tench', 'goldfish', 'great_white_shark']
    # La segunda llamada se sirve de la caché en memoria.
    assert misc.get_imagenet_labels() == labels


def test_predict_image_maps_indices_to_labels(toy_model, sample_image, monkeypatch):
    import tf_vis.utils.misc as misc

    # Etiquetar con ImageNet requiere descargar el índice de clases de Keras.
    monkeypatch.setattr(misc, 'get_imagenet_labels', lambda name=None: [
        f'clase_{i}' for i in range(1000)
    ])

    results = misc.predict_image(toy_model, sample_image, top_k=3)

    assert len(results) == 3
    probabilities = [prob for _, _, prob in results]
    assert probabilities == sorted(probabilities, reverse=True)
    for idx, label, _ in results:
        assert label == f'clase_{idx}'


def test_predict_image_falls_back_for_out_of_range_indices(toy_model, sample_image, monkeypatch):
    import tf_vis.utils.misc as misc

    monkeypatch.setattr(misc, 'get_imagenet_labels', lambda name=None: ['solo_una'])
    results = misc.predict_image(toy_model, sample_image, top_k=2)

    assert all(label.startswith(('solo_una', 'class_')) for _, label, _ in results)


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
