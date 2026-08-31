"""Tests de los métodos de visualización."""

from __future__ import annotations

import numpy as np
import pytest

from tf_vis.visualization import (
    apply_gradient_ascent,
    create_class_activation_map,
    display_activation_grid,
    normalize_01,
    overlay_heatmap,
    resolve_ascent_size,
    visualize_layer_filters,
    visualize_max_activations,
)


def test_normalize_01_handles_constant_arrays():
    result = normalize_01(np.full((4, 4), 7.0))
    assert np.isfinite(result).all()
    assert result.max() == pytest.approx(0.0)


def test_display_activation_grid_layout():
    activations = np.random.default_rng(0).random((4, 5, 6))
    grid = display_activation_grid(activations, padding=1)
    # 4 filtros -> cuadrícula 2x2 con un píxel de separación.
    assert grid.shape == (2 * 5 + 1, 2 * 6 + 1)
    assert 0.0 <= grid.min() <= grid.max() <= 1.0


def test_display_activation_grid_rejects_wrong_rank():
    with pytest.raises(ValueError, match='n_filtros'):
        display_activation_grid(np.zeros((1, 4, 4, 3)))


def test_display_activation_grid_respects_explicit_size():
    grid = display_activation_grid(np.zeros((3, 4, 4)), grid_size=(1, 3), padding=0)
    assert grid.shape == (4, 12)


def test_visualize_layer_filters(toy_model):
    grid = visualize_layer_filters(toy_model, 'conv1')
    # 8 filtros de 3x3 -> cuadrícula 3x3 con padding de 1.
    assert grid.shape == (3 * 3 + 2, 3 * 3 + 2)


def test_visualize_layer_filters_rejects_non_conv(toy_model):
    with pytest.raises(ValueError, match='no es una capa convolucional'):
        visualize_layer_filters(toy_model, 'predictions')


def test_resolve_ascent_size_defaults_to_native_input(wrapper):
    assert resolve_ascent_size(wrapper, None) == (32, 32)


def test_resolve_ascent_size_rejects_mismatch_on_fixed_models(wrapper):
    # El modelo de juguete fija su entrada en 32x32, igual que un modelo con
    # include_top=True; pedir otro tamaño debe fallar con un mensaje claro.
    with pytest.raises(ValueError, match='include_top=False'):
        resolve_ascent_size(wrapper, (64, 64))


def test_resolve_ascent_size_allows_any_size_on_headless_models():
    import keras

    from tf_vis.model_wrapper import ModelWrapper

    inputs = keras.Input((None, None, 3))
    outputs = keras.layers.Conv2D(4, 3, name='conv')(inputs)
    headless = ModelWrapper(keras.Model(inputs, outputs))

    assert resolve_ascent_size(headless, (48, 64)) == (48, 64)
    assert resolve_ascent_size(headless, None) == (224, 224)


def test_gradient_ascent_uses_native_size_by_default(wrapper):
    assert apply_gradient_ascent(wrapper, 'conv1', 0, iterations=1).shape == (32, 32, 3)


def test_gradient_ascent_handles_small_feature_maps(wrapper):
    # `conv2` produce mapas de 16x16, pero el recorte de bordes no debe vaciarlos.
    image = apply_gradient_ascent(wrapper, 'conv2', 0, iterations=2, seed=0)
    assert np.isfinite(image).all()


def test_gradient_ascent_produces_valid_image(wrapper):
    image = apply_gradient_ascent(wrapper, 'conv2', filter_index=0, iterations=3,
                                  image_size=(32, 32), seed=0)
    assert image.shape == (32, 32, 3)
    assert 0.0 <= image.min() <= image.max() <= 1.0
    assert np.isfinite(image).all()


def test_gradient_ascent_is_reproducible_with_seed(wrapper):
    kwargs = {'iterations': 2, 'image_size': (32, 32), 'seed': 7}
    first = apply_gradient_ascent(wrapper, 'conv2', 1, **kwargs)
    second = apply_gradient_ascent(wrapper, 'conv2', 1, **kwargs)
    assert np.allclose(first, second)


def test_gradient_ascent_works_on_dense_layers(wrapper):
    image = apply_gradient_ascent(wrapper, 'predictions', filter_index=2, iterations=2,
                                  image_size=(32, 32), seed=0)
    assert image.shape == (32, 32, 3)


def test_gradcam_shape_and_range(wrapper, sample_image):
    cam = create_class_activation_map(wrapper, sample_image, 'conv2', class_idx=0)
    assert cam.shape == (16, 16)  # conv2 va después de un pooling 2x2
    assert 0.0 <= cam.min() <= cam.max() <= 1.0


def test_gradcam_requires_spatial_layer(wrapper, sample_image):
    with pytest.raises(ValueError, match='capa convolucional'):
        create_class_activation_map(wrapper, sample_image, 'predictions', class_idx=0)


def test_overlay_heatmap_resizes_and_returns_uint8(sample_image):
    heatmap = np.linspace(0, 1, 16 * 16).reshape(16, 16).astype(np.float32)
    overlay = overlay_heatmap(sample_image, heatmap)
    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8


def test_overlay_heatmap_accepts_grayscale():
    gray = np.zeros((8, 8), dtype=np.uint8)
    overlay = overlay_heatmap(gray, np.ones((8, 8), dtype=np.float32))
    assert overlay.shape == (8, 8, 3)


def test_visualize_max_activations_ranks_images(wrapper):
    rng = np.random.default_rng(3)
    dataset = [rng.random((32, 32, 3)).astype(np.float32) for _ in range(4)]

    result = visualize_max_activations(wrapper, dataset, 'conv2', n_top=2, n_filters=3)

    assert set(result) == {0, 1, 2}
    for record in result.values():
        assert len(record['values']) == 2
        # Los valores se mantienen ordenados de mayor a menor.
        assert record['values'][0] >= record['values'][1]
        assert record['images'][0] is not None
