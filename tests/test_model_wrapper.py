"""Tests de `ModelWrapper`."""

from __future__ import annotations

import numpy as np
import pytest


def test_rejects_non_keras_model():
    from tf_vis.model_wrapper import ModelWrapper

    with pytest.raises(TypeError):
        ModelWrapper(object())


def test_layer_names_and_visualizable_layers(wrapper):
    assert wrapper.layer_names[0] == 'input_image'
    # `gap` produce un vector [batch, canales], así que también es visualizable.
    assert wrapper.visualizable_layers() == [
        'input_image', 'conv1', 'pool1', 'conv2', 'gap', 'predictions'
    ]


def test_layer_info_reports_shape_on_keras3(wrapper):
    info = wrapper.get_layer_info('conv1')
    assert info['type'] == 'Conv2D'
    assert info['shape'] == (None, 32, 32, 8)
    assert info['units'] == 8
    assert info['is_spatial'] is True

    dense_info = wrapper.get_layer_info('predictions')
    assert dense_info['is_spatial'] is False
    assert dense_info['units'] == 10


def test_input_layer_shape_is_available(wrapper):
    # Keras 3 eliminó `batch_input_shape`; la forma debe salir del tensor de salida.
    assert wrapper.get_layer_shape('input_image') == (None, 32, 32, 3)


def test_extractors_are_built_lazily_and_cached(wrapper):
    wrapper.clear_cache()
    assert wrapper._extractors == {}

    first = wrapper.get_activation_model('conv1')
    assert set(wrapper._extractors) == {'conv1'}
    assert wrapper.get_activation_model('conv1') is first


def test_unknown_layer_raises(wrapper):
    with pytest.raises(ValueError, match='Capa no encontrada'):
        wrapper.get_activation_model('no_existe')


def test_forward_pass_single_layer(wrapper, sample_image):
    activations = wrapper.forward_pass(sample_image, 'conv1')
    assert set(activations) == {'conv1'}
    assert activations['conv1'].shape == (1, 32, 32, 8)


def test_forward_pass_accepts_batched_input(wrapper, sample_image):
    batched = np.expand_dims(sample_image, 0)
    assert wrapper.forward_pass(batched, 'conv1')['conv1'].shape == (1, 32, 32, 8)


def test_forward_pass_all_layers(wrapper, sample_image):
    activations = wrapper.forward_pass(sample_image)
    assert set(activations) == set(wrapper.visualizable_layers())


def test_compute_gradients_matches_input_shape(wrapper, sample_image):
    grads = wrapper.compute_gradients(sample_image, 'conv2', filter_indices=3)
    assert grads.shape == (1, 32, 32, 3)
    assert np.isfinite(grads).all()


def test_compute_gradients_accepts_filter_lists(wrapper, sample_image):
    grads = wrapper.compute_gradients(sample_image, 'conv2', filter_indices=[0, 1, 2])
    assert grads.shape == (1, 32, 32, 3)


def test_gradients_differ_between_filters(wrapper, sample_image):
    first = wrapper.compute_gradients(sample_image, 'conv2', filter_indices=0)
    second = wrapper.compute_gradients(sample_image, 'conv2', filter_indices=5)
    assert not np.allclose(first, second)


def test_guided_backprop_restores_activations(wrapper, sample_image):
    import keras

    conv = wrapper.model.get_layer('conv1')
    original = conv.activation

    result = wrapper.guided_backprop(sample_image, 'conv2', filter_indices=1)

    assert result.shape == (32, 32, 3)
    assert conv.activation is original is keras.activations.relu


def test_deconv_is_normalized(wrapper, sample_image):
    deconv = wrapper.deconv(sample_image, 'conv2', filter_indices=1)
    assert deconv.shape == (32, 32, 3)
    assert deconv.min() >= 0.0
    assert deconv.max() <= 1.0


def test_saliency_map_is_2d(wrapper, sample_image):
    saliency = wrapper.saliency_map(sample_image, 'conv2', filter_indices=1)
    assert saliency.shape == (32, 32)
    assert 0.0 <= saliency.min() <= saliency.max() <= 1.0


def test_num_filters(wrapper):
    assert wrapper.num_filters('conv1') == 8
    assert wrapper.num_filters('conv2') == 16
    assert wrapper.num_filters('predictions') == 10
