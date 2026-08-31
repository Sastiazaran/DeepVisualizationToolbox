"""Fixtures compartidas para la suite de tests."""

from __future__ import annotations

import os

import numpy as np
import pytest

# TensorFlow no debe intentar usar la GPU ni imprimir sus logs informativos.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# Qt debe poder crear widgets sin un servidor gráfico.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

IMAGE_SIZE = 32


@pytest.fixture(scope='session')
def toy_model():
    """
    Modelo funcional pequeño con la misma estructura que una CNN de clasificación.

    Se usa en lugar de VGG16 para que los tests no descarguen pesos ni tarden
    minutos en ejecutarse.
    """
    import keras

    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name='input_image')
    x = keras.layers.Conv2D(8, 3, activation='relu', padding='same', name='conv1')(inputs)
    x = keras.layers.MaxPooling2D(2, name='pool1')(x)
    x = keras.layers.Conv2D(16, 3, activation='relu', padding='same', name='conv2')(x)
    x = keras.layers.GlobalAveragePooling2D(name='gap')(x)
    outputs = keras.layers.Dense(10, activation='softmax', name='predictions')(x)
    return keras.Model(inputs, outputs, name='toy_cnn')


@pytest.fixture(scope='session')
def wrapper(toy_model):
    """`ModelWrapper` construido sobre el modelo de juguete."""
    from tf_vis.model_wrapper import ModelWrapper

    return ModelWrapper(toy_model)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Imagen RGB determinista en el rango [0, 1]."""
    rng = np.random.default_rng(1234)
    return rng.random((IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32)


@pytest.fixture
def image_dir(tmp_path):
    """Directorio temporal con tres imágenes JPEG distinguibles entre sí."""
    import cv2

    for index, value in enumerate((10, 120, 240)):
        img = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), value, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f'img_{index}.jpg'), img)
    return tmp_path
