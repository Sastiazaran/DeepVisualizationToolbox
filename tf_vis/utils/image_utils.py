"""
Utilidades para procesamiento de imágenes.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import keras
import numpy as np
import tensorflow as tf

# Media BGR que resta el preprocesamiento estilo Caffe (VGG16, VGG19, ResNet50).
_CAFFE_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)
_CAFFE_MODELS = {'vgg16', 'vgg19', 'resnet50'}
_TF_SCALED_MODELS = {'inception_v3', 'mobilenet', 'mobilenet_v2', 'efficientnetv2_b0'}


def load_image(path: str, target_size: tuple[int, int] | None = None,
               preprocess_fn: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    """
    Carga una imagen desde un archivo en formato RGB.

    Args:
        path: Ruta al archivo de imagen
        target_size: Tamaño objetivo (ancho, alto)
        preprocess_fn: Función de preprocesamiento opcional

    Returns:
        Imagen como array numpy RGB
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    if preprocess_fn:
        img = preprocess_fn(img)

    return img


def preprocess_image_for_model(img: np.ndarray, model_name: str) -> np.ndarray:
    """
    Preprocesa una imagen para un modelo concreto del registro.

    Args:
        img: Imagen RGB como array numpy
        model_name: Nombre del modelo ('vgg16', 'resnet50', etc.)

    Returns:
        Imagen preprocesada
    """
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # Importación diferida para evitar un ciclo entre utils.misc y utils.image_utils.
    from .misc import get_model_spec

    try:
        spec = get_model_spec(model_name)
    except ValueError:
        return img.astype(np.float32) / 255.0

    return spec.preprocess(np.array(img, dtype=np.float32, copy=True))


def deprocess_image(img: np.ndarray, model_name: str | None = None) -> np.ndarray:
    """
    Convierte una imagen preprocesada de vuelta a un formato visualizable.

    Args:
        img: Imagen preprocesada
        model_name: Nombre del modelo (opcional)

    Returns:
        Imagen con valores en [0, 1]
    """
    img = np.asarray(img, dtype=np.float32)

    if model_name is None:
        return np.clip(img, 0, 1)

    key = model_name.lower().replace('-', '_')

    if key in _CAFFE_MODELS:
        restored = img.copy() + _CAFFE_MEAN  # el preprocesado deja los canales en BGR
        restored = restored[..., ::-1]
        return np.clip(restored / 255.0, 0, 1)

    if key in _TF_SCALED_MODELS:
        return np.clip((img + 1.0) / 2.0, 0, 1)

    return np.clip(img, 0, 1)


def apply_gradcam(model: keras.Model, img: np.ndarray,
                  layer_name: str, class_idx: int | None = None) -> np.ndarray:
    """
    Aplica Grad-CAM para localizar las regiones que sustentan una predicción.

    Args:
        model: Modelo de Keras
        img: Imagen de entrada, ya preprocesada
        layer_name: Nombre de la capa convolucional a explicar
        class_idx: Índice de la clase; `None` usa la clase predicha

    Returns:
        Mapa de calor 2D normalizado a [0, 1]
    """
    batch = np.asarray(img, dtype=np.float32)
    if batch.ndim == 3:
        batch = np.expand_dims(batch, axis=0)

    # Un modelo de una sola entrada debe declararse con el tensor suelto; con una
    # lista de uno Keras avisa de que la estructura del lote no coincide.
    inputs = model.inputs[0] if len(model.inputs) == 1 else model.inputs
    grad_model = keras.Model(
        inputs=inputs,
        outputs=[model.get_layer(layer_name).output, model.outputs[0]],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch, training=False)
        target_idx = int(tf.argmax(predictions[0])) if class_idx is None else class_idx
        loss = predictions[:, target_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError(f"No se pudieron calcular gradientes para la capa '{layer_name}'")

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads[0], axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy()

    # La división directa por el máximo produce NaN cuando el mapa es todo cero.
    maximum = float(np.max(heatmap))
    return heatmap / maximum if maximum > 0 else heatmap


def overlay_gradcam(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Superpone un mapa de calor Grad-CAM sobre una imagen.

    Args:
        img: Imagen original
        heatmap: Mapa de calor Grad-CAM en [0, 1]
        alpha: Factor de mezcla

    Returns:
        Imagen `uint8` con el mapa de calor superpuesto
    """
    from ..visualization import overlay_heatmap

    return overlay_heatmap(img, heatmap, alpha=alpha)


def create_grid_of_images(images: list[np.ndarray], grid_size: tuple[int, int] | None = None,
                          padding: int = 1) -> np.ndarray:
    """
    Crea una cuadrícula con una lista de imágenes del mismo tamaño.

    Args:
        images: Lista de imágenes
        grid_size: Tamaño de la cuadrícula (filas, columnas)
        padding: Píxeles de padding entre imágenes

    Returns:
        Imagen con la cuadrícula
    """
    if not images:
        raise ValueError("Se necesita al menos una imagen para crear la cuadrícula")

    n_images = len(images)

    if grid_size is None or grid_size[0] * grid_size[1] < n_images:
        side = int(np.ceil(np.sqrt(n_images)))
        grid_size = (side, side)

    h, w = images[0].shape[:2]
    grid_h = grid_size[0] * h + (grid_size[0] - 1) * padding
    grid_w = grid_size[1] * w + (grid_size[1] - 1) * padding

    if images[0].ndim == 3:
        grid = np.zeros((grid_h, grid_w, images[0].shape[2]), dtype=images[0].dtype)
    else:
        grid = np.zeros((grid_h, grid_w), dtype=images[0].dtype)

    for i in range(min(n_images, grid_size[0] * grid_size[1])):
        row, col = divmod(i, grid_size[1])
        y = row * (h + padding)
        x = col * (w + padding)
        grid[y:y + h, x:x + w] = images[i]

    return grid
