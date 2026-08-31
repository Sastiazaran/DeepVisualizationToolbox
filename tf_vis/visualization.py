"""
Métodos para visualizar características y activaciones de redes neuronales.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import keras
import numpy as np
import tensorflow as tf

from .utils.layers import is_spatial_shape

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from .model_wrapper import ModelWrapper


def normalize_01(array: np.ndarray) -> np.ndarray:
    """Escala un array al rango [0, 1] de forma segura frente a arrays constantes."""
    array = np.asarray(array, dtype=np.float32)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    return (array - minimum) / (maximum - minimum + 1e-8)


def standardize_for_display(array: np.ndarray, spread: float = 0.25) -> np.ndarray:
    """
    Centra un gradiente en gris medio y ajusta su contraste por desviación típica.

    Los gradientes tienen unos pocos valores extremos que dominan el rango, así
    que escalarlos por mínimo y máximo deja una imagen casi plana. Normalizar por
    la desviación típica mantiene visible la estructura.

    Args:
        array: Gradiente a visualizar
        spread: Fracción del rango que ocupa una desviación típica

    Returns:
        Array en [0, 1]
    """
    array = np.asarray(array, dtype=np.float32)
    centered = array - float(np.mean(array))
    std = float(np.std(centered))
    if std > 0:
        centered = centered / std
    return np.clip(centered * spread + 0.5, 0.0, 1.0)


def clip_outliers(array: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """
    Recorta la cola superior de un mapa y lo normaliza a [0, 1].

    Un único píxel con un gradiente muy alto puede dejar el resto del mapa de
    saliencia en negro; recortar por percentil evita ese efecto.

    Args:
        array: Mapa a normalizar
        percentile: Percentil por encima del cual se satura

    Returns:
        Array en [0, 1]
    """
    array = np.asarray(array, dtype=np.float32)
    limit = float(np.percentile(array, percentile))
    if limit <= 0:
        return normalize_01(array)
    return np.clip(array / limit, 0.0, 1.0)


def display_activation_grid(activations: np.ndarray, grid_size: tuple[int, int] | None = None,
                            padding: int = 1) -> np.ndarray:
    """
    Organiza las activaciones en una cuadrícula para visualización.

    Args:
        activations: Array de activaciones con forma [n_filtros, alto, ancho]
        grid_size: Tamaño de la cuadrícula (filas, columnas)
        padding: Píxeles de padding entre imágenes

    Returns:
        Imagen con la cuadrícula de activaciones, normalizada a [0, 1]
    """
    activations = np.asarray(activations)
    if activations.ndim != 3:
        raise ValueError(
            f"Se esperaban activaciones [n_filtros, alto, ancho], se recibió {activations.shape}"
        )

    n_filters, height, width = activations.shape

    if grid_size is None or grid_size[0] * grid_size[1] < n_filters:
        side = int(np.ceil(np.sqrt(n_filters)))
        grid_size = (side, side)

    grid_height = grid_size[0] * height + (grid_size[0] - 1) * padding
    grid_width = grid_size[1] * width + (grid_size[1] - 1) * padding
    grid = np.zeros((grid_height, grid_width), dtype=np.float32)

    for filter_idx in range(n_filters):
        row, col = divmod(filter_idx, grid_size[1])
        if row >= grid_size[0]:
            break
        row_start = row * (height + padding)
        col_start = col * (width + padding)
        grid[row_start:row_start + height, col_start:col_start + width] = normalize_01(
            activations[filter_idx]
        )

    return grid


def visualize_layer_filters(model: keras.Model, layer_name: str,
                            grid_size: tuple[int, int] | None = None) -> np.ndarray:
    """
    Visualiza los pesos (filtros) de una capa convolucional.

    Args:
        model: Modelo de Keras
        layer_name: Nombre de la capa
        grid_size: Tamaño de la cuadrícula (filas, columnas)

    Returns:
        Imagen con la cuadrícula de filtros
    """
    layer = model.get_layer(layer_name)

    if not isinstance(layer, keras.layers.Conv2D):
        raise ValueError(f"La capa {layer_name} no es una capa convolucional")

    weights = layer.get_weights()
    if not weights:
        raise ValueError(f"La capa {layer_name} no tiene pesos cargados")

    # Los kernels de Keras tienen forma [alto, ancho, canales_entrada, n_filtros].
    kernels = normalize_01(weights[0])
    kernels = np.mean(kernels, axis=2)          # promedio sobre canales de entrada
    kernels = np.transpose(kernels, (2, 0, 1))  # -> [n_filtros, alto, ancho]

    return display_activation_grid(kernels, grid_size)


def resolve_ascent_size(model_wrapper: ModelWrapper,
                        image_size: tuple[int, int] | None) -> tuple[int, int]:
    """
    Determina el tamaño `(alto, ancho)` de la imagen a optimizar.

    Los modelos con `include_top=True` fijan su resolución de entrada, así que
    pedir otro tamaño produciría un error de forma poco descriptivo dentro de
    Keras. Los modelos sin cabeza aceptan cualquier resolución.

    Args:
        model_wrapper: Wrapper del modelo
        image_size: Tamaño solicitado, o `None` para usar el nativo del modelo

    Returns:
        Tamaño `(alto, ancho)` utilizable
    """
    shape = model_wrapper.input_shape
    native = (shape[1], shape[2]) if len(shape) == 4 else (None, None)

    if image_size is None:
        return (native[0] or 224, native[1] or 224)

    fixed = tuple(dim for dim in native if dim is not None)
    if fixed and tuple(native) != tuple(image_size):
        raise ValueError(
            f"El modelo espera entradas de {native[0]}x{native[1]}; no se puede optimizar "
            f"una imagen de {image_size[0]}x{image_size[1]}. Carga el modelo con "
            "include_top=False para admitir cualquier resolución."
        )
    return tuple(image_size)


def apply_gradient_ascent(model_wrapper: ModelWrapper, layer_name: str, filter_index: int,
                          iterations: int = 30, step_size: float = 1.0,
                          image_size: tuple[int, int] | None = None,
                          seed: int | None = None) -> np.ndarray:
    """
    Genera por ascenso de gradiente una imagen que maximiza la activación de un filtro.

    Args:
        model_wrapper: Wrapper del modelo
        layer_name: Nombre de la capa objetivo
        filter_index: Índice del filtro a maximizar
        iterations: Número de iteraciones de optimización
        step_size: Tamaño del paso
        image_size: Tamaño `(alto, ancho)` de la imagen; `None` usa el nativo del modelo
        seed: Semilla para la inicialización aleatoria, útil en tests

    Returns:
        Imagen generada, normalizada a [0, 1]
    """
    extractor = model_wrapper.get_activation_model(layer_name)
    height, width = resolve_ascent_size(model_wrapper, image_size)

    rng = np.random.default_rng(seed)
    img = rng.random((1, height, width, 3), dtype=np.float32) * 0.1 + 0.45
    img_tensor = tf.Variable(img, dtype=tf.float32)

    layer_shape = model_wrapper.get_layer_shape(layer_name)
    is_spatial = is_spatial_shape(layer_shape)
    # Se descartan los bordes, dominados por el padding, salvo en mapas de
    # características tan pequeños que recortarlos los dejaría vacíos.
    margin = 2 if is_spatial and min(layer_shape[1] or 0, layer_shape[2] or 0) > 6 else 0

    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            layer_output = extractor(img_tensor, training=False)
            if is_spatial:
                cropped = (layer_output[:, margin:-margin, margin:-margin, filter_index]
                           if margin else layer_output[:, :, :, filter_index])
                loss = tf.reduce_mean(cropped)
            else:
                loss = tf.reduce_mean(layer_output[:, filter_index])

        grads = tape.gradient(loss, img_tensor)
        if grads is None:
            raise ValueError(f"No se pudieron calcular gradientes para la capa '{layer_name}'")

        grads = tf.math.l2_normalize(grads)
        img_tensor.assign_add(step_size * grads)
        img_tensor.assign(tf.clip_by_value(img_tensor, 0.0, 1.0))

    return normalize_01(img_tensor.numpy()[0])


def visualize_max_activations(model_wrapper: ModelWrapper, dataset, layer_name: str,
                              n_top: int = 9, n_filters: int | None = None) -> dict:
    """
    Encuentra las imágenes que provocan las activaciones máximas de cada filtro.

    Args:
        model_wrapper: Wrapper del modelo
        dataset: Iterable de imágenes (o de tuplas `(imagen, etiqueta)`)
        layer_name: Nombre de la capa
        n_top: Número de imágenes top a guardar por filtro
        n_filters: Número de filtros a procesar (`None` = todos)

    Returns:
        Diccionario `índice de filtro -> {'values', 'images'}`
    """
    total_filters = model_wrapper.num_filters(layer_name)
    if total_filters == 0:
        raise ValueError(f"No se pudo determinar el número de filtros de '{layer_name}'")

    n_filters = total_filters if n_filters is None else min(n_filters, total_filters)
    is_spatial = model_wrapper.is_spatial_layer(layer_name)

    max_activations = {
        i: {'values': np.full(n_top, -np.inf), 'images': [None] * n_top}
        for i in range(n_filters)
    }

    for img_data in dataset:
        img = img_data[0] if isinstance(img_data, tuple) else img_data
        activations = model_wrapper.forward_pass(img, layer_name)[layer_name]

        scores = np.max(activations[0], axis=(0, 1)) if is_spatial else activations[0]

        for filter_idx in range(n_filters):
            record = max_activations[filter_idx]
            min_idx = int(np.argmin(record['values']))
            if scores[filter_idx] > record['values'][min_idx]:
                record['values'][min_idx] = scores[filter_idx]
                record['images'][min_idx] = np.array(img, copy=True)

                order = np.argsort(-record['values'])
                record['values'] = record['values'][order]
                record['images'] = [record['images'][i] for i in order]

    return max_activations


def create_class_activation_map(model_wrapper: ModelWrapper, img: np.ndarray,
                                layer_name: str, class_idx: int) -> np.ndarray:
    """
    Crea un mapa de activación de clase con Grad-CAM (Selvaraju et al., 2017).

    A diferencia del CAM original, Grad-CAM no exige una arquitectura con
    average pooling global seguido de una única capa densa, por lo que funciona
    con cualquier modelo del registro.

    Args:
        model_wrapper: Wrapper del modelo
        img: Imagen de entrada, ya preprocesada
        layer_name: Nombre de una capa convolucional
        class_idx: Índice de la clase a explicar

    Returns:
        Mapa de calor 2D normalizado a [0, 1], con el tamaño del mapa de características
    """
    if not model_wrapper.is_spatial_layer(layer_name):
        raise ValueError(f"Grad-CAM requiere una capa convolucional, '{layer_name}' no lo es")

    batch = np.asarray(img, dtype=np.float32)
    if batch.ndim == 3:
        batch = np.expand_dims(batch, axis=0)

    grad_model = keras.Model(
        inputs=model_wrapper.model_inputs,
        outputs=[
            model_wrapper.model.get_layer(layer_name).output,
            model_wrapper.model.outputs[0],
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch, training=False)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError(f"No se pudieron calcular gradientes para la capa '{layer_name}'")

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_outputs[0] * weights[0], axis=-1)
    cam = tf.nn.relu(cam).numpy()

    return normalize_01(cam)


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Superpone un mapa de calor sobre una imagen.

    Args:
        img: Imagen original
        heatmap: Mapa de calor con valores en [0, 1]
        alpha: Factor de mezcla
        colormap: Mapa de colores de OpenCV

    Returns:
        Imagen `uint8` con el mapa de calor superpuesto
    """
    img = np.asarray(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)

    if img.dtype != np.uint8:
        img = np.clip(normalize_01(img) * 255, 0, 255).astype(np.uint8)

    # El mapa de calor suele venir del mapa de características, más pequeño que la imagen.
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0, 1)), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(img, 1.0 - alpha, heatmap_colored, alpha, 0)
