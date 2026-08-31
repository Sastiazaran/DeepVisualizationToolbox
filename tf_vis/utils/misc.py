"""
Funciones misceláneas de utilidad: registro de modelos, carga de pesos y resúmenes.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import keras
import numpy as np

from .layers import layer_output_shape

# El registro se construye de forma perezosa para no importar todas las
# aplicaciones de Keras (y descargar metadatos) al importar el paquete.
_MODEL_REGISTRY: dict[str, ModelSpec] | None = None
_IMAGENET_LABELS: list[str] | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Descripción de un modelo pre-entrenado disponible en `keras.applications`."""

    name: str
    constructor: Callable[..., keras.Model]
    preprocess: Callable[[np.ndarray], np.ndarray]
    input_shape: tuple[int, int, int]
    decode_predictions: Callable[..., Any] | None = field(default=None)

    def as_dict(self) -> dict[str, Any]:
        """Vista de diccionario, por compatibilidad con la API previa a 0.2."""
        return {
            'module': self.constructor,
            'constructor': self.constructor,
            'preprocess': self.preprocess,
            'input_shape': self.input_shape,
            'decode_predictions': self.decode_predictions,
        }


def _build_registry() -> dict[str, ModelSpec]:
    apps = keras.applications
    specs = [
        ModelSpec('vgg16', apps.VGG16, apps.vgg16.preprocess_input, (224, 224, 3),
                  apps.vgg16.decode_predictions),
        ModelSpec('vgg19', apps.VGG19, apps.vgg19.preprocess_input, (224, 224, 3),
                  apps.vgg19.decode_predictions),
        ModelSpec('resnet50', apps.ResNet50, apps.resnet50.preprocess_input, (224, 224, 3),
                  apps.resnet50.decode_predictions),
        ModelSpec('resnet50v2', apps.ResNet50V2, apps.resnet_v2.preprocess_input, (224, 224, 3),
                  apps.resnet_v2.decode_predictions),
        ModelSpec('inception_v3', apps.InceptionV3, apps.inception_v3.preprocess_input,
                  (299, 299, 3), apps.inception_v3.decode_predictions),
        ModelSpec('mobilenet', apps.MobileNet, apps.mobilenet.preprocess_input, (224, 224, 3),
                  apps.mobilenet.decode_predictions),
        ModelSpec('mobilenet_v2', apps.MobileNetV2, apps.mobilenet_v2.preprocess_input,
                  (224, 224, 3), apps.mobilenet_v2.decode_predictions),
        ModelSpec('efficientnet_b0', apps.EfficientNetB0, apps.efficientnet.preprocess_input,
                  (224, 224, 3), apps.efficientnet.decode_predictions),
        ModelSpec('efficientnetv2_b0', apps.EfficientNetV2B0,
                  apps.efficientnet_v2.preprocess_input, (224, 224, 3),
                  apps.efficientnet_v2.decode_predictions),
        ModelSpec('convnext_tiny', apps.ConvNeXtTiny, apps.convnext.preprocess_input,
                  (224, 224, 3), apps.convnext.decode_predictions),
    ]
    return {spec.name: spec for spec in specs}


def get_model_specs() -> dict[str, ModelSpec]:
    """
    Devuelve el registro de modelos pre-entrenados soportados.

    Returns:
        Diccionario `nombre -> ModelSpec`
    """
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is None:
        _MODEL_REGISTRY = _build_registry()
    return _MODEL_REGISTRY


def get_available_models() -> dict[str, dict]:
    """
    Obtiene los modelos pre-entrenados disponibles como diccionarios.

    Se mantiene por compatibilidad con versiones anteriores; el código nuevo
    debería usar :func:`get_model_specs`.

    Returns:
        Diccionario con información de modelos disponibles
    """
    return {name: spec.as_dict() for name, spec in get_model_specs().items()}


def get_model_spec(model_name: str) -> ModelSpec:
    """
    Busca la especificación de un modelo por nombre (sin distinguir mayúsculas).

    Raises:
        ValueError: si el modelo no está registrado.
    """
    specs = get_model_specs()
    key = model_name.lower().replace('-', '_')
    if key not in specs:
        raise ValueError(
            f"Modelo '{model_name}' no disponible. Opciones: {sorted(specs)}"
        )
    return specs[key]


def load_model(model_name: str, weights: str | None = 'imagenet',
               include_top: bool = True) -> keras.Model:
    """
    Carga un modelo pre-entrenado del registro.

    Args:
        model_name: Nombre del modelo
        weights: Pesos a utilizar ('imagenet', None, o ruta a un archivo)
        include_top: Si se incluye la capa de clasificación

    Returns:
        Modelo cargado
    """
    spec = get_model_spec(model_name)
    return spec.constructor(weights=weights, include_top=include_top)


def load_model_from_file(path: str) -> keras.Model:
    """
    Carga un modelo guardado en disco (`.keras`, `.h5` o SavedModel).

    Args:
        path: Ruta al modelo

    Returns:
        Modelo cargado
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el modelo en la ruta: {path}")
    return keras.models.load_model(path)


def get_preprocessing_function(model_name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Devuelve la función de preprocesamiento asociada a un modelo del registro."""
    return get_model_spec(model_name).preprocess


def predict_image(model: keras.Model, img: np.ndarray, top_k: int = 5,
                  model_name: str | None = None) -> list[tuple[int, str, float]]:
    """
    Realiza una predicción y devuelve las clases top-k.

    Args:
        model: Modelo Keras
        img: Imagen ya preprocesada
        top_k: Número de clases top a devolver
        model_name: Aceptado por compatibilidad; todas las aplicaciones de Keras
            comparten el mismo orden de clases de ImageNet.

    Returns:
        Lista de tuplas (índice, etiqueta, probabilidad)
    """
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    preds = np.asarray(model.predict(img, verbose=0))
    top_indices = np.argsort(preds[0])[-top_k:][::-1]

    labels = get_imagenet_labels(model_name)
    results = []
    for idx in top_indices:
        idx = int(idx)
        label = labels[idx] if idx < len(labels) else f'class_{idx}'
        results.append((idx, label, float(preds[0][idx])))
    return results


def get_imagenet_labels(model_name: str | None = None) -> list[str]:
    """
    Obtiene las 1000 etiquetas de ImageNet en el orden que usan los modelos de Keras.

    Se leen del índice de clases oficial de Keras. Esto evita el clásico desfase
    de un índice que aparece al usar el archivo `ImageNetLabels.txt` de
    TensorFlow, que incluye una clase extra de fondo en la posición 0 y por lo
    tanto no coincide con las salidas de `keras.applications`.

    Args:
        model_name: Aceptado por compatibilidad; todos los modelos del registro
            comparten el mismo orden de clases.

    Returns:
        Lista de 1000 etiquetas
    """
    del model_name  # Todas las aplicaciones de Keras comparten el índice de clases.

    global _IMAGENET_LABELS
    if _IMAGENET_LABELS is None:
        path = keras.utils.get_file(
            'imagenet_class_index.json',
            'https://storage.googleapis.com/download.tensorflow.org/data/'
            'imagenet_class_index.json',
            cache_subdir='models',
        )
        with open(path) as handle:
            class_index = json.load(handle)
        _IMAGENET_LABELS = [class_index[str(i)][1] for i in range(len(class_index))]
    return list(_IMAGENET_LABELS)


def time_execution(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """
    Mide el tiempo de ejecución de una función.

    Args:
        func: Función a medir
        *args, **kwargs: Argumentos para la función

    Returns:
        Tupla (resultado de la función, tiempo en segundos)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start_time


def save_visualizations(visualizations: dict[str, np.ndarray],
                        output_dir: str, prefix: str = '') -> list[str]:
    """
    Guarda visualizaciones en disco como PNG.

    Args:
        visualizations: Diccionario de visualizaciones
        output_dir: Directorio de salida
        prefix: Prefijo para los nombres de archivo

    Returns:
        Lista de rutas escritas
    """
    # matplotlib se importa aquí para no pagar su coste al importar el paquete.
    import matplotlib

    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    written = []
    for name, vis in visualizations.items():
        filename = f"{prefix}_{name}.png" if prefix else f"{name}.png"
        filepath = os.path.join(output_dir, filename)

        if vis.dtype != np.uint8:
            vis = np.clip(vis * 255, 0, 255).astype(np.uint8)

        plt.imsave(filepath, vis)
        written.append(filepath)

    return written


def count_params(weights) -> int:
    """Cuenta parámetros en una lista de pesos (sustituto de `keras.backend.count_params`)."""
    return int(sum(np.prod(w.shape) for w in weights))


def create_model_summary(model: keras.Model) -> dict:
    """
    Crea un resumen detallado del modelo.

    Args:
        model: Modelo Keras

    Returns:
        Diccionario con información del modelo
    """
    summary = {
        'name': model.name,
        'layers': [],
        'total_params': model.count_params(),
        'trainable_params': count_params(model.trainable_weights),
        'non_trainable_params': count_params(model.non_trainable_weights),
    }

    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': type(layer).__name__,
            'output_shape': str(layer_output_shape(layer)),
            'params': layer.count_params(),
            'trainable': layer.trainable,
        }

        if isinstance(layer, keras.layers.Conv2D):
            layer_info.update({
                'filters': layer.filters,
                'kernel_size': layer.kernel_size,
                'strides': layer.strides,
                'padding': layer.padding,
                'activation': getattr(layer.activation, '__name__', None),
            })
        elif isinstance(layer, keras.layers.Dense):
            layer_info.update({
                'units': layer.units,
                'activation': getattr(layer.activation, '__name__', None),
            })
        elif isinstance(layer, keras.layers.MaxPooling2D):
            layer_info.update({
                'pool_size': layer.pool_size,
                'strides': layer.strides,
                'padding': layer.padding,
            })

        summary['layers'].append(layer_info)

    return summary
