"""
Introspección de capas compatible con Keras 3.

Keras 3 eliminó `layer.output_shape` y `layer.batch_input_shape`. Estas ayudas
centralizan la forma correcta de obtener la información de una capa tanto en
Keras 3 como en las versiones 2.x más antiguas.
"""

from __future__ import annotations

from typing import Any

import keras

# Tipos de capa cuya salida se interpreta como un mapa de características
# espacial [batch, alto, ancho, canales].
_SPATIAL_RANK = 4


def layer_output_shape(layer: keras.layers.Layer) -> tuple[int | None, ...] | None:
    """
    Devuelve la forma de salida de una capa, o `None` si no se puede determinar.

    Args:
        layer: Capa de Keras

    Returns:
        Tupla de dimensiones (con `None` en las dimensiones dinámicas)
    """
    output = getattr(layer, 'output', None)
    shape = getattr(output, 'shape', None)
    if shape is not None:
        return tuple(shape)

    # Keras 2 y capas de entrada que aún no han sido llamadas.
    for attribute in ('output_shape', 'batch_shape', 'batch_input_shape'):
        shape = getattr(layer, attribute, None)
        if shape is not None:
            return tuple(shape)

    return None


def layer_num_units(layer: keras.layers.Layer) -> int:
    """
    Número de filtros (capa convolucional) o neuronas (capa densa) de una capa.

    Returns:
        Número de unidades visualizables, o 0 si no se puede determinar.
    """
    shape = layer_output_shape(layer)
    if not shape:
        return 0
    last = shape[-1]
    return int(last) if last is not None else 0


def is_spatial_shape(shape: tuple[int | None, ...] | None) -> bool:
    """Indica si una forma corresponde a un mapa de características espacial."""
    return bool(shape) and len(shape) == _SPATIAL_RANK


def describe_layer(layer: keras.layers.Layer) -> dict[str, Any]:
    """
    Resume una capa en un diccionario serializable.

    Args:
        layer: Capa de Keras

    Returns:
        Diccionario con nombre, tipo, forma, número de unidades y parámetros
    """
    shape = layer_output_shape(layer)
    return {
        'name': layer.name,
        'type': type(layer).__name__,
        'shape': shape,
        'units': layer_num_units(layer),
        'is_spatial': is_spatial_shape(shape),
        'params': layer.count_params(),
    }
