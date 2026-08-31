"""
Wrapper para modelos de Keras/TensorFlow que facilita el acceso a activaciones
y gradientes en cualquier capa de la red.
"""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
import tensorflow as tf

from .utils.layers import describe_layer, is_spatial_shape, layer_num_units, layer_output_shape


class ModelWrapper:
    """
    Wrapper para modelos de Keras que proporciona métodos para:

    - Obtener activaciones de cualquier capa
    - Calcular gradientes con respecto a la imagen de entrada
    - Realizar retropropagación guiada (Springenberg et al., 2015)
    - Optimizar imágenes para maximizar activaciones

    Los extractores de activaciones se construyen bajo demanda y se cachean, de
    modo que abrir un modelo grande como VGG16 es instantáneo en lugar de
    construir un sub-modelo por cada capa.
    """

    def __init__(self, model: keras.Model):
        """
        Inicializa el wrapper con un modelo de Keras.

        Args:
            model: Modelo de Keras pre-entrenado

        Raises:
            TypeError: si `model` no es un modelo de Keras.
            ValueError: si el modelo no tiene entradas definidas (por ejemplo un
                modelo subclasificado que nunca ha sido llamado).
        """
        if not isinstance(model, keras.Model):
            raise TypeError(f"Se esperaba un keras.Model, se recibió {type(model).__name__}")

        self.model = model
        self.layer_names: list[str] = [layer.name for layer in model.layers]
        self._extractors: dict[str, keras.Model] = {}

        try:
            self._inputs = model.inputs if model.inputs else [model.input]
        except (AttributeError, ValueError) as error:
            raise ValueError(
                "El modelo no expone tensores de entrada. Construye el modelo "
                "(por ejemplo llamándolo con un lote de ejemplo) antes de envolverlo."
            ) from error

    # ------------------------------------------------------------------
    # Introspección
    # ------------------------------------------------------------------
    @property
    def model_inputs(self):
        """
        Tensores de entrada tal y como espera recibirlos `keras.Model`.

        Para un modelo de una sola entrada se devuelve el tensor suelto: pasar
        una lista de uno hace que Keras avise de que la estructura del lote no
        coincide con la declarada.
        """
        return self._inputs[0] if len(self._inputs) == 1 else self._inputs

    @property
    def input_shape(self) -> tuple[int | None, ...]:
        """Forma de entrada del modelo, incluida la dimensión de lote."""
        return tuple(self._inputs[0].shape)

    def get_layer_info(self, layer_name: str | None = None) -> dict:
        """
        Obtiene información sobre las capas del modelo.

        Args:
            layer_name: Nombre de una capa específica (opcional)

        Returns:
            Información de una capa, o un diccionario `nombre -> información`
        """
        if layer_name is not None:
            return describe_layer(self.model.get_layer(layer_name))
        return {layer.name: describe_layer(layer) for layer in self.model.layers}

    def get_layer_shape(self, layer_name: str) -> tuple[int | None, ...] | None:
        """Forma de salida de una capa, o `None` si no puede determinarse."""
        return layer_output_shape(self.model.get_layer(layer_name))

    def num_filters(self, layer_name: str) -> int:
        """Número de filtros o neuronas visualizables en una capa."""
        return layer_num_units(self.model.get_layer(layer_name))

    def is_spatial_layer(self, layer_name: str) -> bool:
        """Indica si la capa produce un mapa de características espacial."""
        return is_spatial_shape(self.get_layer_shape(layer_name))

    def visualizable_layers(self) -> list[str]:
        """Capas cuyas activaciones tienen sentido visualizar (convolucionales o densas)."""
        names = []
        for layer in self.model.layers:
            shape = layer_output_shape(layer)
            if shape and len(shape) in (2, 4):
                names.append(layer.name)
        return names

    # ------------------------------------------------------------------
    # Extractores de activaciones
    # ------------------------------------------------------------------
    def get_activation_model(self, layer_name: str) -> keras.Model:
        """
        Devuelve (y cachea) un modelo que produce la salida de `layer_name`.

        Args:
            layer_name: Nombre de la capa

        Returns:
            Modelo de Keras cuya salida es la activación de la capa

        Raises:
            ValueError: si la capa no existe o no es alcanzable desde la entrada.
        """
        cached = self._extractors.get(layer_name)
        if cached is not None:
            return cached

        if layer_name not in self.layer_names:
            raise ValueError(
                f"Capa no encontrada: {layer_name}. Capas disponibles: {self.layer_names}"
            )

        layer = self.model.get_layer(layer_name)
        try:
            extractor = keras.Model(inputs=self.model_inputs, outputs=layer.output)
        except (AttributeError, ValueError, TypeError) as error:
            raise ValueError(
                f"No se pudo crear un extractor para la capa '{layer_name}': {error}"
            ) from error

        self._extractors[layer_name] = extractor
        return extractor

    def clear_cache(self) -> None:
        """Libera los extractores de activaciones cacheados."""
        self._extractors.clear()

    def _as_batch(self, image: np.ndarray) -> np.ndarray:
        """Añade la dimensión de lote si hace falta y fuerza `float32`."""
        array = np.asarray(image)
        if array.ndim == 3:
            array = np.expand_dims(array, axis=0)
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        return array

    # ------------------------------------------------------------------
    # Pases hacia adelante y gradientes
    # ------------------------------------------------------------------
    def forward_pass(self, image: np.ndarray,
                     layer_name: str | None = None) -> dict[str, np.ndarray]:
        """
        Realiza un pase hacia adelante y devuelve las activaciones.

        Args:
            image: Imagen de entrada, ya preprocesada para el modelo
            layer_name: Nombre de la capa específica (opcional; `None` devuelve
                todas las capas visualizables)

        Returns:
            Diccionario `nombre de capa -> activaciones`
        """
        batch = self._as_batch(image)

        if layer_name is not None:
            extractor = self.get_activation_model(layer_name)
            return {layer_name: np.asarray(extractor(batch, training=False))}

        names = self.visualizable_layers()
        outputs = keras.Model(
            inputs=self.model_inputs,
            outputs=[self.model.get_layer(name).output for name in names],
        )(batch, training=False)
        return {name: np.asarray(value) for name, value in zip(names, outputs, strict=True)}

    def compute_gradients(self, image: np.ndarray, layer_name: str,
                          filter_indices: int | list[int] | None = None) -> np.ndarray:
        """
        Calcula gradientes de la activación de una capa respecto a la imagen de entrada.

        Args:
            image: Imagen de entrada
            layer_name: Nombre de la capa objetivo
            filter_indices: Índice(s) de filtro concreto(s); `None` usa la capa entera

        Returns:
            Gradientes con la misma forma que la imagen de entrada (con lote)
        """
        extractor = self.get_activation_model(layer_name)
        img_tensor = tf.convert_to_tensor(self._as_batch(image))

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            layer_output = extractor(img_tensor, training=False)
            target = self._reduce_to_target(layer_output, filter_indices)

        grads = tape.gradient(target, img_tensor)
        if grads is None:
            raise ValueError(
                f"La capa '{layer_name}' no es diferenciable respecto a la entrada."
            )
        return grads.numpy()

    @staticmethod
    def _reduce_to_target(layer_output: tf.Tensor,
                          filter_indices: int | list[int] | None) -> tf.Tensor:
        """
        Reduce la salida de una capa a un escalar para derivar.

        Selecciona los filtros pedidos mediante `tf.gather`, lo que evita
        materializar una máscara del tamaño completo de la activación.
        """
        if filter_indices is None:
            return tf.reduce_mean(layer_output)

        indices = [filter_indices] if isinstance(filter_indices, int) else list(filter_indices)
        selected = tf.gather(layer_output, indices, axis=-1)
        return tf.reduce_mean(selected)

    def guided_backprop(self, image: np.ndarray, layer_name: str,
                        filter_indices: int | list[int] | None = None) -> np.ndarray:
        """
        Retropropagación guiada (Springenberg et al., 2015).

        Propaga hacia atrás solo los gradientes positivos que además provienen de
        activaciones positivas, lo que produce visualizaciones mucho más nítidas
        que el gradiente en bruto.

        Args:
            image: Imagen de entrada
            layer_name: Nombre de la capa objetivo
            filter_indices: Índice(s) de filtro concreto(s)

        Returns:
            Gradiente guiado, con la dimensión de lote eliminada
        """
        extractor = self.get_activation_model(layer_name)
        img_tensor = tf.convert_to_tensor(self._as_batch(image))

        @tf.custom_gradient
        def guided_relu(x):
            def grad(dy):
                return tf.cast(dy > 0, dy.dtype) * tf.cast(x > 0, dy.dtype) * dy

            return tf.nn.relu(x), grad

        replaced = self._swap_relu_activations(extractor, guided_relu)
        try:
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                layer_output = extractor(img_tensor, training=False)
                target = self._reduce_to_target(layer_output, filter_indices)
            grads = tape.gradient(target, img_tensor)
        finally:
            for layer, activation in replaced:
                layer.activation = activation

        if grads is None:
            raise ValueError(
                f"La capa '{layer_name}' no es diferenciable respecto a la entrada."
            )
        return grads.numpy()[0]

    @staticmethod
    def _swap_relu_activations(model: keras.Model, replacement) -> list[tuple[Any, Any]]:
        """Sustituye temporalmente las activaciones ReLU y devuelve las originales."""
        replaced = []
        for layer in model.layers:
            activation = getattr(layer, 'activation', None)
            if activation is keras.activations.relu:
                replaced.append((layer, activation))
                layer.activation = replacement
        return replaced

    def deconv(self, image: np.ndarray, layer_name: str,
               filter_indices: int | list[int] | None = None) -> np.ndarray:
        """
        Visualización tipo deconvolución mediante retropropagación guiada.

        Args:
            image: Imagen de entrada
            layer_name: Nombre de la capa objetivo
            filter_indices: Índice(s) de filtro concreto(s)

        Returns:
            Visualización normalizada a [0, 1]
        """
        grads = self.guided_backprop(image, layer_name, filter_indices)
        return self._normalize_for_display(grads)

    def saliency_map(self, image: np.ndarray, layer_name: str,
                     filter_indices: int | list[int] | None = None) -> np.ndarray:
        """
        Mapa de saliencia (Simonyan et al., 2014): máximo del gradiente absoluto
        sobre los canales de color.

        Args:
            image: Imagen de entrada
            layer_name: Nombre de la capa objetivo
            filter_indices: Índice(s) de filtro concreto(s)

        Returns:
            Mapa 2D normalizado a [0, 1]
        """
        grads = self.compute_gradients(image, layer_name, filter_indices)[0]
        saliency = np.max(np.abs(grads), axis=-1)
        return self._normalize_for_display(saliency)

    @staticmethod
    def _normalize_for_display(img: np.ndarray) -> np.ndarray:
        """Normaliza una imagen o gradiente al rango [0, 1] para visualización."""
        img = np.asarray(img, dtype=np.float32)
        if img.ndim == 4:
            img = img[0]

        img = img - np.min(img)
        max_val = float(np.max(img))
        if max_val > 0:
            img = img / max_val
        return img
