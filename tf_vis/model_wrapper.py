"""
Wrapper para modelos de TensorFlow que facilita el acceso a activaciones
y gradientes en cualquier capa de la red.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any


class ModelWrapper:
    """
    Wrapper para modelos de TensorFlow que proporciona métodos para:
    - Obtener activaciones de cualquier capa
    - Calcular gradientes con respecto a cualquier capa
    - Realizar deconvolución (similar a Zeiler & Fergus)
    - Optimizar imágenes para maximizar activaciones
    """
    
    def __init__(self, model: tf.keras.Model):
        """
        Inicializa el wrapper con un modelo TensorFlow.
        
        Args:
            model: Modelo de TensorFlow/Keras pre-entrenado
        """
        self.model = model
        self.layer_names = [layer.name for layer in model.layers]
        self._layer_outputs = {}
        self._setup_feature_extractors()
        
    def _setup_feature_extractors(self):
        """Configura extractores de características para cada capa del modelo."""
        # Crear modelos que devuelven las activaciones de cada capa
        for layer_name in self.layer_names:
            try:
                layer = self.model.get_layer(layer_name)
                self._layer_outputs[layer_name] = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )
            except:
                print(f"No se pudo crear extractor para capa: {layer_name}")
    
    def forward_pass(self, image: np.ndarray, layer_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Realiza un pase hacia adelante y devuelve las activaciones.
        
        Args:
            image: Imagen de entrada (debe tener la forma esperada por el modelo)
            layer_name: Nombre de la capa específica (opcional)
            
        Returns:
            Diccionario con activaciones por capa o activación de una capa específica
        """
        # Asegurar que la imagen tiene la forma correcta (añadir batch dimension si es necesario)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Preprocesar imagen si es necesario
        # (aquí se podría añadir preprocesamiento específico del modelo)
        
        if layer_name is not None:
            # Devolver activaciones solo para la capa solicitada
            if layer_name in self._layer_outputs:
                return {layer_name: self._layer_outputs[layer_name](image).numpy()}
            else:
                raise ValueError(f"Capa no encontrada: {layer_name}")
        else:
            # Devolver activaciones para todas las capas
            return {name: extractor(image).numpy() 
                   for name, extractor in self._layer_outputs.items()}
    
    def compute_gradients(self, image: np.ndarray, layer_name: str, 
                         filter_indices: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        """
        Calcula gradientes de la activación de una capa con respecto a la imagen de entrada.
        
        Args:
            image: Imagen de entrada
            layer_name: Nombre de la capa objetivo
            filter_indices: Índice(s) de filtro específico(s) para calcular gradientes
            
        Returns:
            Gradientes calculados
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        # Convertir la imagen a un tensor y habilitar la grabación de gradientes
        img_tensor = tf.convert_to_tensor(image)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            
            # Obtener la salida de la capa especificada
            layer_output = self._layer_outputs[layer_name](img_tensor)
            
            # Si se especifican índices de filtro, calcular gradientes solo para esos filtros
            if filter_indices is not None:
                if isinstance(filter_indices, int):
                    filter_indices = [filter_indices]
                    
                # Crear una máscara para los filtros seleccionados
                mask = np.zeros_like(layer_output.numpy())
                for idx in filter_indices:
                    if len(layer_output.shape) == 4:  # Para capas convolucionales
                        mask[:, :, :, idx] = 1
                    else:  # Para capas densas
                        mask[:, idx] = 1
                        
                layer_output = layer_output * tf.convert_to_tensor(mask, dtype=layer_output.dtype)
            
            # Calcular la suma de las activaciones como objetivo para el gradiente
            target = tf.reduce_sum(layer_output)
            
        # Calcular gradientes
        grads = tape.gradient(target, img_tensor)
        return grads.numpy()
    
    def deconv(self, image: np.ndarray, layer_name: str, 
              filter_indices: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        """
        Implementa deconvolución similar a Zeiler & Fergus para visualizar qué
        partes de la imagen activan ciertos filtros.
        
        Args:
            image: Imagen de entrada
            layer_name: Nombre de la capa objetivo
            filter_indices: Índice(s) de filtro específico(s)
            
        Returns:
            Visualización de deconvolución
        """
        # Implementación simplificada de deconvolución
        # En una implementación completa, se necesitaría implementar
        # la propagación hacia atrás con ReLUs guiadas o similar
        
        # Por ahora, usamos gradientes con algunas modificaciones
        grads = self.compute_gradients(image, layer_name, filter_indices)
        
        # Aplicar ReLU a los gradientes (solo valores positivos)
        grads = np.maximum(grads, 0)
        
        # Normalizar para visualización
        grads = self._normalize_for_display(grads)
        
        return grads
    
    def _normalize_for_display(self, img: np.ndarray) -> np.ndarray:
        """Normaliza una imagen o gradiente para visualización."""
        # Eliminar dimensión de batch si existe
        if len(img.shape) == 4:
            img = img[0]
            
        # Normalizar a [0, 1]
        img = img - np.min(img)
        max_val = np.max(img)
        if max_val > 0:
            img = img / max_val
            
        return img
    
    def get_layer_info(self, layer_name: Optional[str] = None) -> Dict:
        """
        Obtiene información sobre las capas del modelo.
        
        Args:
            layer_name: Nombre de una capa específica (opcional)
            
        Returns:
            Diccionario con información de las capas
        """
        if layer_name is not None:
            layer = self.model.get_layer(layer_name)
            # Manejar caso especial para InputLayer
            if isinstance(layer, tf.keras.layers.InputLayer):
                # Para InputLayer, intentamos diferentes atributos
                if hasattr(layer, 'batch_input_shape'):
                    shape = layer.batch_input_shape
                elif hasattr(layer, 'input_shape'):
                    shape = layer.input_shape
                else:
                    # Si no hay información de forma, usar una forma genérica
                    shape = (None, None, None, None)
            else:
                # Manejar caso donde output_shape puede no estar disponible
                try:
                    shape = layer.output_shape
                except AttributeError:
                    # Si output_shape no está disponible, intentar obtener la forma de otra manera
                    if hasattr(layer, 'shape'):
                        shape = layer.shape
                    else:
                        shape = None
                
            return {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'shape': shape,
                'params': layer.count_params()
            }
        else:
            result = {}
            for layer in self.model.layers:
                # Manejar caso especial para InputLayer
                if isinstance(layer, tf.keras.layers.InputLayer):
                    # Para InputLayer, intentamos diferentes atributos
                    if hasattr(layer, 'batch_input_shape'):
                        shape = layer.batch_input_shape
                    elif hasattr(layer, 'input_shape'):
                        shape = layer.input_shape
                    else:
                        # Si no hay información de forma, usar una forma genérica
                        shape = (None, None, None, None)
                else:
                    # Manejar caso donde output_shape puede no estar disponible
                    try:
                        shape = layer.output_shape
                    except AttributeError:
                        # Si output_shape no está disponible, intentar obtener la forma de otra manera
                        if hasattr(layer, 'shape'):
                            shape = layer.shape
                        else:
                            shape = None
                    
                result[layer.name] = {
                    'type': layer.__class__.__name__,
                    'shape': shape,
                    'params': layer.count_params()
                }
            return result
