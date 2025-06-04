"""
Métodos para visualizar características y activaciones de redes neuronales.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
import cv2
from scipy.ndimage import zoom


def display_activation_grid(activations: np.ndarray, grid_size: Optional[Tuple[int, int]] = None, 
                           padding: int = 1) -> np.ndarray:
    """
    Organiza las activaciones en una cuadrícula para visualización.
    
    Args:
        activations: Array de activaciones con forma [n_filters, height, width]
        grid_size: Tamaño de la cuadrícula (filas, columnas)
        padding: Píxeles de padding entre imágenes
        
    Returns:
        Imagen con la cuadrícula de activaciones
    """
    # Obtener número de filtros y dimensiones
    n_filters, height, width = activations.shape
    
    # Determinar tamaño de cuadrícula si no se proporciona
    if grid_size is None:
        grid_size = (int(np.ceil(np.sqrt(n_filters))), int(np.ceil(np.sqrt(n_filters))))
    
    # Asegurarse de que la cuadrícula puede contener todas las activaciones
    if grid_size[0] * grid_size[1] < n_filters:
        grid_size = (int(np.ceil(np.sqrt(n_filters))), int(np.ceil(np.sqrt(n_filters))))
    
    # Crear imagen de cuadrícula
    grid_height = grid_size[0] * height + (grid_size[0] - 1) * padding
    grid_width = grid_size[1] * width + (grid_size[1] - 1) * padding
    grid = np.zeros((grid_height, grid_width))
    
    # Llenar la cuadrícula con activaciones
    filter_idx = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if filter_idx < n_filters:
                # Calcular posición en la cuadrícula
                row_start = i * (height + padding)
                col_start = j * (width + padding)
                
                # Normalizar activación para visualización
                activation = activations[filter_idx]
                activation = (activation - np.min(activation)) / (np.max(activation) - np.min(activation) + 1e-8)
                
                # Colocar en la cuadrícula
                grid[row_start:row_start + height, col_start:col_start + width] = activation
                
                filter_idx += 1
    
    return grid


def visualize_layer_filters(model: tf.keras.Model, layer_name: str, 
                           grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Visualiza los filtros de una capa convolucional.
    
    Args:
        model: Modelo TensorFlow
        layer_name: Nombre de la capa
        grid_size: Tamaño de la cuadrícula (filas, columnas)
        
    Returns:
        Imagen con la cuadrícula de filtros
    """
    # Obtener la capa
    layer = model.get_layer(layer_name)
    
    # Verificar que es una capa convolucional
    if not isinstance(layer, tf.keras.layers.Conv2D):
        raise ValueError(f"La capa {layer_name} no es una capa convolucional")
    
    # Obtener los pesos (filtros)
    weights = layer.get_weights()[0]
    
    # Normalizar los pesos para visualización
    weights_min, weights_max = weights.min(), weights.max()
    filters = (weights - weights_min) / (weights_max - weights_min + 1e-8)
    
    # Organizar los filtros para visualización
    n_filters, height, width, n_channels = filters.shape
    
    # Para filtros multicanal, promediar sobre los canales
    if n_channels > 1:
        filters = np.mean(filters, axis=3)
    else:
        filters = filters[:, :, :, 0]
    
    # Crear cuadrícula
    return display_activation_grid(filters, grid_size)


def apply_gradient_ascent(model_wrapper, layer_name: str, filter_index: int, 
                         iterations: int = 30, step_size: float = 1.0,
                         image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Aplica ascenso de gradiente para generar una imagen que maximiza la activación
    de un filtro específico.
    
    Args:
        model_wrapper: Wrapper del modelo
        layer_name: Nombre de la capa objetivo
        filter_index: Índice del filtro a maximizar
        iterations: Número de iteraciones de optimización
        step_size: Tamaño del paso para el ascenso de gradiente
        image_size: Tamaño de la imagen a generar
        
    Returns:
        Imagen generada que maximiza la activación
    """
    # Inicializar imagen con ruido aleatorio
    img = np.random.random((1, image_size[0], image_size[1], 3)) * 0.1
    
    # Convertir a tensor de TensorFlow
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    
    # Iteraciones de ascenso de gradiente
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            
            # Obtener activación del filtro específico
            layer_output = model_wrapper._layer_outputs[layer_name](img_tensor)
            
            # Si es una capa convolucional, tomar el máximo espacial
            if len(layer_output.shape) == 4:
                # [batch, height, width, channels]
                loss = tf.reduce_mean(layer_output[:, :, :, filter_index])
            else:
                # Para capas densas
                loss = layer_output[:, filter_index]
        
        # Calcular gradientes
        grads = tape.gradient(loss, img_tensor)
        
        # Normalizar gradientes
        grads = tf.math.l2_normalize(grads)
        
        # Actualizar imagen
        img_tensor = img_tensor + step_size * grads
        
        # Opcional: aplicar restricciones para mantener la imagen natural
        img_tensor = tf.clip_by_value(img_tensor, 0, 1)
    
    # Convertir de vuelta a numpy
    img = img_tensor.numpy()[0]
    
    return img


def visualize_max_activations(model_wrapper, dataset, layer_name: str, 
                             n_top: int = 9, n_filters: Optional[int] = None) -> Dict:
    """
    Encuentra las imágenes que causan las activaciones máximas para cada filtro.
    
    Args:
        model_wrapper: Wrapper del modelo
        dataset: Dataset de imágenes para buscar activaciones máximas
        layer_name: Nombre de la capa
        n_top: Número de imágenes top a guardar por filtro
        n_filters: Número de filtros a procesar (None = todos)
        
    Returns:
        Diccionario con las imágenes de máxima activación por filtro
    """
    # Obtener información de la capa
    layer_info = model_wrapper.get_layer_info(layer_name)
    
    # Determinar número de filtros
    if len(layer_info['shape']) == 4:  # Capa convolucional
        total_filters = layer_info['shape'][-1]
    else:  # Capa densa
        total_filters = layer_info['shape'][-1]
    
    if n_filters is None:
        n_filters = total_filters
    else:
        n_filters = min(n_filters, total_filters)
    
    # Inicializar estructura para almacenar resultados
    max_activations = {i: {'values': np.zeros(n_top), 'images': [None] * n_top} 
                      for i in range(n_filters)}
    
    # Procesar cada imagen en el dataset
    for img_data in dataset:
        # Extraer imagen y posiblemente etiqueta
        if isinstance(img_data, tuple):
            img = img_data[0]  # (imagen, etiqueta)
        else:
            img = img_data
        
        # Obtener activaciones
        activations = model_wrapper.forward_pass(img, layer_name)[layer_name]
        
        # Para cada filtro, verificar si esta imagen produce activaciones más altas
        for filter_idx in range(n_filters):
            if len(activations.shape) == 4:  # Capa convolucional
                # Tomar el máximo espacial
                filter_activation = np.max(activations[0, :, :, filter_idx])
            else:  # Capa densa
                filter_activation = activations[0, filter_idx]
            
            # Verificar si esta activación es mayor que alguna de las top
            min_idx = np.argmin(max_activations[filter_idx]['values'])
            if filter_activation > max_activations[filter_idx]['values'][min_idx]:
                # Reemplazar la activación mínima actual
                max_activations[filter_idx]['values'][min_idx] = filter_activation
                max_activations[filter_idx]['images'][min_idx] = img.copy()
                
                # Reordenar para mantener el orden descendente
                sort_idx = np.argsort(-max_activations[filter_idx]['values'])
                max_activations[filter_idx]['values'] = max_activations[filter_idx]['values'][sort_idx]
                max_activations[filter_idx]['images'] = [max_activations[filter_idx]['images'][i] for i in sort_idx]
    
    return max_activations


def create_class_activation_map(model_wrapper, img: np.ndarray, 
                               layer_name: str, class_idx: int) -> np.ndarray:
    """
    Crea un mapa de activación de clase (CAM) para visualizar qué partes de la imagen
    son importantes para una clase específica.
    
    Args:
        model_wrapper: Wrapper del modelo
        img: Imagen de entrada
        layer_name: Nombre de la última capa convolucional
        class_idx: Índice de la clase a visualizar
        
    Returns:
        Mapa de calor CAM
    """
    # Asegurar que la imagen tiene la forma correcta
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    # Obtener activaciones de la capa convolucional
    activations = model_wrapper.forward_pass(img, layer_name)[layer_name][0]
    
    # Obtener predicciones
    predictions = model_wrapper.model.predict(img)
    
    # Obtener pesos de la capa densa final para la clase específica
    # Esto asume una arquitectura típica donde la última capa densa sigue a la convolucional
    # En una implementación real, se necesitaría adaptar esto a la arquitectura específica
    final_layer = model_wrapper.model.layers[-1]
    weights = final_layer.get_weights()[0]
    
    # Obtener pesos para la clase específica
    class_weights = weights[:, class_idx]
    
    # Crear mapa de activación de clase
    cam = np.zeros(activations.shape[0:2], dtype=np.float32)
    
    # Multiplicar activaciones por pesos y sumar
    for i, w in enumerate(class_weights):
        cam += w * activations[:, :, i]
    
    # Aplicar ReLU
    cam = np.maximum(cam, 0)
    
    # Normalizar
    cam = cam / (np.max(cam) + 1e-10)
    
    # Redimensionar al tamaño de la imagen original
    cam = cv2.resize(cam, (img.shape[2], img.shape[1]))
    
    return cam


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, 
                   alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Superpone un mapa de calor en una imagen.
    
    Args:
        img: Imagen original
        heatmap: Mapa de calor (valores entre 0 y 1)
        alpha: Factor de mezcla
        colormap: Mapa de colores a utilizar
        
    Returns:
        Imagen con mapa de calor superpuesto
    """
    # Convertir mapa de calor a mapa de colores
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    
    # Convertir a RGB si es necesario
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    # Asegurar que img está en formato uint8 con rango [0, 255]
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # Superponer mapa de calor
    superimposed = cv2.addWeighted(img, 1.0 - alpha, heatmap_colored, alpha, 0)
    
    return superimposed
