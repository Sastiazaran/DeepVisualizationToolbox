"""
Funciones misceláneas de utilidad.
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Union, Optional, Any
import time
import matplotlib.pyplot as plt


def get_available_models() -> Dict[str, Dict]:
    """
    Obtiene una lista de modelos pre-entrenados disponibles en TensorFlow.
    
    Returns:
        Diccionario con información de modelos disponibles
    """
    models = {
        'vgg16': {
            'module': tf.keras.applications.vgg16,
            'input_shape': (224, 224, 3),
            'preprocess': tf.keras.applications.vgg16.preprocess_input
        },
        'vgg19': {
            'module': tf.keras.applications.vgg19,
            'input_shape': (224, 224, 3),
            'preprocess': tf.keras.applications.vgg19.preprocess_input
        },
        'resnet50': {
            'module': tf.keras.applications.resnet50,
            'input_shape': (224, 224, 3),
            'preprocess': tf.keras.applications.resnet50.preprocess_input
        },
        'inception_v3': {
            'module': tf.keras.applications.inception_v3,
            'input_shape': (299, 299, 3),
            'preprocess': tf.keras.applications.inception_v3.preprocess_input
        },
        'mobilenet': {
            'module': tf.keras.applications.mobilenet,
            'input_shape': (224, 224, 3),
            'preprocess': tf.keras.applications.mobilenet.preprocess_input
        },
        'efficientnet_b0': {
            'module': tf.keras.applications.efficientnet.EfficientNetB0,
            'input_shape': (224, 224, 3),
            'preprocess': tf.keras.applications.efficientnet.preprocess_input
        }
    }
    
    return models


def load_model(model_name: str, weights: str = 'imagenet', 
              include_top: bool = True) -> tf.keras.Model:
    """
    Carga un modelo pre-entrenado.
    
    Args:
        model_name: Nombre del modelo
        weights: Pesos a utilizar ('imagenet' o None)
        include_top: Si se incluye la capa de clasificación
        
    Returns:
        Modelo cargado
    """
    available_models = get_available_models()
    
    if model_name.lower() not in available_models:
        raise ValueError(f"Modelo {model_name} no disponible. Opciones: {list(available_models.keys())}")
    
    model_info = available_models[model_name.lower()]
    
    # Cargar modelo
    if model_name.lower() == 'efficientnet_b0':
        # EfficientNet tiene una API ligeramente diferente
        model = model_info['module'](
            weights=weights,
            include_top=include_top
        )
    else:
        model = model_info['module'].Model(
            weights=weights,
            include_top=include_top
        )
    
    return model


def get_imagenet_labels() -> List[str]:
    """
    Obtiene las etiquetas de ImageNet.
    
    Returns:
        Lista de etiquetas
    """
    # Cargar etiquetas de ImageNet
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    
    with open(labels_path) as f:
        labels = f.readlines()
    
    return [label.strip() for label in labels]


def predict_image(model: tf.keras.Model, img: np.ndarray, 
                 top_k: int = 5) -> List[Tuple[int, str, float]]:
    """
    Realiza una predicción con un modelo y devuelve las clases top-k.
    
    Args:
        model: Modelo TensorFlow
        img: Imagen preprocesada
        top_k: Número de clases top a devolver
        
    Returns:
        Lista de tuplas (índice, etiqueta, probabilidad)
    """
    # Asegurar que la imagen tiene la forma correcta
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    # Realizar predicción
    preds = model.predict(img)
    
    # Obtener índices de las clases top-k
    top_indices = np.argsort(preds[0])[-top_k:][::-1]
    
    # Obtener etiquetas
    labels = get_imagenet_labels()
    
    # Crear lista de resultados
    results = [(idx, labels[idx], float(preds[0][idx])) for idx in top_indices]
    
    return results


def time_execution(func: callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Mide el tiempo de ejecución de una función.
    
    Args:
        func: Función a medir
        *args, **kwargs: Argumentos para la función
        
    Returns:
        Tupla (resultado de la función, tiempo en segundos)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time


def save_visualizations(visualizations: Dict[str, np.ndarray], 
                       output_dir: str, prefix: str = ''):
    """
    Guarda visualizaciones en disco.
    
    Args:
        visualizations: Diccionario de visualizaciones
        output_dir: Directorio de salida
        prefix: Prefijo para los nombres de archivo
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar cada visualización
    for name, vis in visualizations.items():
        # Crear nombre de archivo
        filename = f"{prefix}_{name}.png" if prefix else f"{name}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Normalizar si es necesario
        if vis.dtype != np.uint8:
            vis = np.clip(vis * 255, 0, 255).astype(np.uint8)
        
        # Guardar imagen
        plt.imsave(filepath, vis)
    
    print(f"Visualizaciones guardadas en {output_dir}")


def create_model_summary(model: tf.keras.Model) -> Dict:
    """
    Crea un resumen detallado del modelo.
    
    Args:
        model: Modelo TensorFlow
        
    Returns:
        Diccionario con información del modelo
    """
    # Información básica
    summary = {
        'name': model.name,
        'layers': [],
        'total_params': model.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    }
    
    # Información de capas
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape),
            'params': layer.count_params(),
            'trainable': layer.trainable
        }
        
        # Añadir información específica según el tipo de capa
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer_info.update({
                'filters': layer.filters,
                'kernel_size': layer.kernel_size,
                'strides': layer.strides,
                'padding': layer.padding,
                'activation': str(layer.activation.__name__) if layer.activation else None
            })
        elif isinstance(layer, tf.keras.layers.Dense):
            layer_info.update({
                'units': layer.units,
                'activation': str(layer.activation.__name__) if layer.activation else None
            })
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            layer_info.update({
                'pool_size': layer.pool_size,
                'strides': layer.strides,
                'padding': layer.padding
            })
        
        summary['layers'].append(layer_info)
    
    return summary
