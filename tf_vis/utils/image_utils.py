"""
Utilidades para procesamiento de imágenes.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Union
from PIL import Image
import tensorflow as tf


def load_image(path: str, target_size: Optional[Tuple[int, int]] = None, 
              preprocess_fn: Optional[callable] = None) -> np.ndarray:
    """
    Carga una imagen desde un archivo.
    
    Args:
        path: Ruta al archivo de imagen
        target_size: Tamaño objetivo (ancho, alto)
        preprocess_fn: Función de preprocesamiento opcional
        
    Returns:
        Imagen como array numpy
    """
    # Cargar imagen
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    
    # Convertir de BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar si se especifica un tamaño
    if target_size:
        img = cv2.resize(img, target_size)
    
    # Aplicar preprocesamiento si se proporciona
    if preprocess_fn:
        img = preprocess_fn(img)
    
    return img


def preprocess_image_for_model(img: np.ndarray, model_name: str) -> np.ndarray:
    """
    Preprocesa una imagen para un modelo específico.
    
    Args:
        img: Imagen como array numpy
        model_name: Nombre del modelo ('vgg16', 'resnet50', etc.)
        
    Returns:
        Imagen preprocesada
    """
    # Asegurar que la imagen tiene 3 canales
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    
    # Preprocesamiento específico del modelo
    if model_name.lower() == 'vgg16':
        # VGG16 espera valores en [0, 255] y luego resta la media RGB
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input(img.copy())
    
    elif model_name.lower() == 'resnet50':
        # ResNet50 espera valores en [0, 255] y luego normaliza con media/std de ImageNet
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(img.copy())
    
    elif model_name.lower() == 'inception_v3':
        # InceptionV3 espera valores en [-1, 1]
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        return preprocess_input(img.copy())
    
    elif model_name.lower() == 'mobilenet':
        # MobileNet espera valores en [-1, 1]
        from tensorflow.keras.applications.mobilenet import preprocess_input
        return preprocess_input(img.copy())
    
    else:
        # Preprocesamiento genérico: normalizar a [0, 1]
        return img.astype(np.float32) / 255.0


def deprocess_image(img: np.ndarray, model_name: str = None) -> np.ndarray:
    """
    Convierte una imagen preprocesada de vuelta a formato visualizable.
    
    Args:
        img: Imagen preprocesada
        model_name: Nombre del modelo (opcional)
        
    Returns:
        Imagen en formato visualizable (valores en [0, 1])
    """
    # Si no se especifica modelo, asumir valores en [0, 1]
    if model_name is None:
        return np.clip(img, 0, 1)
    
    # Deshacer preprocesamiento específico del modelo
    if model_name.lower() == 'vgg16':
        # Deshacer resta de media
        mean = [103.939, 116.779, 123.68]
        img = img.copy()
        
        # Convertir de BGR a RGB
        img = img[..., ::-1]
        
        # Añadir media
        for i in range(3):
            img[..., i] += mean[i]
        
        # Normalizar a [0, 1]
        return np.clip(img / 255.0, 0, 1)
    
    elif model_name.lower() in ['inception_v3', 'mobilenet']:
        # Convertir de [-1, 1] a [0, 1]
        return np.clip((img + 1) / 2.0, 0, 1)
    
    else:
        # Normalizar a [0, 1]
        return np.clip(img, 0, 1)


def apply_gradcam(model: tf.keras.Model, img: np.ndarray, 
                 layer_name: str, class_idx: int) -> np.ndarray:
    """
    Aplica Grad-CAM para visualizar qué partes de la imagen son importantes.
    
    Args:
        model: Modelo TensorFlow
        img: Imagen de entrada
        layer_name: Nombre de la capa para Grad-CAM
        class_idx: Índice de la clase a visualizar
        
    Returns:
        Mapa de calor Grad-CAM
    """
    # Crear modelo que devuelve tanto la predicción como las activaciones
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Asegurar que la imagen tiene la forma correcta
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    # Registrar operaciones para el cálculo de gradientes
    with tf.GradientTape() as tape:
        # Ejecutar pase hacia adelante
        conv_outputs, predictions = grad_model(img)
        
        # Obtener pérdida para la clase objetivo
        loss = predictions[:, class_idx]
    
    # Calcular gradientes
    grads = tape.gradient(loss, conv_outputs)
    
    # Calcular pesos de importancia
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiplicar cada canal por su peso de importancia
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Aplicar ReLU
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Convertir a numpy
    heatmap = heatmap.numpy()
    
    return heatmap


def overlay_gradcam(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Superpone un mapa de calor Grad-CAM en una imagen.
    
    Args:
        img: Imagen original
        heatmap: Mapa de calor Grad-CAM
        alpha: Factor de mezcla
        
    Returns:
        Imagen con mapa de calor superpuesto
    """
    # Redimensionar mapa de calor al tamaño de la imagen
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convertir a mapa de colores
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Asegurar que img está en formato uint8 con rango [0, 255]
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # Superponer mapa de calor
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    return superimposed


def create_grid_of_images(images: List[np.ndarray], grid_size: Optional[Tuple[int, int]] = None,
                         padding: int = 1) -> np.ndarray:
    """
    Crea una cuadrícula de imágenes.
    
    Args:
        images: Lista de imágenes
        grid_size: Tamaño de la cuadrícula (filas, columnas)
        padding: Píxeles de padding entre imágenes
        
    Returns:
        Imagen con la cuadrícula
    """
    n_images = len(images)
    
    # Determinar tamaño de cuadrícula si no se proporciona
    if grid_size is None:
        grid_size = (int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images))))
    
    # Asegurarse de que la cuadrícula puede contener todas las imágenes
    if grid_size[0] * grid_size[1] < n_images:
        grid_size = (int(np.ceil(np.sqrt(n_images))), int(np.ceil(np.sqrt(n_images))))
    
    # Determinar tamaño de imagen
    h, w = images[0].shape[:2]
    
    # Crear imagen de cuadrícula
    grid_h = grid_size[0] * h + (grid_size[0] - 1) * padding
    grid_w = grid_size[1] * w + (grid_size[1] - 1) * padding
    
    # Determinar número de canales
    if len(images[0].shape) == 3:
        channels = images[0].shape[2]
        grid = np.zeros((grid_h, grid_w, channels), dtype=images[0].dtype)
    else:
        grid = np.zeros((grid_h, grid_w), dtype=images[0].dtype)
    
    # Llenar la cuadrícula con imágenes
    for i in range(min(n_images, grid_size[0] * grid_size[1])):
        row = i // grid_size[1]
        col = i % grid_size[1]
        
        # Calcular posición en la cuadrícula
        y = row * (h + padding)
        x = col * (w + padding)
        
        # Colocar imagen en la cuadrícula
        if len(images[i].shape) == 3:
            grid[y:y+h, x:x+w, :] = images[i]
        else:
            grid[y:y+h, x:x+w] = images[i]
    
    return grid
