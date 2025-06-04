#!/usr/bin/env python3
"""
Script para generar imágenes de ejemplo para la documentación.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Asegurar que el directorio de imágenes existe
os.makedirs('docs/images', exist_ok=True)

# Cargar un modelo pre-entrenado
model = tf.keras.applications.VGG16(weights='imagenet')

# Cargar una imagen de ejemplo
img_path = tf.keras.utils.get_file(
    'elephant.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/elephant.jpg')
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

# Crear un modelo para extraer activaciones
layer_name = 'block3_conv1'
feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer(layer_name).output
)

# Obtener activaciones
features = feature_extractor(img_array)
features = features[0]

# Crear una imagen de ejemplo para la documentación
plt.figure(figsize=(12, 8))

# Panel izquierdo: imagen original
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Imagen de entrada')
plt.axis('off')

# Panel derecho: activaciones
plt.subplot(1, 2, 2)
# Mostrar solo los primeros 64 filtros en una cuadrícula de 8x8
n_filters = 64
size = 8
fig = plt.figure(figsize=(12, 8))

for i in range(n_filters):
    ax = fig.add_subplot(size, size, i + 1)
    
    # Obtener activación para este filtro
    feature = features[:, :, i]
    
    # Normalizar para visualización
    feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
    
    # Mostrar activación
    ax.imshow(feature, cmap='viridis')
    ax.axis('off')

plt.tight_layout()
plt.suptitle(f'Activaciones de la capa {layer_name}', fontsize=16)
plt.subplots_adjust(top=0.9)

# Guardar imagen
plt.savefig('docs/images/main_screen.png', dpi=150, bbox_inches='tight')
print("Imagen de ejemplo generada en docs/images/main_screen.png")
