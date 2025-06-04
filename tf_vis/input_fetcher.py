"""
Módulo para cargar y preprocesar imágenes de diferentes fuentes.
"""

import os
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union, Dict, Any
import tensorflow as tf
from PIL import Image


class InputFetcher:
    """
    Clase para cargar imágenes de diferentes fuentes:
    - Archivos de imagen
    - Directorio de imágenes
    - Webcam
    - Dataset de TensorFlow
    """
    
    def __init__(self, input_source: str = 'webcam', 
                preprocessing_function: Optional[callable] = None,
                target_size: Tuple[int, int] = (224, 224)):
        """
        Inicializa el fetcher de entrada.
        
        Args:
            input_source: Fuente de entrada ('webcam', 'directory', 'file', 'dataset')
            preprocessing_function: Función de preprocesamiento opcional
            target_size: Tamaño objetivo para las imágenes
        """
        self.input_source = input_source
        self.preprocessing_function = preprocessing_function
        self.target_size = target_size
        self.current_index = 0
        self.images = []
        self.image_paths = []
        self.webcam = None
        self.dataset = None
        
        # Inicializar la fuente de entrada
        self._initialize_source()
    
    def _initialize_source(self):
        """Inicializa la fuente de entrada según el tipo."""
        if self.input_source == 'webcam':
            self._initialize_webcam()
        elif self.input_source.startswith('directory:'):
            directory = self.input_source.split(':', 1)[1]
            self._load_directory(directory)
        elif self.input_source.startswith('file:'):
            file_path = self.input_source.split(':', 1)[1]
            self._load_file(file_path)
        elif self.input_source.startswith('dataset:'):
            dataset_name = self.input_source.split(':', 1)[1]
            self._load_dataset(dataset_name)
    
    def _initialize_webcam(self, device_id: int = 0):
        """Inicializa la webcam."""
        try:
            self.webcam = cv2.VideoCapture(device_id)
            if not self.webcam.isOpened():
                print(f"Advertencia: No se pudo abrir la webcam con ID {device_id}")
                # Crear una imagen de prueba en lugar de usar la webcam
                self.use_test_image = True
                self.test_image = np.random.random((self.target_size[1], self.target_size[0], 3))
                self.test_image = (self.test_image * 255).astype(np.uint8)
            else:
                self.use_test_image = False
        except Exception as e:
            print(f"Error al inicializar webcam: {e}")
            # Crear una imagen de prueba en lugar de usar la webcam
            self.use_test_image = True
            self.test_image = np.random.random((self.target_size[1], self.target_size[0], 3))
            self.test_image = (self.test_image * 255).astype(np.uint8)
    
    def _load_directory(self, directory: str):
        """Carga imágenes desde un directorio."""
        if not os.path.isdir(directory):
            raise ValueError(f"El directorio {directory} no existe")
        
        # Obtener todas las imágenes en el directorio
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        self.image_paths = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(valid_extensions)
        ]
        
        if not self.image_paths:
            raise ValueError(f"No se encontraron imágenes en {directory}")
        
        # Precargar la primera imagen
        self._load_image_at_index(0)
    
    def _load_file(self, file_path: str):
        """Carga una imagen desde un archivo."""
        if not os.path.isfile(file_path):
            raise ValueError(f"El archivo {file_path} no existe")
        
        self.image_paths = [file_path]
        self._load_image_at_index(0)
    
    def _load_dataset(self, dataset_name: str):
        """Carga un dataset de TensorFlow."""
        # Implementar carga de datasets comunes
        if dataset_name == 'cifar10':
            (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
            self.images = x_train
        elif dataset_name == 'mnist':
            (x_train, _), _ = tf.keras.datasets.mnist.load_data()
            # Convertir a RGB
            x_rgb = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2], 3), dtype=np.uint8)
            for i in range(x_train.shape[0]):
                x_rgb[i] = cv2.cvtColor(x_train[i], cv2.COLOR_GRAY2RGB)
            self.images = x_rgb
        else:
            raise ValueError(f"Dataset {dataset_name} no soportado")
    
    def _load_image_at_index(self, index: int):
        """Carga la imagen en el índice especificado."""
        if 0 <= index < len(self.image_paths):
            img_path = self.image_paths[index]
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"No se pudo cargar la imagen {img_path}")
            
            # Convertir de BGR a RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar
            img = cv2.resize(img, self.target_size)
            
            self.current_index = index
            self.current_image = img
    
    def get_next_image(self) -> np.ndarray:
        """
        Obtiene la siguiente imagen de la fuente.
        
        Returns:
            Imagen como array numpy
        """
        if self.input_source == 'webcam':
            return self._get_webcam_frame()
        elif self.input_source.startswith('directory:') or self.input_source.startswith('file:'):
            return self._get_next_file_image()
        elif self.input_source.startswith('dataset:'):
            return self._get_next_dataset_image()
        else:
            raise ValueError(f"Fuente de entrada no válida: {self.input_source}")
    
    def _get_webcam_frame(self) -> np.ndarray:
        """Captura y devuelve un frame de la webcam o una imagen de prueba."""
        if hasattr(self, 'use_test_image') and self.use_test_image:
            # Devolver imagen de prueba si la webcam no está disponible
            frame = self.test_image.copy()
        else:
            # Intentar capturar frame de la webcam
            try:
                ret, frame = self.webcam.read()
                if not ret:
                    print("No se pudo capturar frame de la webcam, usando imagen de prueba")
                    # Crear una imagen de prueba
                    frame = np.random.random((self.target_size[1], self.target_size[0], 3))
                    frame = (frame * 255).astype(np.uint8)
                else:
                    # Convertir de BGR a RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Redimensionar
                    frame = cv2.resize(frame, self.target_size)
            except Exception as e:
                print(f"Error al capturar frame: {e}")
                # Crear una imagen de prueba
                frame = np.random.random((self.target_size[1], self.target_size[0], 3))
                frame = (frame * 255).astype(np.uint8)
        
        # Aplicar preprocesamiento si está definido
        if self.preprocessing_function:
            frame = self.preprocessing_function(frame)
        
        return frame
    
    def _get_next_file_image(self) -> np.ndarray:
        """Obtiene la siguiente imagen de la lista de archivos."""
        if not self.image_paths:
            raise ValueError("No hay imágenes disponibles")
        
        # Avanzar al siguiente índice
        next_index = (self.current_index + 1) % len(self.image_paths)
        self._load_image_at_index(next_index)
        
        # Aplicar preprocesamiento si está definido
        if self.preprocessing_function:
            self.current_image = self.preprocessing_function(self.current_image)
        
        return self.current_image
    
    def _get_next_dataset_image(self) -> np.ndarray:
        """Obtiene la siguiente imagen del dataset."""
        if not self.images:
            raise ValueError("No hay imágenes disponibles en el dataset")
        
        # Avanzar al siguiente índice
        next_index = (self.current_index + 1) % len(self.images)
        self.current_index = next_index
        
        # Obtener imagen
        img = self.images[self.current_index]
        
        # Redimensionar si es necesario
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size)
        
        # Aplicar preprocesamiento si está definido
        if self.preprocessing_function:
            img = self.preprocessing_function(img)
        
        return img
    
    def get_previous_image(self) -> np.ndarray:
        """
        Obtiene la imagen anterior (solo para fuentes basadas en archivos o datasets).
        
        Returns:
            Imagen como array numpy
        """
        if self.input_source == 'webcam':
            return self._get_webcam_frame()
        elif self.input_source.startswith('directory:') or self.input_source.startswith('file:'):
            # Retroceder al índice anterior
            prev_index = (self.current_index - 1) % len(self.image_paths)
            self._load_image_at_index(prev_index)
            
            # Aplicar preprocesamiento si está definido
            if self.preprocessing_function:
                self.current_image = self.preprocessing_function(self.current_image)
            
            return self.current_image
        elif self.input_source.startswith('dataset:'):
            # Retroceder al índice anterior
            prev_index = (self.current_index - 1) % len(self.images)
            self.current_index = prev_index
            
            # Obtener imagen
            img = self.images[self.current_index]
            
            # Redimensionar si es necesario
            if img.shape[:2] != self.target_size:
                img = cv2.resize(img, self.target_size)
            
            # Aplicar preprocesamiento si está definido
            if self.preprocessing_function:
                img = self.preprocessing_function(img)
            
            return img
    
    def get_specific_image(self, index: int) -> np.ndarray:
        """
        Obtiene una imagen específica por índice.
        
        Args:
            index: Índice de la imagen
            
        Returns:
            Imagen como array numpy
        """
        if self.input_source == 'webcam':
            return self._get_webcam_frame()
        elif self.input_source.startswith('directory:') or self.input_source.startswith('file:'):
            if 0 <= index < len(self.image_paths):
                self._load_image_at_index(index)
                
                # Aplicar preprocesamiento si está definido
                if self.preprocessing_function:
                    self.current_image = self.preprocessing_function(self.current_image)
                
                return self.current_image
            else:
                raise IndexError(f"Índice {index} fuera de rango")
        elif self.input_source.startswith('dataset:'):
            if 0 <= index < len(self.images):
                self.current_index = index
                
                # Obtener imagen
                img = self.images[index]
                
                # Redimensionar si es necesario
                if img.shape[:2] != self.target_size:
                    img = cv2.resize(img, self.target_size)
                
                # Aplicar preprocesamiento si está definido
                if self.preprocessing_function:
                    img = self.preprocessing_function(img)
                
                return img
            else:
                raise IndexError(f"Índice {index} fuera de rango")
    
    def close(self):
        """Libera recursos."""
        if self.webcam is not None:
            self.webcam.release()
