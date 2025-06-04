"""
Aplicación principal para la visualización de características de redes neuronales.
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication

from .model_wrapper import ModelWrapper
from .input_fetcher import InputFetcher
from .ui.main_window import MainWindow
from .utils.misc import load_model, get_available_models


def parse_args():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='TensorFlow Feature Visualization Toolbox')
    
    # Argumentos para el modelo
    parser.add_argument('--model', type=str, default='vgg16',
                       help='Modelo a utilizar (vgg16, resnet50, etc.)')
    parser.add_argument('--weights', type=str, default='imagenet',
                       help='Pesos a utilizar (imagenet o ruta a archivo)')
    parser.add_argument('--include-top', action='store_true', default=True,
                       help='Incluir capas de clasificación')
    
    # Argumentos para la entrada
    parser.add_argument('--input-source', type=str, default='webcam',
                       help='Fuente de entrada (webcam, directory:path, file:path, dataset:name)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[224, 224],
                       help='Tamaño de entrada (ancho, alto)')
    
    # Argumentos para la visualización
    parser.add_argument('--gpu', action='store_true',
                       help='Usar GPU para cálculos')
    
    return parser.parse_args()


def main():
    """Función principal de la aplicación."""
    # Analizar argumentos
    args = parse_args()
    
    # Configurar GPU/CPU
    if not args.gpu:
        print("Ejecutando en CPU (use --gpu para habilitar GPU)")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Cargar modelo
    try:
        print(f"Cargando modelo {args.model}...")
        model = load_model(args.model, weights=args.weights, include_top=args.include_top)
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        available_models = get_available_models()
        print(f"Modelos disponibles: {list(available_models.keys())}")
        sys.exit(1)
    
    # Crear wrapper del modelo
    model_wrapper = ModelWrapper(model)
    
    # Crear fetcher de entrada
    try:
        # Obtener función de preprocesamiento para el modelo
        available_models = get_available_models()
        preprocess_fn = available_models[args.model.lower()]['preprocess']
        
        # Modificar la fuente de entrada si se especificó webcam pero no hay acceso
        input_source = args.input_source
        if input_source == 'webcam':
            # Verificar si podemos acceder a la webcam
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("No se pudo acceder a la webcam, usando imágenes de prueba")
                    # Crear directorio temporal para imágenes de prueba si no existe
                    if not os.path.exists('input_images'):
                        os.makedirs('input_images')
                    # Crear algunas imágenes de prueba
                    for i in range(5):
                        img = np.random.random((args.input_size[1], args.input_size[0], 3))
                        img = (img * 255).astype(np.uint8)
                        cv2.imwrite(f'input_images/test_{i}.jpg', img)
                    input_source = 'directory:input_images'
                cap.release()
            except Exception as e:
                print(f"Error al verificar webcam: {e}, usando imágenes de prueba")
                # Crear directorio temporal para imágenes de prueba si no existe
                if not os.path.exists('input_images'):
                    os.makedirs('input_images')
                # Crear algunas imágenes de prueba
                for i in range(5):
                    img = np.random.random((args.input_size[1], args.input_size[0], 3))
                    img = (img * 255).astype(np.uint8)
                    cv2.imwrite(f'input_images/test_{i}.jpg', img)
                input_source = 'directory:input_images'
        
        # Crear fetcher
        input_fetcher = InputFetcher(
            input_source=input_source,
            preprocessing_function=preprocess_fn,
            target_size=tuple(args.input_size)
        )
        print(f"Fuente de entrada configurada: {input_source}")
    except Exception as e:
        print(f"Error al configurar la fuente de entrada: {e}")
        sys.exit(1)
    
    # Iniciar aplicación Qt
    app = QApplication(sys.argv)
    
    # Crear ventana principal
    window = MainWindow(model_wrapper, input_fetcher)
    
    # Ejecutar aplicación
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
