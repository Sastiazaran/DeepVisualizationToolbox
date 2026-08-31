"""
Módulo para cargar y preprocesar imágenes de diferentes fuentes.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import cv2
import numpy as np

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


class InputFetcher:
    """
    Carga imágenes RGB desde distintas fuentes y las preprocesa para el modelo.

    Fuentes admitidas (indicadas mediante la cadena `input_source`):

    - ``webcam`` o ``webcam:<id>``: captura desde una cámara
    - ``file:<ruta>``: una sola imagen
    - ``directory:<ruta>``: todas las imágenes de un directorio
    - ``dataset:<nombre>``: ``cifar10`` o ``mnist``
    - ``synthetic``: ruido determinista, útil sin cámara ni imágenes en disco

    Cada llamada devuelve la imagen preprocesada, mientras que la versión RGB
    original queda accesible en :attr:`current_raw_image` para poder mostrarla
    en pantalla sin los desplazamientos que introduce el preprocesamiento.
    """

    def __init__(self, input_source: str = 'webcam',
                 preprocessing_function: Callable[[np.ndarray], np.ndarray] | None = None,
                 target_size: tuple[int, int] = (224, 224)):
        """
        Inicializa el fetcher de entrada.

        Args:
            input_source: Fuente de entrada
            preprocessing_function: Función de preprocesamiento opcional
            target_size: Tamaño objetivo `(ancho, alto)`
        """
        self.input_source = input_source
        self.preprocessing_function = preprocessing_function
        self.target_size = tuple(target_size)
        self.current_index = 0
        self.image_paths: list[str] = []
        self.images: np.ndarray | None = None
        self.webcam = None
        self.current_raw_image: np.ndarray | None = None

        self.kind, _, self.argument = input_source.partition(':')
        self._initialize_source()

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------
    def _initialize_source(self) -> None:
        """Inicializa la fuente de entrada según su tipo."""
        if self.kind == 'webcam':
            self._initialize_webcam(int(self.argument) if self.argument else 0)
        elif self.kind == 'directory':
            self._load_directory(self.argument)
        elif self.kind == 'file':
            self._load_file(self.argument)
        elif self.kind == 'dataset':
            self._load_dataset(self.argument)
        elif self.kind == 'synthetic':
            self.images = self._synthetic_images()
        else:
            raise ValueError(
                f"Fuente de entrada no válida: '{self.input_source}'. Usa webcam, "
                "file:<ruta>, directory:<ruta>, dataset:<nombre> o synthetic."
            )

    @property
    def is_live(self) -> bool:
        """Indica si la fuente produce imágenes nuevas continuamente."""
        return self.kind == 'webcam' and self.webcam is not None

    def _synthetic_images(self, count: int = 5) -> np.ndarray:
        """Genera imágenes de ruido reproducibles como fuente de reserva."""
        width, height = self.target_size
        rng = np.random.default_rng(0)
        return (rng.random((count, height, width, 3)) * 255).astype(np.uint8)

    def _initialize_webcam(self, device_id: int = 0) -> None:
        """Abre la webcam, cayendo a imágenes sintéticas si no está disponible."""
        try:
            webcam = cv2.VideoCapture(device_id)
        except cv2.error as error:
            print(f"Error al inicializar la webcam: {error}. Usando imágenes sintéticas.")
            webcam = None

        if webcam is not None and webcam.isOpened():
            self.webcam = webcam
            return

        if webcam is not None:
            webcam.release()
        print(f"Advertencia: no se pudo abrir la webcam {device_id}. Usando imágenes sintéticas.")
        self.images = self._synthetic_images()

    def _load_directory(self, directory: str) -> None:
        """Carga las rutas de las imágenes de un directorio."""
        if not os.path.isdir(directory):
            raise ValueError(f"El directorio {directory} no existe")

        self.image_paths = sorted(
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.lower().endswith(VALID_EXTENSIONS)
        )

        if not self.image_paths:
            raise ValueError(f"No se encontraron imágenes en {directory}")

    def _load_file(self, file_path: str) -> None:
        """Registra una única imagen como fuente."""
        if not os.path.isfile(file_path):
            raise ValueError(f"El archivo {file_path} no existe")
        self.image_paths = [file_path]

    def _load_dataset(self, dataset_name: str) -> None:
        """Carga un dataset incluido en Keras."""
        import keras

        if dataset_name == 'cifar10':
            (x_train, _), _ = keras.datasets.cifar10.load_data()
            self.images = x_train
        elif dataset_name == 'mnist':
            (x_train, _), _ = keras.datasets.mnist.load_data()
            self.images = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        else:
            raise ValueError(f"Dataset {dataset_name} no soportado (usa cifar10 o mnist)")

    # ------------------------------------------------------------------
    # Acceso a imágenes
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Número de imágenes disponibles (0 para fuentes en vivo)."""
        if self.image_paths:
            return len(self.image_paths)
        if self.images is not None:
            return len(self.images)
        return 0

    def get_next_image(self) -> np.ndarray:
        """Obtiene la siguiente imagen de la fuente, ya preprocesada."""
        if self.is_live:
            return self._finalize(self._capture_frame())
        return self.get_specific_image((self.current_index + 1) % max(len(self), 1))

    def get_previous_image(self) -> np.ndarray:
        """Obtiene la imagen anterior, ya preprocesada."""
        if self.is_live:
            return self._finalize(self._capture_frame())
        return self.get_specific_image((self.current_index - 1) % max(len(self), 1))

    def get_current_image(self) -> np.ndarray:
        """Obtiene de nuevo la imagen actual sin avanzar el índice."""
        if self.is_live:
            return self._finalize(self._capture_frame())
        return self.get_specific_image(self.current_index)

    def get_specific_image(self, index: int) -> np.ndarray:
        """
        Obtiene una imagen concreta por índice.

        Args:
            index: Índice de la imagen

        Returns:
            Imagen preprocesada
        """
        if self.is_live:
            return self._finalize(self._capture_frame())

        total = len(self)
        if total == 0:
            raise ValueError("No hay imágenes disponibles en la fuente de entrada")
        if not 0 <= index < total:
            raise IndexError(f"Índice {index} fuera de rango (0-{total - 1})")

        self.current_index = index

        if self.image_paths:
            raw = self._read_image_file(self.image_paths[index])
        else:
            raw = self._resize(np.asarray(self.images[index]))

        return self._finalize(raw)

    def _read_image_file(self, path: str) -> np.ndarray:
        """Lee una imagen de disco como RGB con el tamaño objetivo."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen {path}")
        return self._resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _capture_frame(self) -> np.ndarray:
        """Captura un frame de la webcam, con reserva sintética si falla."""
        try:
            ok, frame = self.webcam.read()
        except cv2.error as error:
            print(f"Error al capturar frame: {error}")
            ok, frame = False, None

        if not ok or frame is None:
            print("No se pudo capturar un frame de la webcam, usando imagen sintética")
            return self._synthetic_images(1)[0]

        return self._resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Redimensiona al tamaño objetivo `(ancho, alto)` si hace falta."""
        width, height = self.target_size
        if img.shape[1] == width and img.shape[0] == height:
            return img
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def _finalize(self, raw: np.ndarray) -> np.ndarray:
        """Guarda la imagen original y devuelve la versión preprocesada."""
        self.current_raw_image = raw
        if self.preprocessing_function is None:
            return raw
        # Se preprocesa una copia para que `current_raw_image` no se altere y para
        # que reprocesar la misma imagen no acumule transformaciones.
        return self.preprocessing_function(np.array(raw, dtype=np.float32, copy=True))

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Libera los recursos asociados a la fuente."""
        if self.webcam is not None:
            self.webcam.release()
            self.webcam = None

    def __enter__(self) -> InputFetcher:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
