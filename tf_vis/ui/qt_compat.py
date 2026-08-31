"""
Utilidades compartidas de Qt.

La interfaz usa PyQt6, donde los enumerados están anidados (por ejemplo
`Qt.AlignmentFlag.AlignCenter` en lugar de `Qt.AlignCenter`).
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
HORIZONTAL = Qt.Orientation.Horizontal
KEEP_ASPECT_RATIO = Qt.AspectRatioMode.KeepAspectRatio
SMOOTH_TRANSFORMATION = Qt.TransformationMode.SmoothTransformation


def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    """
    Convierte un array numpy en un `QPixmap` listo para mostrar.

    Acepta imágenes en escala de grises o RGB, en `uint8` o en coma flotante.

    QImage no copia el búfer que se le pasa, así que se construye sobre un array
    contiguo y se fuerza una copia con `QPixmap.fromImage`; de lo contrario el
    array temporal podría liberarse y dejar el pixmap apuntando a memoria
    inválida.
    """
    array = np.asarray(img)

    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

    array = np.ascontiguousarray(array)
    height, width = array.shape[:2]

    if array.ndim == 2:
        image = QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
    elif array.shape[2] == 4:
        image = QImage(array.data, width, height, 4 * width, QImage.Format.Format_RGBA8888)
    else:
        image = QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888)

    return QPixmap.fromImage(image.copy())
