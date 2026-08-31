"""
Widget para visualizar activaciones de capas.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..model_wrapper import ModelWrapper
from ..visualization import normalize_01
from .qt_compat import ALIGN_CENTER, KEEP_ASPECT_RATIO, SMOOTH_TRANSFORMATION, numpy_to_qpixmap

# Número máximo de filtros dibujados a la vez; por encima de esto la cuadrícula
# tarda más en construirse que en refrescarse la imagen de entrada.
MAX_TILES = 64
GRID_COLUMNS = 8


class LayerView(QWidget):
    """
    Cuadrícula con las activaciones de la capa seleccionada.
    """

    filter_clicked = pyqtSignal(int)

    def __init__(self, model_wrapper: ModelWrapper):
        """
        Inicializa el widget de visualización de capas.

        Args:
            model_wrapper: Wrapper del modelo
        """
        super().__init__()

        self.model_wrapper = model_wrapper
        self.current_layer: str | None = None
        self.current_activations: np.ndarray | None = None
        self._tiles: list[_ActivationTile] = []
        self._tile_layout: tuple[str, int] | None = None

        self.init_ui()

    def init_ui(self) -> None:
        """Construye la interfaz del widget."""
        layout = QVBoxLayout(self)

        self.layer_info_label = QLabel("Selecciona una capa para visualizar")
        self.layer_info_label.setAlignment(ALIGN_CENTER)
        layout.addWidget(self.layer_info_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        scroll_area.setWidget(self.grid_container)

        self.setMinimumSize(400, 300)

    def update_activations(self, activations: np.ndarray, layer_name: str) -> None:
        """
        Actualiza la cuadrícula con nuevas activaciones.

        Args:
            activations: Array de activaciones con la dimensión de lote
            layer_name: Nombre de la capa
        """
        self.current_layer = layer_name
        self.current_activations = activations

        info = self.model_wrapper.get_layer_info(layer_name)
        self.layer_info_label.setText(
            f"Capa: {layer_name} | Tipo: {info['type']} | Forma: {info['shape']}"
        )

        if activations.ndim == 4:
            count = min(MAX_TILES, activations.shape[-1])
            self._ensure_tiles(layer_name, count, dense=False)
            for i, tile in enumerate(self._tiles):
                tile.set_activation_map(normalize_01(activations[0, :, :, i]))
        elif activations.ndim == 2:
            count = min(MAX_TILES, activations.shape[-1])
            self._ensure_tiles(layer_name, count, dense=True)
            for i, tile in enumerate(self._tiles):
                tile.set_activation_value(float(activations[0, i]))
        else:
            self._clear_grid()

    def _ensure_tiles(self, layer_name: str, count: int, dense: bool) -> None:
        """
        Reutiliza los widgets existentes si la capa no ha cambiado.

        Reconstruir la cuadrícula en cada frame hacía que la interfaz parpadease
        y consumía CPU innecesariamente cuando la entrada es una webcam.
        """
        if self._tile_layout == (layer_name, count):
            return

        self._clear_grid()
        self._tile_layout = (layer_name, count)

        for i in range(count):
            tile = _ActivationTile(i, dense=dense)
            tile.clicked.connect(self.filter_clicked.emit)
            row, col = divmod(i, GRID_COLUMNS)
            self.grid_layout.addWidget(tile, row, col)
            self._tiles.append(tile)

    def _clear_grid(self) -> None:
        """Elimina todos los widgets de la cuadrícula."""
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._tiles.clear()
        self._tile_layout = None


class _ActivationTile(QWidget):
    """Celda de la cuadrícula que muestra un filtro o una neurona."""

    clicked = pyqtSignal(int)

    def __init__(self, index: int, dense: bool):
        super().__init__()

        self.index = index
        self.dense = dense
        self.value = 0.0

        self.setFixedSize(100, 120)
        self.setStyleSheet(
            "background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel(f"{'Neurona' if dense else 'Filtro'} {index}")
        title.setAlignment(ALIGN_CENTER)
        layout.addWidget(title)

        self.value_label = QLabel()
        self.value_label.setAlignment(ALIGN_CENTER)
        layout.addWidget(self.value_label)

        if dense:
            self.bar = _ValueBar()
            layout.addWidget(self.bar)
            self.image_label = None
        else:
            self.bar = None
            self.image_label = QLabel()
            self.image_label.setAlignment(ALIGN_CENTER)
            layout.addWidget(self.image_label)

    def set_activation_map(self, activation: np.ndarray) -> None:
        """Muestra un mapa de activación 2D ya normalizado a [0, 1]."""
        pixmap = numpy_to_qpixmap(activation)
        self.image_label.setPixmap(
            pixmap.scaled(80, 80, KEEP_ASPECT_RATIO, SMOOTH_TRANSFORMATION)
        )

    def set_activation_value(self, value: float) -> None:
        """Muestra el valor escalar de una neurona densa."""
        self.value = value
        self.value_label.setText(f"{value:.4f}")
        self.bar.set_value(value)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - API de Qt
        self.clicked.emit(self.index)
        super().mousePressEvent(event)


class _ValueBar(QWidget):
    """Barra horizontal que representa el valor de activación de una neurona."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(80, 30)
        self.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.value = 0.0

    def set_value(self, value: float) -> None:
        """Actualiza el valor y solicita un repintado."""
        self.value = value
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - API de Qt
        painter = QPainter(self)
        try:
            normalized = max(0.0, min(1.0, (self.value + 1.0) / 2.0))
            color = QColor(255, 100, 100) if self.value < 0 else QColor(100, 255, 100)
            painter.fillRect(0, 0, int(normalized * self.width()), self.height(), color)
        finally:
            painter.end()
