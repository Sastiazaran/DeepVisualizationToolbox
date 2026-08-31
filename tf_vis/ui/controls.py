"""
Panel de controles para la aplicación de visualización.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..model_wrapper import ModelWrapper
from .qt_compat import HORIZONTAL

# Modos de visualización: identificador interno -> etiqueta mostrada.
VIS_MODES = {
    'activations': 'Activaciones',
    'gradients': 'Gradientes',
    'deconv': 'Deconvolución (guided backprop)',
    'optimization': 'Optimización',
    'gradcam': 'Grad-CAM',
}


class ControlPanel(QWidget):
    """
    Panel de controles para seleccionar capa, filtro y modo de visualización.
    """

    layer_selected = pyqtSignal(str)
    filter_selected = pyqtSignal(int)
    vis_mode_changed = pyqtSignal(str)

    def __init__(self, model_wrapper: ModelWrapper):
        """
        Inicializa el panel de controles.

        Args:
            model_wrapper: Wrapper del modelo
        """
        super().__init__()

        self.model_wrapper = model_wrapper
        # Se listan solo las capas visualizables para que el desplegable no se
        # llene de capas cuya activación no se puede dibujar.
        self.layer_names = model_wrapper.visualizable_layers() or model_wrapper.layer_names
        self.current_layer_idx = 0
        self.current_filter_idx = 0

        self.init_ui()

    def init_ui(self) -> None:
        """Construye la interfaz del panel."""
        layout = QVBoxLayout(self)

        layout.addWidget(self._build_layer_group())
        layout.addWidget(self._build_filter_group())
        layout.addWidget(self._build_vis_mode_group())

        help_btn = QPushButton("Ayuda")
        help_btn.clicked.connect(self.show_help)
        layout.addWidget(help_btn)

        self.on_layer_changed(0)

    def _build_layer_group(self) -> QGroupBox:
        group = QGroupBox("Selección de Capa")
        group_layout = QVBoxLayout(group)

        self.layer_combo = QComboBox()
        self.layer_combo.addItems(self.layer_names)
        self.layer_combo.currentIndexChanged.connect(self.on_layer_changed)
        group_layout.addWidget(self.layer_combo)

        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("< Anterior")
        prev_btn.clicked.connect(self.select_previous_layer)
        next_btn = QPushButton("Siguiente >")
        next_btn.clicked.connect(self.select_next_layer)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        group_layout.addLayout(nav_layout)

        return group

    def _build_filter_group(self) -> QGroupBox:
        group = QGroupBox("Selección de Filtro/Neurona")
        group_layout = QVBoxLayout(group)

        spin_layout = QHBoxLayout()
        spin_layout.addWidget(QLabel("Índice:"))
        self.filter_spin = QSpinBox()
        self.filter_spin.setMinimum(0)
        self.filter_spin.valueChanged.connect(self.on_filter_changed)
        spin_layout.addWidget(self.filter_spin)
        self.filter_count_label = QLabel("")
        spin_layout.addWidget(self.filter_count_label)
        group_layout.addLayout(spin_layout)

        self.filter_slider = QSlider(HORIZONTAL)
        self.filter_slider.setMinimum(0)
        self.filter_slider.valueChanged.connect(self.on_slider_changed)
        group_layout.addWidget(self.filter_slider)

        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("< Anterior")
        prev_btn.clicked.connect(self.select_previous_filter)
        next_btn = QPushButton("Siguiente >")
        next_btn.clicked.connect(self.select_next_filter)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        group_layout.addLayout(nav_layout)

        return group

    def _build_vis_mode_group(self) -> QGroupBox:
        group = QGroupBox("Modo de Visualización")
        group_layout = QVBoxLayout(group)

        self.vis_mode_group = QButtonGroup(self)
        self.vis_mode_buttons: dict[str, QRadioButton] = {}

        for index, (mode, label) in enumerate(VIS_MODES.items()):
            button = QRadioButton(label)
            button.setChecked(index == 0)
            self.vis_mode_group.addButton(button)
            # `toggled` se emite para el botón que se marca y para el que se
            # desmarca, así que solo se propaga el que queda activo.
            button.toggled.connect(
                lambda checked, m=mode: checked and self.vis_mode_changed.emit(m)
            )
            group_layout.addWidget(button)
            self.vis_mode_buttons[mode] = button

        return group

    # ------------------------------------------------------------------
    # Manejadores
    # ------------------------------------------------------------------
    def on_layer_changed(self, index: int) -> None:
        """
        Ajusta los controles de filtro a la capa seleccionada.

        Args:
            index: Índice de la capa en el desplegable
        """
        if not 0 <= index < len(self.layer_names):
            return

        self.current_layer_idx = index
        layer_name = self.layer_names[index]

        n_filters = self.model_wrapper.num_filters(layer_name)
        max_index = max(0, n_filters - 1)

        self.filter_slider.setMaximum(max_index)
        self.filter_spin.setMaximum(max_index)
        self.filter_count_label.setText(f"de {n_filters}" if n_filters else "(sin filtros)")

        self.filter_slider.setValue(0)
        self.filter_spin.setValue(0)
        self.current_filter_idx = 0

        self.layer_selected.emit(layer_name)

    def on_filter_changed(self, value: int) -> None:
        """Sincroniza el slider cuando cambia el spinner."""
        if value == self.current_filter_idx and self.filter_slider.value() == value:
            return
        self.current_filter_idx = value
        self.filter_slider.setValue(value)
        self.filter_selected.emit(value)

    def on_slider_changed(self, value: int) -> None:
        """Sincroniza el spinner cuando cambia el slider."""
        if value == self.current_filter_idx and self.filter_spin.value() == value:
            return
        self.current_filter_idx = value
        self.filter_spin.setValue(value)
        self.filter_selected.emit(value)

    def select_previous_layer(self) -> None:
        """Selecciona la capa anterior."""
        self.layer_combo.setCurrentIndex(max(0, self.current_layer_idx - 1))

    def select_next_layer(self) -> None:
        """Selecciona la capa siguiente."""
        self.layer_combo.setCurrentIndex(
            min(len(self.layer_names) - 1, self.current_layer_idx + 1)
        )

    def select_previous_filter(self) -> None:
        """Selecciona el filtro anterior."""
        self.filter_spin.setValue(max(0, self.current_filter_idx - 1))

    def select_next_filter(self) -> None:
        """Selecciona el filtro siguiente."""
        self.filter_spin.setValue(
            min(self.filter_slider.maximum(), self.current_filter_idx + 1)
        )

    def set_vis_mode(self, mode: str) -> None:
        """Marca programáticamente un modo de visualización."""
        button = self.vis_mode_buttons.get(mode)
        if button is not None:
            button.setChecked(True)

    def show_help(self) -> None:
        """Muestra información de ayuda."""
        print("Ayuda de la aplicación:")
        print("- Selecciona una capa para visualizar sus activaciones")
        print("- Selecciona un filtro/neurona específico para ver detalles")
        print("- Cambia el modo de visualización para diferentes perspectivas")
        print("- Usa las teclas de flecha para navegar entre imágenes y filtros")
