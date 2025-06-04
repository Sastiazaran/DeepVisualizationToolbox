"""
Panel de controles para la aplicación de visualización.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QComboBox, QSlider, QPushButton, QGroupBox,
                            QRadioButton, QButtonGroup, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal

from ..model_wrapper import ModelWrapper


class ControlPanel(QWidget):
    """
    Panel de controles para la aplicación de visualización.
    """
    
    # Señales
    layer_selected = pyqtSignal(str)  # Emitida cuando se selecciona una capa
    filter_selected = pyqtSignal(int)  # Emitida cuando se selecciona un filtro
    vis_mode_changed = pyqtSignal(str)  # Emitida cuando cambia el modo de visualización
    
    def __init__(self, model_wrapper: ModelWrapper):
        """
        Inicializa el panel de controles.
        
        Args:
            model_wrapper: Wrapper del modelo TensorFlow
        """
        super().__init__()
        
        self.model_wrapper = model_wrapper
        self.layer_names = model_wrapper.layer_names
        self.current_layer_idx = 0
        self.current_filter_idx = 0
        
        # Configurar interfaz
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario."""
        # Layout principal
        layout = QVBoxLayout(self)
        
        # Grupo de selección de capa
        layer_group = QGroupBox("Selección de Capa")
        layer_layout = QVBoxLayout(layer_group)
        
        # Combo para seleccionar capa
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(self.layer_names)
        self.layer_combo.currentIndexChanged.connect(self.on_layer_changed)
        layer_layout.addWidget(self.layer_combo)
        
        # Botones para navegar entre capas
        layer_nav_layout = QHBoxLayout()
        prev_layer_btn = QPushButton("< Anterior")
        prev_layer_btn.clicked.connect(self.select_previous_layer)
        next_layer_btn = QPushButton("Siguiente >")
        next_layer_btn.clicked.connect(self.select_next_layer)
        layer_nav_layout.addWidget(prev_layer_btn)
        layer_nav_layout.addWidget(next_layer_btn)
        layer_layout.addLayout(layer_nav_layout)
        
        layout.addWidget(layer_group)
        
        # Grupo de selección de filtro
        filter_group = QGroupBox("Selección de Filtro/Neurona")
        filter_layout = QVBoxLayout(filter_group)
        
        # Spinner para seleccionar filtro
        filter_layout_h = QHBoxLayout()
        filter_layout_h.addWidget(QLabel("Índice:"))
        self.filter_spin = QSpinBox()
        self.filter_spin.setMinimum(0)
        self.filter_spin.setMaximum(999)  # Se ajustará según la capa
        self.filter_spin.valueChanged.connect(self.on_filter_changed)
        filter_layout_h.addWidget(self.filter_spin)
        filter_layout.addLayout(filter_layout_h)
        
        # Slider para seleccionar filtro
        self.filter_slider = QSlider(Qt.Horizontal)
        self.filter_slider.setMinimum(0)
        self.filter_slider.setMaximum(999)  # Se ajustará según la capa
        self.filter_slider.valueChanged.connect(self.on_slider_changed)
        filter_layout.addWidget(self.filter_slider)
        
        # Botones para navegar entre filtros
        filter_nav_layout = QHBoxLayout()
        prev_filter_btn = QPushButton("< Anterior")
        prev_filter_btn.clicked.connect(self.select_previous_filter)
        next_filter_btn = QPushButton("Siguiente >")
        next_filter_btn.clicked.connect(self.select_next_filter)
        filter_nav_layout.addWidget(prev_filter_btn)
        filter_nav_layout.addWidget(next_filter_btn)
        filter_layout.addLayout(filter_nav_layout)
        
        layout.addWidget(filter_group)
        
        # Grupo de modo de visualización
        vis_mode_group = QGroupBox("Modo de Visualización")
        vis_mode_layout = QVBoxLayout(vis_mode_group)
        
        # Radio buttons para modos de visualización
        self.vis_mode_group = QButtonGroup(self)
        
        self.radio_activations = QRadioButton("Activaciones")
        self.radio_activations.setChecked(True)
        self.vis_mode_group.addButton(self.radio_activations)
        vis_mode_layout.addWidget(self.radio_activations)
        
        self.radio_gradients = QRadioButton("Gradientes")
        self.vis_mode_group.addButton(self.radio_gradients)
        vis_mode_layout.addWidget(self.radio_gradients)
        
        self.radio_deconv = QRadioButton("Deconvolución")
        self.vis_mode_group.addButton(self.radio_deconv)
        vis_mode_layout.addWidget(self.radio_deconv)
        
        self.radio_optimization = QRadioButton("Optimización")
        self.vis_mode_group.addButton(self.radio_optimization)
        vis_mode_layout.addWidget(self.radio_optimization)
        
        # Conectar cambios de modo
        self.radio_activations.toggled.connect(
            lambda: self.on_vis_mode_changed('activations'))
        self.radio_gradients.toggled.connect(
            lambda: self.on_vis_mode_changed('gradients'))
        self.radio_deconv.toggled.connect(
            lambda: self.on_vis_mode_changed('deconv'))
        self.radio_optimization.toggled.connect(
            lambda: self.on_vis_mode_changed('optimization'))
        
        layout.addWidget(vis_mode_group)
        
        # Botón de ayuda
        help_btn = QPushButton("Ayuda")
        help_btn.clicked.connect(self.show_help)
        layout.addWidget(help_btn)
        
        # Inicializar con la primera capa
        self.on_layer_changed(0)
    
    def on_layer_changed(self, index: int):
        """
        Maneja el cambio de capa seleccionada.
        
        Args:
            index: Índice de la capa en el combo
        """
        if 0 <= index < len(self.layer_names):
            self.current_layer_idx = index
            layer_name = self.layer_names[index]
            
            # Obtener información de la capa
            layer_info = self.model_wrapper.get_layer_info(layer_name)
            
            # Ajustar máximo de filtros según la capa
            max_filters = 0
            shape = layer_info['shape']
            
            # Verificar que shape no sea None y tenga elementos
            if shape is not None:
                if isinstance(shape, tuple) and len(shape) == 4:  # Capa convolucional
                    if shape[-1] is not None:
                        max_filters = shape[-1] - 1
                    else:
                        max_filters = 0
                elif isinstance(shape, tuple) and len(shape) == 2:  # Capa densa
                    if shape[-1] is not None:
                        max_filters = shape[-1] - 1
                    else:
                        max_filters = 0
            
            # Actualizar controles de filtro
            self.filter_slider.setMaximum(max(0, max_filters))
            self.filter_spin.setMaximum(max(0, max_filters))
            
            # Resetear índice de filtro
            self.filter_slider.setValue(0)
            self.filter_spin.setValue(0)
            self.current_filter_idx = 0
            
            # Emitir señal
            self.layer_selected.emit(layer_name)
    
    def on_filter_changed(self, value: int):
        """
        Maneja el cambio de filtro seleccionado mediante el spinner.
        
        Args:
            value: Índice del filtro
        """
        self.current_filter_idx = value
        self.filter_slider.setValue(value)
        self.filter_selected.emit(value)
    
    def on_slider_changed(self, value: int):
        """
        Maneja el cambio de filtro seleccionado mediante el slider.
        
        Args:
            value: Índice del filtro
        """
        self.current_filter_idx = value
        self.filter_spin.setValue(value)
        self.filter_selected.emit(value)
    
    def on_vis_mode_changed(self, mode: str):
        """
        Maneja el cambio de modo de visualización.
        
        Args:
            mode: Modo de visualización
        """
        # Solo emitir si el radio button está marcado
        sender = self.sender()
        if isinstance(sender, QRadioButton) and sender.isChecked():
            self.vis_mode_changed.emit(mode)
    
    def select_previous_layer(self):
        """Selecciona la capa anterior."""
        new_idx = max(0, self.current_layer_idx - 1)
        self.layer_combo.setCurrentIndex(new_idx)
    
    def select_next_layer(self):
        """Selecciona la siguiente capa."""
        new_idx = min(len(self.layer_names) - 1, self.current_layer_idx + 1)
        self.layer_combo.setCurrentIndex(new_idx)
    
    def select_previous_filter(self):
        """Selecciona el filtro anterior."""
        new_idx = max(0, self.current_filter_idx - 1)
        self.filter_spin.setValue(new_idx)
    
    def select_next_filter(self):
        """Selecciona el siguiente filtro."""
        new_idx = min(self.filter_slider.maximum(), self.current_filter_idx + 1)
        self.filter_spin.setValue(new_idx)
    
    def show_help(self):
        """Muestra información de ayuda."""
        # En una implementación real, esto podría mostrar un diálogo de ayuda
        print("Ayuda de la aplicación:")
        print("- Selecciona una capa para visualizar sus activaciones")
        print("- Selecciona un filtro/neurona específico para ver detalles")
        print("- Cambia el modo de visualización para diferentes perspectivas")
        print("- Usa las teclas de flecha para navegar entre imágenes y filtros")
