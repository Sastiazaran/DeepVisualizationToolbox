"""
Widget para visualizar activaciones de capas.
"""

import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QScrollArea, QGridLayout, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QSize

from ..model_wrapper import ModelWrapper


class LayerView(QWidget):
    """
    Widget para visualizar activaciones de capas de una red neuronal.
    """
    
    filter_clicked = pyqtSignal(int)  # Señal emitida cuando se hace clic en un filtro
    
    def __init__(self, model_wrapper: ModelWrapper):
        """
        Inicializa el widget de visualización de capas.
        
        Args:
            model_wrapper: Wrapper del modelo TensorFlow
        """
        super().__init__()
        
        self.model_wrapper = model_wrapper
        self.current_layer = None
        self.current_activations = None
        
        # Configurar interfaz
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario."""
        # Layout principal
        layout = QVBoxLayout(self)
        
        # Etiqueta para información de la capa
        self.layer_info_label = QLabel("Selecciona una capa para visualizar")
        self.layer_info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.layer_info_label)
        
        # Área de desplazamiento para la cuadrícula de activaciones
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Widget contenedor para la cuadrícula
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        scroll_area.setWidget(self.grid_container)
        
        # Configurar el widget
        self.setMinimumSize(400, 300)
    
    def update_activations(self, activations: np.ndarray, layer_name: str):
        """
        Actualiza la visualización con nuevas activaciones.
        
        Args:
            activations: Array de activaciones
            layer_name: Nombre de la capa
        """
        self.current_layer = layer_name
        self.current_activations = activations
        
        # Actualizar etiqueta de información
        layer_info = self.model_wrapper.get_layer_info(layer_name)
        info_text = f"Capa: {layer_name} | Tipo: {layer_info['type']} | Forma: {layer_info['shape']}"
        self.layer_info_label.setText(info_text)
        
        # Limpiar cuadrícula existente
        self._clear_grid()
        
        # Crear nueva cuadrícula de activaciones
        self._create_activation_grid(activations)
    
    def _clear_grid(self):
        """Limpia la cuadrícula de activaciones."""
        # Eliminar todos los widgets de la cuadrícula
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def _create_activation_grid(self, activations: np.ndarray):
        """
        Crea una cuadrícula de visualizaciones de activaciones.
        
        Args:
            activations: Array de activaciones
        """
        # Determinar la forma de las activaciones
        if len(activations.shape) == 4:  # [batch, height, width, channels]
            # Para capas convolucionales
            _, height, width, n_filters = activations.shape
            
            # Crear una cuadrícula de visualizaciones
            max_cols = 8  # Número máximo de columnas
            
            for i in range(min(64, n_filters)):  # Limitar a 64 filtros para rendimiento
                # Obtener activación para este filtro
                activation = activations[0, :, :, i]
                
                # Normalizar para visualización
                activation = (activation - np.min(activation)) / (np.max(activation) - np.min(activation) + 1e-8)
                
                # Crear widget de visualización
                vis_widget = self._create_activation_widget(activation, i)
                
                # Añadir a la cuadrícula
                row, col = i // max_cols, i % max_cols
                self.grid_layout.addWidget(vis_widget, row, col)
                
        elif len(activations.shape) == 2:  # [batch, features]
            # Para capas densas
            _, n_features = activations.shape
            
            # Crear una visualización de barras para cada neurona
            max_cols = 8
            
            for i in range(min(64, n_features)):
                # Obtener activación para esta neurona
                activation_value = activations[0, i]
                
                # Crear widget de visualización
                vis_widget = self._create_dense_activation_widget(activation_value, i)
                
                # Añadir a la cuadrícula
                row, col = i // max_cols, i % max_cols
                self.grid_layout.addWidget(vis_widget, row, col)
    
    def _create_activation_widget(self, activation: np.ndarray, filter_idx: int) -> QWidget:
        """
        Crea un widget para visualizar una activación convolucional.
        
        Args:
            activation: Array 2D de activación
            filter_idx: Índice del filtro
            
        Returns:
            Widget de visualización
        """
        # Crear widget
        widget = QWidget()
        widget.setFixedSize(100, 120)
        widget.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px;")
        
        # Layout
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Etiqueta para el índice del filtro
        filter_label = QLabel(f"Filtro {filter_idx}")
        filter_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(filter_label)
        
        # Visualización de activación
        act_label = QLabel()
        act_label.setAlignment(Qt.AlignCenter)
        
        # Convertir activación a imagen
        act_img = (activation * 255).astype(np.uint8)
        height, width = act_img.shape
        q_img = QImage(act_img.data, width, height, width, QImage.Format_Grayscale8)
        
        # Mostrar imagen
        pixmap = QPixmap.fromImage(q_img)
        act_label.setPixmap(pixmap.scaled(
            80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        layout.addWidget(act_label)
        
        # Hacer que el widget sea clickeable
        widget.mousePressEvent = lambda event, idx=filter_idx: self.filter_clicked.emit(idx)
        
        return widget
    
    def _create_dense_activation_widget(self, activation_value: float, neuron_idx: int) -> QWidget:
        """
        Crea un widget para visualizar una activación de capa densa.
        
        Args:
            activation_value: Valor de activación
            neuron_idx: Índice de la neurona
            
        Returns:
            Widget de visualización
        """
        # Crear widget
        widget = QWidget()
        widget.setFixedSize(100, 120)
        widget.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px;")
        
        # Layout
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Etiqueta para el índice de la neurona
        neuron_label = QLabel(f"Neurona {neuron_idx}")
        neuron_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(neuron_label)
        
        # Etiqueta para el valor de activación
        value_label = QLabel(f"{activation_value:.4f}")
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)
        
        # Barra de visualización
        bar_widget = QWidget()
        bar_widget.setFixedSize(80, 30)
        bar_widget.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        
        # Personalizar la pintura de la barra
        def paintEvent(event):
            painter = QPainter(bar_widget)
            
            # Normalizar valor entre 0 y 1
            norm_value = max(0, min(1, (activation_value + 1) / 2))  # Asumiendo rango [-1, 1]
            
            # Dibujar barra
            bar_width = int(norm_value * bar_widget.width())
            
            # Color basado en el valor (rojo para negativo, verde para positivo)
            if activation_value < 0:
                color = QColor(255, 100, 100)  # Rojo claro
            else:
                color = QColor(100, 255, 100)  # Verde claro
            
            painter.fillRect(0, 0, bar_width, bar_widget.height(), color)
            painter.end()
        
        # Asignar método de pintura personalizado
        bar_widget.paintEvent = paintEvent
        
        layout.addWidget(bar_widget)
        
        # Hacer que el widget sea clickeable
        widget.mousePressEvent = lambda event, idx=neuron_idx: self.filter_clicked.emit(idx)
        
        return widget
