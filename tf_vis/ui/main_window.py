"""
Ventana principal de la aplicación de visualización.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QComboBox, 
                            QSlider, QCheckBox, QTabWidget, QSplitter, 
                            QScrollArea, QGridLayout, QGroupBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize

from ..model_wrapper import ModelWrapper
from ..input_fetcher import InputFetcher
from ..visualization import (display_activation_grid, visualize_layer_filters,
                           apply_gradient_ascent, create_class_activation_map,
                           overlay_heatmap)
from .layer_view import LayerView
from .controls import ControlPanel


class MainWindow(QMainWindow):
    """
    Ventana principal de la aplicación de visualización de características.
    """
    
    def __init__(self, model_wrapper: ModelWrapper, input_fetcher: InputFetcher):
        """
        Inicializa la ventana principal.
        
        Args:
            model_wrapper: Wrapper del modelo TensorFlow
            input_fetcher: Objeto para obtener imágenes de entrada
        """
        super().__init__()
        
        self.model_wrapper = model_wrapper
        self.input_fetcher = input_fetcher
        
        # Estado de la aplicación
        self.current_image = None
        self.current_layer = None
        self.current_filter = 0
        self.current_vis_mode = 'activations'  # 'activations', 'gradients', 'deconv', 'optimization'
        
        # Configurar la interfaz
        self.init_ui()
        
        # Iniciar temporizador para actualización de la interfaz
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(100)  # Actualizar cada 100ms
    
    def init_ui(self):
        """Inicializa la interfaz de usuario."""
        # Configurar ventana principal
        self.setWindowTitle('TensorFlow Feature Visualization Toolbox')
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        
        # Splitter horizontal para dividir la interfaz
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panel izquierdo: entrada y controles
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Área de imagen de entrada
        self.input_display = QLabel()
        self.input_display.setMinimumSize(300, 300)
        self.input_display.setAlignment(Qt.AlignCenter)
        self.input_display.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        left_layout.addWidget(self.input_display)
        
        # Panel de controles
        self.control_panel = ControlPanel(self.model_wrapper)
        self.control_panel.layer_selected.connect(self.on_layer_selected)
        self.control_panel.filter_selected.connect(self.on_filter_selected)
        self.control_panel.vis_mode_changed.connect(self.on_vis_mode_changed)
        left_layout.addWidget(self.control_panel)
        
        # Panel derecho: visualizaciones
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Pestañas para diferentes visualizaciones
        vis_tabs = QTabWidget()
        
        # Pestaña de activaciones
        self.layer_view = LayerView(self.model_wrapper)
        vis_tabs.addTab(self.layer_view, "Activaciones")
        
        # Pestaña de filtros
        filter_tab = QWidget()
        filter_layout = QVBoxLayout(filter_tab)
        self.filter_display = QLabel()
        self.filter_display.setAlignment(Qt.AlignCenter)
        filter_layout.addWidget(self.filter_display)
        vis_tabs.addTab(filter_tab, "Filtros")
        
        # Pestaña de optimización
        optim_tab = QWidget()
        optim_layout = QVBoxLayout(optim_tab)
        self.optim_display = QLabel()
        self.optim_display.setAlignment(Qt.AlignCenter)
        optim_layout.addWidget(self.optim_display)
        vis_tabs.addTab(optim_tab, "Optimización")
        
        # Pestaña de CAM (Class Activation Mapping)
        cam_tab = QWidget()
        cam_layout = QVBoxLayout(cam_tab)
        self.cam_display = QLabel()
        self.cam_display.setAlignment(Qt.AlignCenter)
        cam_layout.addWidget(self.cam_display)
        vis_tabs.addTab(cam_tab, "CAM")
        
        right_layout.addWidget(vis_tabs)
        
        # Añadir paneles al splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        # Barra de estado
        self.statusBar().showMessage('Listo')
        
        # Mostrar ventana
        self.show()
    
    def update_display(self):
        """Actualiza la visualización con nuevos datos."""
        try:
            # Obtener nueva imagen de entrada
            if self.input_fetcher.input_source == 'webcam':
                self.current_image = self.input_fetcher.get_next_image()
            elif self.current_image is None:
                self.current_image = self.input_fetcher.get_next_image()
            
            # Mostrar imagen de entrada
            self._display_image(self.current_image, self.input_display)
            
            # Procesar imagen con el modelo si hay una capa seleccionada
            if self.current_layer:
                # Obtener activaciones
                activations = self.model_wrapper.forward_pass(
                    self.current_image, self.current_layer)[self.current_layer]
                
                # Actualizar visualización según el modo
                if self.current_vis_mode == 'activations':
                    self.layer_view.update_activations(activations, self.current_layer)
                
                elif self.current_vis_mode == 'gradients':
                    # Calcular gradientes
                    grads = self.model_wrapper.compute_gradients(
                        self.current_image, self.current_layer, self.current_filter)
                    
                    # Normalizar para visualización
                    grads = np.abs(grads[0])
                    grads = grads / (np.max(grads) + 1e-8)
                    
                    # Mostrar gradientes
                    self._display_image(grads, self.filter_display)
                
                elif self.current_vis_mode == 'deconv':
                    # Calcular deconvolución
                    deconv = self.model_wrapper.deconv(
                        self.current_image, self.current_layer, self.current_filter)
                    
                    # Mostrar deconvolución
                    self._display_image(deconv, self.filter_display)
                
                elif self.current_vis_mode == 'optimization':
                    # Generar imagen optimizada si no existe
                    if not hasattr(self, 'optimized_image') or self.optimized_image is None:
                        self.statusBar().showMessage('Generando visualización optimizada...')
                        self.optimized_image = apply_gradient_ascent(
                            self.model_wrapper, self.current_layer, self.current_filter)
                        self.statusBar().showMessage('Visualización optimizada generada')
                    
                    # Mostrar imagen optimizada
                    self._display_image(self.optimized_image, self.optim_display)
                
                # Actualizar CAM si es la última capa convolucional
                layer_info = self.model_wrapper.get_layer_info(self.current_layer)
                if 'Conv' in layer_info['type'] and self.current_filter < 10:
                    # Usar el índice del filtro como clase para CAM (simplificación)
                    cam = create_class_activation_map(
                        self.model_wrapper, self.current_image, self.current_layer, self.current_filter)
                    
                    # Superponer CAM en la imagen original
                    overlay = overlay_heatmap(self.current_image.copy(), cam)
                    
                    # Mostrar CAM
                    self._display_image(overlay, self.cam_display)
        
        except Exception as e:
            self.statusBar().showMessage(f'Error: {str(e)}')
            print(f"Error en update_display: {e}")
    
    def _display_image(self, img: np.ndarray, display_widget: QLabel):
        """
        Muestra una imagen en un widget QLabel.
        
        Args:
            img: Imagen como array numpy
            display_widget: Widget QLabel donde mostrar la imagen
        """
        if img is None:
            return
        
        # Normalizar imagen si es necesario
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # Convertir a formato QImage
        height, width = img.shape[:2]
        bytes_per_line = 3 * width
        
        if len(img.shape) == 2:  # Imagen en escala de grises
            q_img = QImage(img.data, width, height, width, QImage.Format_Grayscale8)
        else:  # Imagen RGB
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Mostrar en el widget
        pixmap = QPixmap.fromImage(q_img)
        display_widget.setPixmap(pixmap.scaled(
            display_widget.width(), display_widget.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def on_layer_selected(self, layer_name: str):
        """
        Maneja la selección de una capa.
        
        Args:
            layer_name: Nombre de la capa seleccionada
        """
        self.current_layer = layer_name
        self.optimized_image = None  # Resetear imagen optimizada
        self.statusBar().showMessage(f'Capa seleccionada: {layer_name}')
    
    def on_filter_selected(self, filter_idx: int):
        """
        Maneja la selección de un filtro.
        
        Args:
            filter_idx: Índice del filtro seleccionado
        """
        self.current_filter = filter_idx
        self.optimized_image = None  # Resetear imagen optimizada
        self.statusBar().showMessage(f'Filtro seleccionado: {filter_idx}')
    
    def on_vis_mode_changed(self, mode: str):
        """
        Maneja el cambio de modo de visualización.
        
        Args:
            mode: Nuevo modo de visualización
        """
        self.current_vis_mode = mode
        self.optimized_image = None  # Resetear imagen optimizada
        self.statusBar().showMessage(f'Modo de visualización: {mode}')
    
    def keyPressEvent(self, event):
        """Maneja eventos de teclado."""
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_Right:
            # Siguiente imagen
            if self.input_fetcher.input_source != 'webcam':
                self.current_image = self.input_fetcher.get_next_image()
        elif event.key() == Qt.Key_Left:
            # Imagen anterior
            if self.input_fetcher.input_source != 'webcam':
                self.current_image = self.input_fetcher.get_previous_image()
        elif event.key() == Qt.Key_Up:
            # Filtro anterior
            self.control_panel.select_previous_filter()
        elif event.key() == Qt.Key_Down:
            # Siguiente filtro
            self.control_panel.select_next_filter()
        elif event.key() == Qt.Key_H:
            # Mostrar ayuda
            self.show_help()
    
    def show_help(self):
        """Muestra información de ayuda."""
        help_text = """
        Atajos de teclado:
        - Esc: Cerrar aplicación
        - Derecha: Siguiente imagen
        - Izquierda: Imagen anterior
        - Arriba: Filtro anterior
        - Abajo: Siguiente filtro
        - H: Mostrar esta ayuda
        """
        self.statusBar().showMessage(help_text, 5000)  # Mostrar por 5 segundos
    
    def closeEvent(self, event):
        """Maneja el cierre de la ventana."""
        # Liberar recursos
        if self.input_fetcher:
            self.input_fetcher.close()
        event.accept()
