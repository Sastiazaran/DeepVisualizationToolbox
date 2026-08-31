"""
Ventana principal de la aplicación de visualización.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..input_fetcher import InputFetcher
from ..model_wrapper import ModelWrapper
from ..utils.misc import predict_image, save_visualizations
from ..visualization import (
    apply_gradient_ascent,
    create_class_activation_map,
    overlay_heatmap,
)
from .controls import ControlPanel
from .layer_view import LayerView
from .qt_compat import ALIGN_CENTER, KEEP_ASPECT_RATIO, SMOOTH_TRANSFORMATION, numpy_to_qpixmap

LIVE_REFRESH_MS = 100
STATIC_REFRESH_MS = 500


class MainWindow(QMainWindow):
    """
    Ventana principal de la aplicación de visualización de características.
    """

    def __init__(self, model_wrapper: ModelWrapper, input_fetcher: InputFetcher,
                 model_name: str | None = None, output_dir: str = 'visualizations'):
        """
        Inicializa la ventana principal.

        Args:
            model_wrapper: Wrapper del modelo
            input_fetcher: Fuente de imágenes de entrada
            model_name: Nombre del modelo, usado para decodificar predicciones
            output_dir: Directorio donde se guardan las capturas
        """
        super().__init__()

        self.model_wrapper = model_wrapper
        self.input_fetcher = input_fetcher
        self.model_name = model_name
        self.output_dir = output_dir

        self.current_image: np.ndarray | None = None
        self.current_layer: str | None = None
        self.current_filter = 0
        self.current_vis_mode = 'activations'
        self.optimized_image: np.ndarray | None = None
        # Las visualizaciones caras solo se recalculan cuando cambia la entrada
        # o la selección, no en cada tic del temporizador.
        self._needs_refresh = True
        self._last_error: str | None = None

        self.init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(LIVE_REFRESH_MS if input_fetcher.is_live else STATIC_REFRESH_MS)

    def init_ui(self) -> None:
        """Construye la interfaz de usuario."""
        self.setWindowTitle('TensorFlow Feature Visualization Toolbox')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([400, 800])

        self.statusBar().showMessage('Listo — pulsa H para ver los atajos de teclado')
        self.show()

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.input_display = QLabel()
        self.input_display.setMinimumSize(300, 300)
        self.input_display.setAlignment(ALIGN_CENTER)
        self.input_display.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(self.input_display)

        self.prediction_label = QLabel("Predicciones: —")
        self.prediction_label.setWordWrap(True)
        layout.addWidget(self.prediction_label)

        self.control_panel = ControlPanel(self.model_wrapper)
        self.control_panel.layer_selected.connect(self.on_layer_selected)
        self.control_panel.filter_selected.connect(self.on_filter_selected)
        self.control_panel.vis_mode_changed.connect(self.on_vis_mode_changed)
        layout.addWidget(self.control_panel)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.vis_tabs = QTabWidget()

        self.layer_view = LayerView(self.model_wrapper)
        # Al hacer clic en un filtro se actualiza la selección del panel de control.
        self.layer_view.filter_clicked.connect(self.control_panel.filter_spin.setValue)
        self.vis_tabs.addTab(self.layer_view, "Activaciones")

        self.filter_display = self._add_image_tab("Filtros")
        self.optim_display = self._add_image_tab("Optimización")
        self.cam_display = self._add_image_tab("Grad-CAM")

        self.vis_tabs.currentChanged.connect(self._request_refresh)
        layout.addWidget(self.vis_tabs)

        return panel

    def _add_image_tab(self, title: str) -> QLabel:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        label = QLabel()
        label.setAlignment(ALIGN_CENTER)
        tab_layout.addWidget(label)
        self.vis_tabs.addTab(tab, title)
        return label

    # ------------------------------------------------------------------
    # Bucle de actualización
    # ------------------------------------------------------------------
    def update_display(self) -> None:
        """Refresca la entrada y la visualización activa."""
        try:
            if self.input_fetcher.is_live:
                self.current_image = self.input_fetcher.get_next_image()
                self._needs_refresh = True
            elif self.current_image is None:
                self.current_image = self.input_fetcher.get_current_image()
                self._needs_refresh = True

            if not self._needs_refresh:
                return
            self._needs_refresh = False

            self._display_input()

            if self.current_layer:
                self._update_visualization()

            if self._last_error is not None:
                self._last_error = None
                self.statusBar().showMessage('Listo')
        except Exception as error:  # la interfaz no debe caerse por un fallo de una capa
            message = f'Error: {error}'
            if message != self._last_error:
                self._last_error = message
                self.statusBar().showMessage(message)
                print(f"Error en update_display: {error}")

    def _display_input(self) -> None:
        """Muestra la imagen de entrada sin preprocesar y sus predicciones."""
        raw = self.input_fetcher.current_raw_image
        if raw is not None:
            self._display_image(raw, self.input_display)

    def _update_visualization(self) -> None:
        """Calcula y muestra la visualización correspondiente al modo activo."""
        mode = self.current_vis_mode
        layer = self.current_layer

        if mode == 'activations':
            activations = self.model_wrapper.forward_pass(self.current_image, layer)[layer]
            self.layer_view.update_activations(activations, layer)
            self.vis_tabs.setCurrentWidget(self.layer_view)

        elif mode == 'gradients':
            saliency = self.model_wrapper.saliency_map(
                self.current_image, layer, self.current_filter)
            self._display_image(saliency, self.filter_display)

        elif mode == 'deconv':
            deconv = self.model_wrapper.deconv(self.current_image, layer, self.current_filter)
            self._display_image(deconv, self.filter_display)

        elif mode == 'optimization':
            if self.optimized_image is None:
                self.statusBar().showMessage('Generando visualización optimizada...')
                height, width = self.current_image.shape[:2]
                self.optimized_image = apply_gradient_ascent(
                    self.model_wrapper, layer, self.current_filter,
                    image_size=(height, width))
                self.statusBar().showMessage('Visualización optimizada generada')
            self._display_image(self.optimized_image, self.optim_display)

        elif mode == 'gradcam':
            self._update_gradcam()

    def _update_gradcam(self) -> None:
        """Calcula Grad-CAM para la clase predicha y lo superpone sobre la entrada."""
        if not self.model_wrapper.is_spatial_layer(self.current_layer):
            self.statusBar().showMessage('Grad-CAM necesita una capa convolucional')
            return

        predictions = self.model_wrapper.model.predict(
            np.expand_dims(self.current_image, 0), verbose=0)
        class_idx = int(np.argmax(predictions[0]))

        cam = create_class_activation_map(
            self.model_wrapper, self.current_image, self.current_layer, class_idx)
        raw = self.input_fetcher.current_raw_image
        self._display_image(overlay_heatmap(raw, cam), self.cam_display)
        self._show_predictions()

    def _show_predictions(self) -> None:
        """Muestra las clases top-3 del modelo para la imagen actual."""
        try:
            top = predict_image(self.model_wrapper.model, self.current_image, top_k=3,
                                model_name=self.model_name)
        except Exception:
            self.prediction_label.setText("Predicciones: no disponibles para este modelo")
            return

        text = ' | '.join(f'{label} {prob:.1%}' for _, label, prob in top)
        self.prediction_label.setText(f"Predicciones: {text}")

    def _display_image(self, img: np.ndarray, display_widget: QLabel) -> None:
        """
        Muestra un array numpy en un `QLabel`, escalado al tamaño del widget.

        Args:
            img: Imagen como array numpy
            display_widget: Widget destino
        """
        if img is None:
            return

        pixmap = numpy_to_qpixmap(img)
        display_widget.setPixmap(pixmap.scaled(
            display_widget.width(), display_widget.height(),
            KEEP_ASPECT_RATIO, SMOOTH_TRANSFORMATION))

    def _request_refresh(self) -> None:
        """Marca la visualización como pendiente de recalcular."""
        self._needs_refresh = True

    # ------------------------------------------------------------------
    # Manejadores de la interfaz
    # ------------------------------------------------------------------
    def on_layer_selected(self, layer_name: str) -> None:
        """Actualiza la capa activa."""
        self.current_layer = layer_name
        self.optimized_image = None
        self._request_refresh()
        self.statusBar().showMessage(f'Capa seleccionada: {layer_name}')

    def on_filter_selected(self, filter_idx: int) -> None:
        """Actualiza el filtro activo."""
        self.current_filter = filter_idx
        self.optimized_image = None
        self._request_refresh()
        self.statusBar().showMessage(f'Filtro seleccionado: {filter_idx}')

    def on_vis_mode_changed(self, mode: str) -> None:
        """Cambia el modo de visualización y trae al frente su pestaña."""
        self.current_vis_mode = mode
        self.optimized_image = None
        self._request_refresh()

        tab_for_mode = {
            'activations': self.layer_view,
            'gradients': self.filter_display,
            'deconv': self.filter_display,
            'optimization': self.optim_display,
            'gradcam': self.cam_display,
        }
        widget = tab_for_mode.get(mode)
        if widget is not None:
            self.vis_tabs.setCurrentIndex(
                self.vis_tabs.indexOf(widget if widget is self.layer_view else widget.parent())
            )

        self.statusBar().showMessage(f'Modo de visualización: {mode}')

    def next_image(self) -> None:
        """Avanza a la siguiente imagen de la fuente."""
        if not self.input_fetcher.is_live:
            self.current_image = self.input_fetcher.get_next_image()
            self._request_refresh()

    def previous_image(self) -> None:
        """Retrocede a la imagen anterior de la fuente."""
        if not self.input_fetcher.is_live:
            self.current_image = self.input_fetcher.get_previous_image()
            self._request_refresh()

    def save_current_view(self) -> None:
        """Guarda la entrada y la visualización activa en `output_dir`."""
        if self.current_image is None:
            return

        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        payload: dict[str, np.ndarray] = {}

        raw = self.input_fetcher.current_raw_image
        if raw is not None:
            payload['input'] = raw
        if self.optimized_image is not None:
            payload['optimization'] = self.optimized_image

        if self.current_layer and self.current_vis_mode in ('gradients', 'deconv'):
            payload[self.current_vis_mode] = self.model_wrapper.deconv(
                self.current_image, self.current_layer, self.current_filter)

        written = save_visualizations(payload, self.output_dir, prefix=stamp)
        self.statusBar().showMessage(
            f'Guardado en {os.path.abspath(self.output_dir)} ({len(written)} archivos)')

    def keyPressEvent(self, event) -> None:  # noqa: N802 - API de Qt
        """Atajos de teclado."""
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close()
        elif key == Qt.Key.Key_Right:
            self.next_image()
        elif key == Qt.Key.Key_Left:
            self.previous_image()
        elif key == Qt.Key.Key_Up:
            self.control_panel.select_previous_filter()
        elif key == Qt.Key.Key_Down:
            self.control_panel.select_next_filter()
        elif key == Qt.Key.Key_S:
            self.save_current_view()
        elif key == Qt.Key.Key_H:
            self.show_help()
        else:
            super().keyPressEvent(event)

    def show_help(self) -> None:
        """Muestra los atajos disponibles en la barra de estado."""
        self.statusBar().showMessage(
            'Esc: salir | ←/→: imagen anterior/siguiente | ↑/↓: filtro | '
            'S: guardar visualización | H: ayuda',
            8000,
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - API de Qt
        """Libera los recursos al cerrar la ventana."""
        self.timer.stop()
        if self.input_fetcher:
            self.input_fetcher.close()
        event.accept()
