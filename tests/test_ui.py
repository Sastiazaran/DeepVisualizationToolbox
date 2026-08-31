"""Tests de la interfaz gráfica, ejecutados con la plataforma Qt `offscreen`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('PyQt6', reason='PyQt6 no está instalado')

from PyQt6.QtWidgets import QApplication  # noqa: E402

from tf_vis.input_fetcher import InputFetcher  # noqa: E402
from tf_vis.ui.controls import VIS_MODES, ControlPanel  # noqa: E402
from tf_vis.ui.layer_view import LayerView  # noqa: E402
from tf_vis.ui.main_window import MainWindow  # noqa: E402
from tf_vis.ui.qt_compat import numpy_to_qpixmap  # noqa: E402


@pytest.fixture(scope='session')
def qt_app():
    """Instancia única de `QApplication` para toda la sesión de tests."""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def fetcher(image_dir):
    with InputFetcher(input_source=f'directory:{image_dir}', target_size=(32, 32)) as source:
        yield source


def test_numpy_to_qpixmap_rgb(qt_app):
    pixmap = numpy_to_qpixmap(np.zeros((8, 6, 3), dtype=np.uint8))
    assert (pixmap.width(), pixmap.height()) == (6, 8)


def test_numpy_to_qpixmap_grayscale_float(qt_app):
    pixmap = numpy_to_qpixmap(np.linspace(0, 1, 64).reshape(8, 8))
    assert pixmap.isNull() is False


def test_numpy_to_qpixmap_handles_non_contiguous_arrays(qt_app):
    # Una vista invertida no es contigua; QImage necesita un búfer contiguo.
    view = np.zeros((8, 8, 3), dtype=np.uint8)[::-1]
    assert view.flags['C_CONTIGUOUS'] is False
    assert numpy_to_qpixmap(view).isNull() is False


def test_control_panel_lists_visualizable_layers(qt_app, wrapper):
    panel = ControlPanel(wrapper)
    assert panel.layer_combo.count() == len(wrapper.visualizable_layers())


def test_control_panel_adjusts_filter_range_per_layer(qt_app, wrapper):
    panel = ControlPanel(wrapper)

    panel.layer_combo.setCurrentIndex(panel.layer_names.index('conv1'))
    assert panel.filter_spin.maximum() == 7

    panel.layer_combo.setCurrentIndex(panel.layer_names.index('conv2'))
    assert panel.filter_spin.maximum() == 15


def test_control_panel_emits_layer_selection(qt_app, wrapper):
    panel = ControlPanel(wrapper)
    seen = []
    panel.layer_selected.connect(seen.append)

    panel.layer_combo.setCurrentIndex(panel.layer_names.index('conv2'))
    assert seen[-1] == 'conv2'


def test_slider_and_spin_stay_in_sync(qt_app, wrapper):
    panel = ControlPanel(wrapper)
    panel.layer_combo.setCurrentIndex(panel.layer_names.index('conv2'))

    panel.filter_slider.setValue(5)
    assert panel.filter_spin.value() == 5

    panel.filter_spin.setValue(9)
    assert panel.filter_slider.value() == 9


def test_vis_mode_emits_once_per_change(qt_app, wrapper):
    panel = ControlPanel(wrapper)
    seen = []
    panel.vis_mode_changed.connect(seen.append)

    panel.set_vis_mode('gradcam')
    # Solo el botón que queda marcado debe propagar su modo.
    assert seen == ['gradcam']
    assert set(VIS_MODES) >= {'activations', 'gradcam'}


def test_layer_view_reuses_tiles_between_frames(qt_app, wrapper, sample_image):
    view = LayerView(wrapper)
    activations = wrapper.forward_pass(sample_image, 'conv1')['conv1']

    view.update_activations(activations, 'conv1')
    tiles = list(view._tiles)
    assert len(tiles) == 8

    view.update_activations(activations, 'conv1')
    assert view._tiles == tiles


def test_layer_view_rebuilds_on_layer_change(qt_app, wrapper, sample_image):
    view = LayerView(wrapper)
    view.update_activations(wrapper.forward_pass(sample_image, 'conv1')['conv1'], 'conv1')
    view.update_activations(wrapper.forward_pass(sample_image, 'conv2')['conv2'], 'conv2')
    assert len(view._tiles) == 16


def test_layer_view_handles_dense_layers(qt_app, wrapper, sample_image):
    view = LayerView(wrapper)
    activations = wrapper.forward_pass(sample_image, 'predictions')['predictions']
    view.update_activations(activations, 'predictions')
    assert len(view._tiles) == 10
    assert all(tile.dense for tile in view._tiles)


def test_main_window_updates_without_errors(qt_app, wrapper, fetcher):
    window = MainWindow(wrapper, fetcher)
    window.timer.stop()

    window.on_layer_selected('conv1')
    window.update_display()

    assert window._last_error is None
    assert window.input_display.pixmap().isNull() is False


def test_main_window_modes_are_all_renderable(qt_app, wrapper, fetcher):
    window = MainWindow(wrapper, fetcher)
    window.timer.stop()
    window.on_layer_selected('conv2')

    for mode in ('activations', 'gradients', 'deconv', 'gradcam'):
        window.on_vis_mode_changed(mode)
        window.update_display()
        assert window._last_error is None, f'el modo {mode} falló'


def test_clicking_a_filter_updates_the_control_panel(qt_app, wrapper, fetcher):
    window = MainWindow(wrapper, fetcher)
    window.timer.stop()
    panel = window.control_panel
    panel.layer_combo.setCurrentIndex(panel.layer_names.index('conv1'))
    window.update_display()

    window.layer_view.filter_clicked.emit(3)
    assert window.control_panel.filter_spin.value() == 3
    assert window.current_filter == 3


def test_optimization_mode_renders_and_caches(qt_app, wrapper, fetcher):
    window = MainWindow(wrapper, fetcher)
    window.timer.stop()
    window.on_layer_selected('conv1')
    window.on_vis_mode_changed('optimization')

    window.update_display()
    assert window._last_error is None
    first = window.optimized_image
    assert first is not None

    # Sin cambiar la selección, la imagen optimizada se reutiliza.
    window._request_refresh()
    window.update_display()
    assert window.optimized_image is first

    # Cambiar de filtro invalida la caché.
    window.on_filter_selected(2)
    assert window.optimized_image is None


def test_gradcam_reports_unsupported_layers(qt_app, wrapper, fetcher):
    window = MainWindow(wrapper, fetcher)
    window.timer.stop()
    window.on_layer_selected('predictions')
    window.on_vis_mode_changed('gradcam')

    window.update_display()

    assert 'convolucional' in window.statusBar().currentMessage()
    assert window._last_error is None


def test_keyboard_shortcuts_are_dispatched(qt_app, wrapper, fetcher):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent

    window = MainWindow(wrapper, fetcher)
    window.timer.stop()
    panel = window.control_panel
    panel.layer_combo.setCurrentIndex(panel.layer_names.index('conv1'))
    window.update_display()

    def press(key):
        window.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
        )

    press(Qt.Key.Key_Down)
    assert panel.filter_spin.value() == 1

    press(Qt.Key.Key_Up)
    assert panel.filter_spin.value() == 0

    first = np.array(fetcher.current_raw_image)
    press(Qt.Key.Key_Right)
    assert not np.array_equal(first, fetcher.current_raw_image)

    press(Qt.Key.Key_Left)
    assert np.array_equal(first, fetcher.current_raw_image)

    press(Qt.Key.Key_H)
    assert 'Esc' in window.statusBar().currentMessage()


def test_predictions_are_skipped_without_an_imagenet_head(qt_app, wrapper, fetcher):
    window = MainWindow(wrapper, fetcher)
    window.timer.stop()

    # El modelo de juguete tiene 10 clases, así que no se puede etiquetar con ImageNet.
    assert window._has_imagenet_head is False
    assert 'no disponibles' in window.prediction_label.text()

    window.update_display()
    assert window._last_error is None


def test_navigation_keys_change_image(qt_app, wrapper, fetcher):
    window = MainWindow(wrapper, fetcher)
    window.timer.stop()
    window.update_display()

    first = np.array(fetcher.current_raw_image)
    window.next_image()
    assert not np.array_equal(first, fetcher.current_raw_image)


def test_save_current_view_writes_files(qt_app, wrapper, fetcher, tmp_path):
    window = MainWindow(wrapper, fetcher, output_dir=str(tmp_path / 'out'))
    window.timer.stop()
    window.on_layer_selected('conv1')
    window.update_display()

    window.save_current_view()
    assert list((tmp_path / 'out').glob('*.png'))


def test_close_releases_the_input_source(qt_app, wrapper, image_dir):
    source = InputFetcher(input_source=f'directory:{image_dir}', target_size=(32, 32))
    window = MainWindow(wrapper, source)
    window.close()
    assert window.timer.isActive() is False
