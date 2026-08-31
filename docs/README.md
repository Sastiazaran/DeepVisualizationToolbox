# Documentación de TensorFlow Feature Visualization Toolbox

Esta documentación describe cómo usar y extender la herramienta de visualización de características para redes neuronales en TensorFlow/Keras.

**Idiomas:** [Español](README.md) · [English](README.en.md)

## Índice

1. [Introducción](#introducción)
2. [Instalación](#instalación)
3. [Uso básico](#uso-básico)
4. [Componentes principales](#componentes-principales)
5. [Visualización de modelos personalizados](#visualización-de-modelos-personalizados)
6. [Técnicas de visualización](#técnicas-de-visualización)
7. [Referencia de API](#referencia-de-api)
8. [Solución de problemas](#solución-de-problemas)

## Introducción

TensorFlow Feature Visualization Toolbox es una herramienta interactiva diseñada para ayudar a entender el funcionamiento interno de redes neuronales convolucionales. Permite observar en tiempo real cómo las distintas capas procesan una imagen, qué características detecta cada filtro, y qué regiones de la imagen sustentan una predicción.

### ¿Por qué visualizar redes neuronales?

Las redes profundas se comportan a menudo como cajas negras. Visualizar sus características ayuda a:

- **Entender qué aprende cada capa**: las primeras capas detectan bordes y texturas, y las profundas patrones cada vez más específicos.
- **Diagnosticar problemas**: identificar filtros muertos o excesivamente especializados.
- **Mejorar arquitecturas**: informar decisiones de diseño con evidencia.
- **Explicar predicciones**: mostrar en qué se fija el modelo al clasificar.

## Instalación

### Requisitos previos

- Python 3.10 o superior
- TensorFlow 2.16 o superior (que incluye Keras 3)
- PyQt6 para la interfaz gráfica

> **Nota sobre versiones.** A partir de TensorFlow 2.16, `tf.keras` es Keras 3, que eliminó atributos como `layer.output_shape` y `layer.batch_input_shape`. El toolbox usa las APIs de Keras 3 y no es compatible con Keras 2.

### Pasos de instalación

```bash
git clone https://github.com/Sastiazaran/DeepVisualizationToolbox.git
cd DeepVisualizationToolbox

python -m venv .venv
source .venv/bin/activate

pip install -e .          # o pip install -e ".[dev]" para desarrollar
```

En Linux, PyQt6 necesita algunas bibliotecas del sistema:

```bash
sudo apt-get install libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0
```

## Uso básico

### Ejecutar la aplicación

```bash
tf-feature-vis
```

Esto inicia la aplicación con VGG16 y la webcam. Si no hay cámara disponible, la fuente cae automáticamente a imágenes sintéticas.

### Opciones de línea de comandos

```bash
tf-feature-vis --model resnet50 --input-source directory:./mis_imagenes --gpu
```

| Argumento | Descripción |
| --- | --- |
| `--model` | Modelo del registro (por defecto `vgg16`) |
| `--model-file` | Ruta a un modelo guardado (`.keras`, `.h5` o SavedModel) |
| `--weights` | `imagenet`, `none` o una ruta a pesos |
| `--no-top` | Excluir las capas de clasificación |
| `--input-source` | `webcam[:id]`, `file:<ruta>`, `directory:<ruta>`, `dataset:<cifar10\|mnist>` o `synthetic` |
| `--input-size` | Tamaño `ANCHO ALTO`; por defecto el nativo del modelo |
| `--gpu` | Usar GPU para los cálculos |
| `--output-dir` | Directorio donde se guardan las visualizaciones |
| `--list-models` | Listar los modelos disponibles y salir |

### Navegación por la interfaz

1. **Panel izquierdo**
   - Imagen de entrada sin preprocesar
   - Las tres clases más probables según el modelo
   - Controles de capa, filtro y modo de visualización

2. **Panel derecho**
   - *Activaciones*: cuadrícula con los primeros 64 filtros de la capa
   - *Filtros*: saliencia o retropropagación guiada del filtro seleccionado
   - *Optimización*: imagen generada por ascenso de gradiente
   - *Grad-CAM*: mapa de calor de la clase predicha sobre la imagen

Al hacer clic en una celda de la cuadrícula de activaciones se selecciona ese filtro.

### Atajos de teclado

| Tecla | Acción |
| --- | --- |
| `←` / `→` | Imagen anterior / siguiente |
| `↑` / `↓` | Filtro anterior / siguiente |
| `S` | Guardar la vista actual en `--output-dir` |
| `H` | Mostrar ayuda |
| `Esc` | Cerrar la aplicación |

## Componentes principales

### ModelWrapper

`ModelWrapper` envuelve un modelo de Keras y expone las operaciones de visualización. Los extractores de activaciones se crean bajo demanda y se cachean, de modo que envolver un modelo grande es inmediato.

```python
import keras

from tf_vis import ModelWrapper

wrapper = ModelWrapper(keras.applications.VGG16())

# Introspección
wrapper.visualizable_layers()          # capas con salida 2D o 4D
wrapper.get_layer_info('block3_conv1') # nombre, tipo, forma, unidades, parámetros
wrapper.num_filters('block3_conv1')    # 256

# Activaciones y gradientes
activations = wrapper.forward_pass(image, 'block3_conv1')['block3_conv1']
gradients = wrapper.compute_gradients(image, 'block3_conv1', filter_indices=0)
saliency = wrapper.saliency_map(image, 'block3_conv1', filter_indices=0)
guided = wrapper.guided_backprop(image, 'block3_conv1', filter_indices=0)
```

### InputFetcher

`InputFetcher` obtiene imágenes de distintas fuentes y las preprocesa para el modelo. Devuelve la imagen preprocesada y conserva la original en `current_raw_image`, para poder mostrarla sin los desplazamientos que introduce el preprocesamiento.

```python
from tf_vis import InputFetcher

with InputFetcher(input_source='directory:./imagenes',
                  preprocessing_function=preprocess,
                  target_size=(224, 224)) as fetcher:
    image = fetcher.get_next_image()   # preprocesada
    original = fetcher.current_raw_image  # RGB uint8
```

El tamaño objetivo se indica como `(ancho, alto)`, igual que en OpenCV.

### Visualización

El módulo `tf_vis.visualization` reúne las técnicas que no dependen de la interfaz:

```python
from tf_vis.visualization import (
    apply_gradient_ascent,
    create_class_activation_map,
    display_activation_grid,
    overlay_heatmap,
    visualize_layer_filters,
    visualize_max_activations,
)

optimized = apply_gradient_ascent(wrapper, 'block3_conv1', filter_index=0)
cam = create_class_activation_map(wrapper, image, 'block5_conv3', class_idx=242)
heatmap = overlay_heatmap(original, cam)
```

## Visualización de modelos personalizados

Desde la línea de comandos basta con `--model-file`:

```bash
tf-feature-vis --model-file ./mi_modelo.keras --input-source directory:./imagenes
```

Desde Python se pueden ensamblar los componentes a mano:

```python
import sys

import keras
from PyQt6.QtWidgets import QApplication

from tf_vis import InputFetcher, ModelWrapper
from tf_vis.ui.main_window import MainWindow

model = keras.models.load_model('mi_modelo.keras')

input_fetcher = InputFetcher(
    input_source='directory:./mis_imagenes',
    preprocessing_function=lambda img: img / 255.0,
    target_size=(224, 224),
)

app = QApplication(sys.argv)
window = MainWindow(ModelWrapper(model), input_fetcher)
sys.exit(app.exec())
```

Requisitos del modelo:

- Debe ser un modelo funcional o secuencial ya construido; los modelos subclasificados que nunca se han llamado no exponen tensores de entrada.
- Las capas deben tener nombres únicos.
- Para el ascenso de gradiente a cualquier resolución, cárgalo con `include_top=False`; con la cabeza puesta, la entrada tiene un tamaño fijo.

## Técnicas de visualización

### Activaciones

Muestra la salida de cada filtro de una capa para la imagen actual. Revela qué partes de la imagen responden a cada detector.

![Activaciones de block3_conv1 en VGG16](images/activations_grid.png)

*Generado con `python docs/generate_examples.py --model vgg16 --layer block3_conv1`. Unos filtros responden a los bordes del cuadrado y otros al patrón de franjas.*

### Mapas de saliencia

Gradiente de la activación respecto a los píxeles de entrada, tomando el máximo del valor absoluto sobre los canales de color (Simonyan et al., 2014). Se recorta el percentil 99 para que un único píxel extremo no oscurezca el resto del mapa.

### Retropropagación guiada

Variante de la deconvolución de Zeiler & Fergus propuesta por Springenberg et al. (2015): en cada ReLU se propagan hacia atrás únicamente los gradientes positivos que provienen de activaciones positivas. Produce visualizaciones mucho más nítidas que el gradiente en bruto.

Se cubren las dos formas de declarar una ReLU en Keras: como argumento `activation` (VGG, InceptionV3) y como capa `ReLU` independiente (ResNet, MobileNet, EfficientNet).

### Optimización de características

Genera por ascenso de gradiente una imagen que maximiza la activación de un filtro, revelando el patrón que ese filtro busca. Se descartan los bordes del mapa de características, dominados por el padding.

### Grad-CAM

Pondera los mapas de características de una capa convolucional por el gradiente medio de la clase objetivo (Selvaraju et al., 2017). A diferencia del CAM original, no exige una arquitectura con *global average pooling* seguido de una única capa densa, por lo que funciona con cualquier modelo del registro.

## Referencia de API

### `tf_vis.model_wrapper.ModelWrapper`

| Método | Descripción |
| --- | --- |
| `forward_pass(image, layer_name=None)` | Activaciones de una capa o de todas las visualizables |
| `compute_gradients(image, layer_name, filter_indices=None)` | Gradiente respecto a la entrada |
| `saliency_map(image, layer_name, filter_indices=None)` | Mapa de saliencia 2D en [0, 1] |
| `guided_backprop(image, layer_name, filter_indices=None)` | Gradiente guiado |
| `deconv(image, layer_name, filter_indices=None)` | Gradiente guiado listo para mostrar |
| `get_activation_model(layer_name)` | Sub-modelo cacheado que produce la activación |
| `get_layer_info(layer_name=None)` | Nombre, tipo, forma, unidades y parámetros |
| `get_layer_shape(layer_name)` | Forma de salida, o `None` |
| `num_filters(layer_name)` | Filtros o neuronas de la capa |
| `is_spatial_layer(layer_name)` | Si la salida es un mapa de características |
| `visualizable_layers()` | Capas con salida 2D o 4D |
| `clear_cache()` | Libera los sub-modelos cacheados |

### `tf_vis.input_fetcher.InputFetcher`

| Miembro | Descripción |
| --- | --- |
| `get_next_image()` | Siguiente imagen, preprocesada |
| `get_previous_image()` | Imagen anterior |
| `get_current_image()` | Imagen actual sin avanzar el índice |
| `get_specific_image(index)` | Imagen por índice |
| `current_raw_image` | Última imagen RGB sin preprocesar |
| `is_live` | Si la fuente es una webcam activa |
| `len(fetcher)` | Número de imágenes (0 en fuentes en vivo) |
| `close()` | Libera la cámara |

### `tf_vis.visualization`

| Función | Descripción |
| --- | --- |
| `display_activation_grid(activations, grid_size=None, padding=1)` | Cuadrícula a partir de `[n_filtros, alto, ancho]` |
| `visualize_layer_filters(model, layer_name, grid_size=None)` | Cuadrícula con los kernels de una capa |
| `apply_gradient_ascent(wrapper, layer_name, filter_index, ...)` | Imagen que maximiza un filtro |
| `visualize_max_activations(wrapper, dataset, layer_name, ...)` | Imágenes que más activan cada filtro |
| `create_class_activation_map(wrapper, img, layer_name, class_idx)` | Grad-CAM |
| `overlay_heatmap(img, heatmap, alpha=0.5, colormap=...)` | Superposición de un mapa de calor |
| `normalize_01(array)` | Escalado a [0, 1] |
| `standardize_for_display(array, spread=0.25)` | Normalización por contraste para gradientes |
| `clip_outliers(array, percentile=99.0)` | Recorte de la cola superior |
| `resolve_ascent_size(wrapper, image_size)` | Tamaño válido para el ascenso de gradiente |

### `tf_vis.utils.misc`

| Función | Descripción |
| --- | --- |
| `get_model_specs()` | Registro `nombre -> ModelSpec` |
| `get_model_spec(name)` | Especificación de un modelo |
| `load_model(name, weights='imagenet', include_top=True)` | Carga un modelo del registro |
| `load_model_from_file(path)` | Carga un modelo de disco |
| `get_preprocessing_function(name)` | Función de preprocesamiento del modelo |
| `predict_image(model, img, top_k=5, model_name=None)` | Clases top-k con etiquetas |
| `get_imagenet_labels()` | Las 1000 etiquetas en el orden de Keras |
| `create_model_summary(model)` | Resumen serializable del modelo |
| `save_visualizations(visualizations, output_dir, prefix='')` | Guarda PNG y devuelve las rutas |

### `tf_vis.utils.layers`

| Función | Descripción |
| --- | --- |
| `layer_output_shape(layer)` | Forma de salida compatible con Keras 3 |
| `layer_num_units(layer)` | Filtros o neuronas de la capa |
| `is_spatial_shape(shape)` | Si la forma es `[batch, alto, ancho, canales]` |
| `describe_layer(layer)` | Resumen de la capa como diccionario |

## Solución de problemas

### `AttributeError: 'Conv2D' object has no attribute 'output_shape'`

Estás mezclando código escrito para Keras 2 con Keras 3. Usa `tf_vis.utils.layers.layer_output_shape`, que funciona en ambas versiones.

### `ImportError: libEGL.so.1: cannot open shared object file`

Faltan las bibliotecas de sistema de Qt. En Debian o Ubuntu:

```bash
sudo apt-get install libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0
```

Para ejecutar sin servidor gráfico, exporta `QT_QPA_PLATFORM=offscreen`.

### La webcam no se abre

La aplicación avisa por consola y cambia a imágenes sintéticas. Para forzar otra cámara usa `--input-source webcam:1`, y para trabajar con archivos `--input-source directory:./imagenes`.

### `El modelo espera entradas de NxN`

El ascenso de gradiente necesita generar una imagen del tamaño de entrada del modelo. Los modelos con `include_top=True` fijan esa resolución; carga el modelo con `--no-top` para poder usar cualquier tamaño.

### Rendimiento lento

- Usa `--gpu` si tienes una GPU configurada.
- Reduce la resolución con `--input-size`.
- Los modos de optimización y Grad-CAM son los más costosos; el de activaciones es el más ligero.
- Con `--model-file` sobre modelos grandes, las primeras iteraciones incluyen la compilación del grafo.

### Errores con modelos personalizados

- El modelo debe estar construido y exponer tensores de entrada.
- Las capas deben tener nombres únicos.
- Proporciona una función de preprocesamiento acorde al entrenamiento del modelo.

## Obtener ayuda

1. Revisa los [problemas conocidos](https://github.com/Sastiazaran/DeepVisualizationToolbox/issues).
2. Abre un nuevo issue con la versión de Python, TensorFlow y el traceback completo.
