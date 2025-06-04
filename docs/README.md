# Documentación de TensorFlow Feature Visualization Toolbox

Esta documentación proporciona información detallada sobre cómo usar y extender la herramienta de visualización de características para redes neuronales en TensorFlow.

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

TensorFlow Feature Visualization Toolbox es una herramienta interactiva diseñada para ayudar a entender y visualizar el funcionamiento interno de redes neuronales convolucionales. Permite observar en tiempo real cómo diferentes capas de la red procesan imágenes, qué características detectan los filtros, y cómo se pueden generar imágenes que maximizan la activación de neuronas específicas.

### ¿Por qué visualizar redes neuronales?

Las redes neuronales profundas a menudo se consideran "cajas negras" debido a su complejidad y la dificultad para interpretar su funcionamiento interno. La visualización de características ayuda a:

- **Entender qué aprende cada capa**: Las primeras capas suelen detectar bordes y texturas simples, mientras que las capas más profundas detectan patrones más complejos y específicos.
- **Diagnosticar problemas**: Identificar filtros que no se activan o que se especializan demasiado.
- **Mejorar arquitecturas**: Informar decisiones sobre el diseño de nuevas arquitecturas de red.
- **Explicar predicciones**: Proporcionar información sobre por qué la red toma ciertas decisiones.

## Instalación

### Requisitos previos

- Python 3.7 o superior
- TensorFlow 2.4 o superior
- PyQt5 para la interfaz gráfica
- Dependencias adicionales listadas en `requirements.txt`

### Pasos de instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/tf-feature-vis.git
   cd tf-feature-vis
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Instalar el paquete en modo desarrollo:
   ```bash
   pip install -e .
   ```

## Uso básico

### Ejecutar la aplicación

La forma más sencilla de iniciar la aplicación es:

```bash
python run_toolbox.py
```

Esto iniciará la aplicación con la configuración predeterminada (modelo VGG16 y entrada de webcam).

### Opciones de línea de comandos

La aplicación acepta varios argumentos para personalizar su comportamiento:

```bash
python run_toolbox.py --model resnet50 --input-source directory:./mis_imagenes --gpu
```

Argumentos disponibles:

- `--model`: Modelo a utilizar (vgg16, resnet50, inception_v3, etc.)
- `--weights`: Pesos a utilizar (imagenet o ruta a archivo)
- `--include-top`: Incluir capas de clasificación (predeterminado: True)
- `--input-source`: Fuente de entrada (webcam, directory:ruta, file:ruta, dataset:nombre)
- `--input-size`: Tamaño de entrada como ancho alto (predeterminado: 224 224)
- `--gpu`: Usar GPU para cálculos

### Navegación por la interfaz

La interfaz se divide en varias secciones:

1. **Panel izquierdo**:
   - Visualización de la imagen de entrada
   - Controles para seleccionar capa y filtro
   - Opciones de modo de visualización

2. **Panel derecho**:
   - Pestañas para diferentes visualizaciones:
     - Activaciones: Muestra activaciones de todos los filtros
     - Filtros: Visualiza gradientes o deconvolución para un filtro específico
     - Optimización: Muestra imágenes generadas para maximizar activaciones
     - CAM: Muestra mapas de activación de clase

### Atajos de teclado

- **Flechas izquierda/derecha**: Navegar entre imágenes
- **Flechas arriba/abajo**: Navegar entre filtros
- **H**: Mostrar ayuda
- **Esc**: Cerrar la aplicación

## Componentes principales

### ModelWrapper

`ModelWrapper` es una clase que envuelve un modelo de TensorFlow y proporciona métodos para:

- Obtener activaciones de cualquier capa
- Calcular gradientes con respecto a cualquier capa
- Realizar deconvolución
- Optimizar imágenes para maximizar activaciones

```python
from tf_vis.model_wrapper import ModelWrapper
import tensorflow as tf

# Cargar modelo
model = tf.keras.applications.VGG16()

# Crear wrapper
wrapper = ModelWrapper(model)

# Obtener activaciones
activations = wrapper.forward_pass(image, 'block3_conv1')

# Calcular gradientes
gradients = wrapper.compute_gradients(image, 'block3_conv1', filter_index=0)
```

### InputFetcher

`InputFetcher` gestiona la obtención de imágenes de diferentes fuentes:

- Webcam
- Archivos de imagen
- Directorios de imágenes
- Datasets de TensorFlow

```python
from tf_vis.input_fetcher import InputFetcher

# Crear fetcher para webcam
webcam_fetcher = InputFetcher(input_source='webcam', target_size=(224, 224))

# Crear fetcher para directorio
dir_fetcher = InputFetcher(input_source='directory:/ruta/a/imagenes', target_size=(224, 224))

# Obtener siguiente imagen
image = fetcher.get_next_image()
```

### Visualización

El módulo `visualization.py` contiene funciones para diferentes técnicas de visualización:

- Visualización de activaciones
- Visualización de filtros
- Ascenso de gradiente para maximizar activaciones
- Mapas de activación de clase (CAM)

```python
from tf_vis.visualization import apply_gradient_ascent, create_class_activation_map

# Generar imagen que maximiza un filtro
optimized_image = apply_gradient_ascent(model_wrapper, 'block3_conv1', 0)

# Crear mapa de activación de clase
cam = create_class_activation_map(model_wrapper, image, 'block5_conv3', class_idx=242)
```

## Visualización de modelos personalizados

Para visualizar tu propio modelo, necesitas:

1. Cargar el modelo con TensorFlow
2. Crear un `ModelWrapper` para el modelo
3. Configurar un `InputFetcher` para la fuente de imágenes
4. Iniciar la aplicación con estos componentes

### Ejemplo con modelo personalizado

```python
import tensorflow as tf
from tf_vis.model_wrapper import ModelWrapper
from tf_vis.input_fetcher import InputFetcher
from tf_vis.ui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import sys

# Cargar tu modelo personalizado
model = tf.keras.models.load_model('mi_modelo.h5')

# Función de preprocesamiento personalizada (opcional)
def preprocess(img):
    return img / 255.0

# Crear componentes
model_wrapper = ModelWrapper(model)
input_fetcher = InputFetcher(
    input_source='directory:./mis_imagenes',
    preprocessing_function=preprocess,
    target_size=(224, 224)
)

# Iniciar aplicación
app = QApplication(sys.argv)
window = MainWindow(model_wrapper, input_fetcher)
sys.exit(app.exec_())
```

## Técnicas de visualización

### Visualización de activaciones

Muestra las activaciones de cada filtro en una capa específica cuando se procesa una imagen. Permite ver qué partes de la imagen activan cada filtro.

### Visualización de gradientes

Calcula los gradientes de la activación de un filtro con respecto a la imagen de entrada. Esto muestra qué píxeles de la imagen tienen mayor influencia en la activación del filtro.

### Deconvolución

Implementa la técnica de deconvolución propuesta por Zeiler & Fergus (2014), que proyecta las activaciones de vuelta al espacio de entrada para visualizar qué partes de la imagen activan un filtro específico.

### Optimización de características

Genera imágenes que maximizan la activación de un filtro específico mediante ascenso de gradiente. Esto revela los patrones ideales que cada filtro busca.

### Mapas de activación de clase (CAM)

Visualiza qué regiones de una imagen son importantes para la clasificación en una clase específica, superponiendo un mapa de calor sobre la imagen original.

## Referencia de API

### tf_vis.model_wrapper.ModelWrapper

```python
class ModelWrapper:
    def __init__(self, model: tf.keras.Model):
        """Inicializa el wrapper con un modelo TensorFlow."""
        
    def forward_pass(self, image: np.ndarray, layer_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Realiza un pase hacia adelante y devuelve las activaciones."""
        
    def compute_gradients(self, image: np.ndarray, layer_name: str, 
                         filter_indices: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        """Calcula gradientes de la activación con respecto a la imagen de entrada."""
        
    def deconv(self, image: np.ndarray, layer_name: str, 
              filter_indices: Optional[Union[int, List[int]]] = None) -> np.ndarray:
        """Implementa deconvolución para visualizar qué partes de la imagen activan ciertos filtros."""
        
    def get_layer_info(self, layer_name: Optional[str] = None) -> Dict:
        """Obtiene información sobre las capas del modelo."""
```

### tf_vis.input_fetcher.InputFetcher

```python
class InputFetcher:
    def __init__(self, input_source: str = 'webcam', 
                preprocessing_function: Optional[callable] = None,
                target_size: Tuple[int, int] = (224, 224)):
        """Inicializa el fetcher de entrada."""
        
    def get_next_image(self) -> np.ndarray:
        """Obtiene la siguiente imagen de la fuente."""
        
    def get_previous_image(self) -> np.ndarray:
        """Obtiene la imagen anterior."""
        
    def get_specific_image(self, index: int) -> np.ndarray:
        """Obtiene una imagen específica por índice."""
        
    def close(self):
        """Libera recursos."""
```

### tf_vis.visualization

```python
def display_activation_grid(activations: np.ndarray, grid_size: Optional[Tuple[int, int]] = None, 
                           padding: int = 1) -> np.ndarray:
    """Organiza las activaciones en una cuadrícula para visualización."""
    
def visualize_layer_filters(model: tf.keras.Model, layer_name: str, 
                           grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Visualiza los filtros de una capa convolucional."""
    
def apply_gradient_ascent(model_wrapper, layer_name: str, filter_index: int, 
                         iterations: int = 30, step_size: float = 1.0,
                         image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Aplica ascenso de gradiente para generar una imagen que maximiza la activación."""
    
def visualize_max_activations(model_wrapper, dataset, layer_name: str, 
                             n_top: int = 9, n_filters: Optional[int] = None) -> Dict:
    """Encuentra las imágenes que causan las activaciones máximas para cada filtro."""
    
def create_class_activation_map(model_wrapper, img: np.ndarray, 
                               layer_name: str, class_idx: int) -> np.ndarray:
    """Crea un mapa de activación de clase (CAM)."""
    
def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, 
                   alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Superpone un mapa de calor en una imagen."""
```

## Solución de problemas

### Problemas comunes

#### La aplicación no inicia

- Verifica que todas las dependencias estén instaladas correctamente
- Comprueba que TensorFlow esté configurado adecuadamente
- Si usas GPU, asegúrate de que CUDA y cuDNN estén instalados correctamente

#### Rendimiento lento

- Considera usar la opción `--gpu` para acelerar los cálculos
- Reduce el tamaño de entrada con `--input-size`
- Limita la visualización a capas específicas

#### Errores con modelos personalizados

- Asegúrate de que el modelo sea compatible con TensorFlow 2.x
- Verifica que todas las capas tengan nombres únicos
- Proporciona una función de preprocesamiento adecuada

### Obtener ayuda

Si encuentras problemas o tienes preguntas:

1. Consulta la [documentación completa](https://github.com/tu-usuario/tf-feature-vis/docs)
2. Revisa los [problemas conocidos](https://github.com/tu-usuario/tf-feature-vis/issues)
3. Abre un nuevo issue en GitHub con detalles sobre tu problema
