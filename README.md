# TensorFlow Feature Visualization Toolbox

Una herramienta interactiva para visualizar y entender redes neuronales convolucionales implementadas en TensorFlow.

## Características

- **Visualización en tiempo real**: Observa las activaciones de la red mientras procesa imágenes de una webcam o archivos.
- **Múltiples modos de visualización**: Activaciones, gradientes, deconvolución y optimización de características.
- **Interfaz intuitiva**: Navegación sencilla entre capas y filtros con una interfaz gráfica amigable.
- **Soporte para modelos populares**: VGG16, ResNet50, InceptionV3, MobileNet y más.
- **Personalizable**: Fácil de adaptar para visualizar tus propios modelos.

## Capturas de pantalla

![Captura de pantalla principal](docs/images/main_screen.png)

*La interfaz principal muestra la imagen de entrada (izquierda) y las activaciones de la capa seleccionada (derecha).*

## Instalación

### Requisitos

- Python 3.7+
- TensorFlow 2.4+
- PyQt5
- OpenCV
- NumPy, SciPy, Matplotlib

### Instalación con pip

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/tf-feature-vis.git
cd tf-feature-vis

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete
pip install -e .
```

## Uso

### Ejecutar la aplicación

```bash
# Ejecutar con configuración predeterminada (webcam + VGG16)
python run_toolbox.py

# Especificar un modelo diferente
python run_toolbox.py --model resnet50

# Usar imágenes de un directorio
python run_toolbox.py --input-source directory:./input_images

# Usar GPU para cálculos
python run_toolbox.py --gpu
```

### Controles

- **Teclas de flecha**: Navegar entre imágenes (izquierda/derecha) y filtros (arriba/abajo)
- **Esc**: Cerrar la aplicación
- **H**: Mostrar ayuda

## Modos de visualización

### Activaciones

Visualiza las activaciones de cada filtro en la capa seleccionada. Muestra qué partes de la imagen activan cada filtro.

### Gradientes

Visualiza los gradientes de la activación con respecto a la imagen de entrada. Muestra qué píxeles influyen más en la activación de un filtro específico.

### Deconvolución

Implementa la técnica de deconvolución de Zeiler & Fergus para visualizar qué partes de la imagen activan un filtro específico.

### Optimización

Genera imágenes que maximizan la activación de un filtro específico mediante ascenso de gradiente.

## Visualización de modelos personalizados

Para visualizar tu propio modelo:

```python
import tensorflow as tf
from tf_vis.model_wrapper import ModelWrapper
from tf_vis.input_fetcher import InputFetcher
from tf_vis.ui.main_window import MainWindow
from PyQt5.QtWidgets import QApplication
import sys

# Cargar tu modelo
model = tf.keras.models.load_model('ruta/a/tu/modelo')

# Crear wrapper y fetcher
model_wrapper = ModelWrapper(model)
input_fetcher = InputFetcher(input_source='webcam', target_size=(224, 224))

# Iniciar aplicación
app = QApplication(sys.argv)
window = MainWindow(model_wrapper, input_fetcher)
sys.exit(app.exec_())
```

## Documentación

Para más detalles sobre la API y ejemplos de uso, consulta la [documentación completa](docs/README.md).

## Citar este proyecto

Si utilizas este proyecto en tu investigación, por favor cítalo:

```
@software{tf_feature_vis,
  author = {Tu Nombre},
  title = {TensorFlow Feature Visualization Toolbox},
  year = {2025},
  url = {https://github.com/tu-usuario/tf-feature-vis}
}
```

## Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).
