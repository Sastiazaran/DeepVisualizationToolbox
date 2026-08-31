# TensorFlow Feature Visualization Toolbox

An interactive tool for visualizing and understanding convolutional neural networks built with TensorFlow/Keras.

**Languages:** [English](README.en.md) · [Español](README.md)

[![CI](https://github.com/Sastiazaran/DeepVisualizationToolbox/actions/workflows/ci.yml/badge.svg)](https://github.com/Sastiazaran/DeepVisualizationToolbox/actions/workflows/ci.yml)

## Features

- **Real-time visualization**: watch network activations as it processes images from a webcam, a directory, or a dataset.
- **Five visualization modes**: activations, saliency maps, guided backpropagation, gradient-ascent optimization, and Grad-CAM.
- **Live predictions**: the top three ImageNet classes are shown next to the input image.
- **Intuitive UI**: navigate layers and filters with the keyboard or mouse, and save the current view with a single key.
- **Ten pretrained models**: VGG16/19, ResNet50, ResNet50V2, InceptionV3, MobileNet, MobileNetV2, EfficientNetB0, EfficientNetV2B0, and ConvNeXtTiny.
- **Custom models**: any Keras model works, either via `--model-file` or by using `ModelWrapper` from Python.

## Screenshots

![Intermediate layer activations](docs/images/main_screen.png)

*The interface shows the input image and its predictions (left) alongside activations for the selected layer (right).*

## Installation

### Requirements

- Python 3.10 or newer
- TensorFlow 2.16 or newer (includes Keras 3)
- PyQt6

### Install

```bash
git clone https://github.com/Sastiazaran/DeepVisualizationToolbox.git
cd DeepVisualizationToolbox

python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

pip install -e .
```

For development, install the dev tools as well:

```bash
pip install -e ".[dev]"
```

On Linux, PyQt6 needs a few system libraries:

```bash
sudo apt-get install libegl1 libgl1 libxkbcommon-x11-0 libdbus-1-3 libxcb-cursor0
```

## Usage

### Run the application

```bash
# Default setup (webcam + VGG16)
tf-feature-vis

# Equivalent, without installing the package
python run_toolbox.py
```

If no camera is available, the app automatically falls back to synthetic images instead of crashing.

### Examples

```bash
# List available models
tf-feature-vis --list-models

# Explore an image directory with ResNet50
tf-feature-vis --model resnet50 --input-source directory:./my_images

# A single image with InceptionV3 (automatically uses its 299×299 input)
tf-feature-vis --model inception_v3 --input-source file:./cat.jpg

# A Keras dataset, on GPU
tf-feature-vis --model mobilenet_v2 --input-source dataset:cifar10 --gpu

# A custom model saved to disk
tf-feature-vis --model-file ./my_model.keras --input-source synthetic
```

### Arguments

| Argument | Description |
| --- | --- |
| `--model` | Registry model (default: `vgg16`) |
| `--model-file` | Path to a saved model (`.keras`, `.h5`, or SavedModel) |
| `--weights` | `imagenet`, `none`, or a path to weights |
| `--no-top` | Exclude classification layers |
| `--input-source` | `webcam[:id]`, `file:<path>`, `directory:<path>`, `dataset:<cifar10\|mnist>`, or `synthetic` |
| `--input-size` | `WIDTH HEIGHT`; defaults to the model's native size |
| `--gpu` | Use the GPU for computation |
| `--output-dir` | Directory where saved visualizations are written |
| `--list-models` | List available models and exit |

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| `←` / `→` | Previous / next image |
| `↑` / `↓` | Previous / next filter |
| `S` | Save the current view |
| `H` | Show help |
| `Esc` | Quit |

## Library usage

Visualization components work without the GUI:

```python
import keras
import numpy as np

from tf_vis import ModelWrapper
from tf_vis.utils.misc import get_model_spec, load_model
from tf_vis.visualization import create_class_activation_map, overlay_heatmap

spec = get_model_spec("resnet50")
wrapper = ModelWrapper(load_model("resnet50"))

raw = np.array(keras.utils.load_img("cat.jpg", target_size=(224, 224)))
image = spec.preprocess(raw.astype("float32"))

# Layer activations
activations = wrapper.forward_pass(image, "conv3_block1_out")["conv3_block1_out"]

# Guided backprop for a specific filter
guided = wrapper.deconv(image, "conv3_block1_out", filter_indices=7)

# Grad-CAM for the predicted class
class_idx = int(np.argmax(wrapper.model.predict(image[None], verbose=0)[0]))
cam = create_class_activation_map(wrapper, image, "conv5_block3_out", class_idx)
heatmap = overlay_heatmap(raw, cam)
```

## Implemented techniques

| Mode | Technique | Reference |
| --- | --- | --- |
| Activations | Per-filter activation maps | — |
| Gradients | Saliency maps | Simonyan et al., 2014 |
| Deconvolution | Guided backpropagation | Springenberg et al., 2015 |
| Optimization | Gradient ascent on a filter | Erhan et al., 2009 |
| Grad-CAM | Class activation map via gradients | Selvaraju et al., 2017 |

## Development

```bash
pip install -e ".[dev]"

# Tests (UI runs headless)
QT_QPA_PLATFORM=offscreen pytest

# Linter
ruff check .
```

Tests use a small model built on the fly, so they do not download weights and finish in a few seconds.

To regenerate documentation images:

```bash
python docs/generate_examples.py --image my_image.jpg
```

## Documentation

The full guide is in [`docs/README.en.md`](docs/README.en.md) (English) and [`docs/README.md`](docs/README.md) (Spanish).

## License

MIT. See [LICENSE](LICENSE).
