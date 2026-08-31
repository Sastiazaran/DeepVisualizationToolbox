"""
TensorFlow Feature Visualization Toolbox

Una herramienta interactiva para visualizar y entender redes neuronales
convolucionales implementadas en TensorFlow/Keras.
"""

from typing import TYPE_CHECKING

__version__ = "0.2.0"

__all__ = ["InputFetcher", "ModelWrapper", "__version__"]

if TYPE_CHECKING:  # pragma: no cover
    from .input_fetcher import InputFetcher
    from .model_wrapper import ModelWrapper


def __getattr__(name: str):
    """Importa TensorFlow y OpenCV solo cuando se usa realmente el paquete."""
    if name == "ModelWrapper":
        from .model_wrapper import ModelWrapper

        return ModelWrapper
    if name == "InputFetcher":
        from .input_fetcher import InputFetcher

        return InputFetcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
