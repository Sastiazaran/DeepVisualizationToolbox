"""
Aplicación principal para la visualización de características de redes neuronales.
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        prog='tf-feature-vis',
        description='TensorFlow Feature Visualization Toolbox',
    )

    model_group = parser.add_argument_group('modelo')
    model_group.add_argument('--model', type=str, default='vgg16',
                             help='Modelo del registro a utilizar (por defecto: vgg16)')
    model_group.add_argument('--model-file', type=str, default=None,
                             help='Ruta a un modelo guardado (.keras, .h5 o SavedModel); '
                                  'tiene prioridad sobre --model')
    model_group.add_argument('--weights', type=str, default='imagenet',
                             help="Pesos a utilizar ('imagenet', 'none' o una ruta)")
    model_group.add_argument('--no-top', dest='include_top', action='store_false',
                             help='Excluir las capas de clasificación')
    model_group.set_defaults(include_top=True)

    input_group = parser.add_argument_group('entrada')
    input_group.add_argument('--input-source', type=str, default='webcam',
                             help='webcam[:id], file:<ruta>, directory:<ruta>, '
                                  'dataset:<cifar10|mnist> o synthetic')
    input_group.add_argument('--input-size', type=int, nargs=2, default=None,
                             metavar=('ANCHO', 'ALTO'),
                             help='Tamaño de entrada; por defecto el nativo del modelo')

    misc_group = parser.add_argument_group('varios')
    misc_group.add_argument('--gpu', action='store_true', help='Usar GPU para los cálculos')
    misc_group.add_argument('--output-dir', type=str, default='visualizations',
                            help='Directorio donde se guardan las visualizaciones')
    misc_group.add_argument('--list-models', action='store_true',
                            help='Listar los modelos disponibles y salir')

    return parser.parse_args(argv)


def _configure_devices(use_gpu: bool) -> None:
    """Configura la visibilidad de la GPU antes de importar TensorFlow."""
    if not use_gpu:
        # Debe hacerse antes de importar TensorFlow para que surta efecto.
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Ejecutando en CPU (usa --gpu para habilitar la GPU)")


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la aplicación."""
    args = parse_args(argv)
    _configure_devices(args.gpu)

    from .utils.misc import get_model_spec, get_model_specs, load_model_from_file

    if args.list_models:
        for name, spec in sorted(get_model_specs().items()):
            print(f"{name:<20} entrada {spec.input_shape}")
        return 0

    try:
        if args.model_file:
            print(f"Cargando modelo desde {args.model_file}...")
            model = load_model_from_file(args.model_file)
            preprocess_fn = None
            model_name = None
            default_size = None
        else:
            print(f"Cargando modelo {args.model}...")
            weights = None if args.weights.lower() == 'none' else args.weights
            spec = get_model_spec(args.model)
            model = spec.constructor(weights=weights, include_top=args.include_top)
            preprocess_fn = spec.preprocess
            model_name = args.model
            default_size = (spec.input_shape[1], spec.input_shape[0])
        print("Modelo cargado correctamente")
    except (ValueError, OSError) as error:
        print(f"Error al cargar el modelo: {error}")
        print(f"Modelos disponibles: {sorted(get_model_specs())}")
        return 1

    from .input_fetcher import InputFetcher
    from .model_wrapper import ModelWrapper

    model_wrapper = ModelWrapper(model)

    input_size = tuple(args.input_size) if args.input_size else (default_size or (224, 224))

    try:
        input_fetcher = InputFetcher(
            input_source=args.input_source,
            preprocessing_function=preprocess_fn,
            target_size=input_size,
        )
        print(f"Fuente de entrada configurada: {args.input_source} {input_size}")
    except ValueError as error:
        print(f"Error al configurar la fuente de entrada: {error}")
        return 1

    from PyQt6.QtWidgets import QApplication

    from .ui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    window = MainWindow(model_wrapper, input_fetcher, model_name=model_name,
                        output_dir=args.output_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
