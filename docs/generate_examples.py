#!/usr/bin/env python3
"""
Genera las imágenes de ejemplo que aparecen en la documentación.

Uso:
    python docs/generate_examples.py [--image ruta/a/imagen.jpg] [--layer block3_conv1]
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tf_vis.model_wrapper import ModelWrapper  # noqa: E402
from tf_vis.utils.image_utils import load_image  # noqa: E402
from tf_vis.utils.misc import get_model_spec, load_model  # noqa: E402
from tf_vis.visualization import normalize_01  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'images')


def synthetic_image(size: int = 224) -> np.ndarray:
    """
    Construye una imagen con bordes, círculos y texturas.

    Un patrón con estructura activa filtros reconocibles, a diferencia del ruido
    puro, que produce activaciones planas poco ilustrativas.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)

    stripes = 0.5 + 0.5 * np.sin(x / 6.0)
    rings = 0.5 + 0.5 * np.cos(np.hypot(x - size / 2, y - size / 2) / 8.0)
    square = ((np.abs(x - size * 0.3) < 30) & (np.abs(y - size * 0.7) < 30)).astype(np.float32)

    image = np.stack([stripes, rings, np.clip(stripes * 0.4 + square, 0, 1)], axis=-1)
    return (image * 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=str, default=None,
                        help='Imagen de entrada; por defecto se genera un patrón sintético')
    parser.add_argument('--model', type=str, default='vgg16', help='Modelo del registro')
    parser.add_argument('--layer', type=str, default='block3_conv1', help='Capa a visualizar')
    parser.add_argument('--n-filters', type=int, default=64,
                        help='Número de filtros a dibujar en la cuadrícula')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    spec = get_model_spec(args.model)
    height, width = spec.input_shape[:2]

    raw = (load_image(args.image, target_size=(width, height)) if args.image
           else synthetic_image(height))
    preprocessed = spec.preprocess(raw.astype(np.float32).copy())

    wrapper = ModelWrapper(load_model(args.model))
    activations = wrapper.forward_pass(preprocessed, args.layer)[args.layer][0]

    n_filters = min(args.n_filters, activations.shape[-1])
    side = int(np.ceil(np.sqrt(n_filters)))

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f'{args.model} — activaciones de la capa {args.layer}', fontsize=16)

    input_ax = fig.add_subplot(1, 2, 1)
    input_ax.imshow(raw)
    input_ax.set_title('Imagen de entrada')
    input_ax.axis('off')

    grid_spec = fig.add_gridspec(side, 2 * side)
    for i in range(n_filters):
        row, col = divmod(i, side)
        ax = fig.add_subplot(grid_spec[row, side + col])
        ax.imshow(normalize_01(activations[:, :, i]), cmap='viridis')
        ax.axis('off')

    output_path = os.path.join(OUTPUT_DIR, 'main_screen.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Imagen de ejemplo generada en {output_path}")


if __name__ == '__main__':
    main()
