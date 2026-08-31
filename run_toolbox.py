#!/usr/bin/env python3
"""
Script para ejecutar la aplicación de visualización de características.

Equivalente al comando `tf-feature-vis` que instala el paquete.
"""

import sys

from tf_vis.app import main

if __name__ == "__main__":
    sys.exit(main())
