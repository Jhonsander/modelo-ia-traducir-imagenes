"""
Utilidades – Funciones auxiliares compartidas por los módulos del sistema.

Responsabilidades:
- Dibujar bounding boxes sobre una copia de la imagen (OpenCV).
- Detectar el idioma de un texto usando langdetect.
- Configurar y retornar el logger del sistema con el formato estándar.
- Retornar el timestamp actual en milisegundos.
"""

from __future__ import annotations

import logging
import time
from typing import List

import cv2
import numpy as np
from langdetect import detect


def dibujar_bboxes(imagen: np.ndarray, regiones: list) -> np.ndarray:
    """Dibuja rectángulos delimitadores sobre una copia de la imagen.

    Args:
        imagen: Imagen original como np.ndarray (BGR).
        regiones: Lista de RegionTexto cuyas bboxes se dibujarán.

    Returns:
        Copia de la imagen con los rectángulos dibujados.
    """
    raise NotImplementedError


def detectar_idioma(texto: str) -> str:
    """Detecta el idioma del texto usando langdetect.

    Args:
        texto: Texto del que se quiere detectar el idioma.

    Returns:
        Código ISO 639-1 del idioma detectado (p. ej. "en", "es", "ja").
    """
    raise NotImplementedError


def configurar_logging(nivel: str = "INFO") -> logging.Logger:
    """Configura y retorna el logger del sistema con el formato estándar.

    Formato: [TIMESTAMP] [NIVEL] [ETAPA] mensaje

    Args:
        nivel: Nivel de logging como cadena (p. ej. "INFO", "DEBUG", "ERROR").

    Returns:
        Instancia de logging.Logger configurada.
    """
    raise NotImplementedError


def milisegundos_actuales() -> int:
    """Retorna el timestamp actual en milisegundos.

    Returns:
        Timestamp Unix en milisegundos como entero.
    """
    raise NotImplementedError
