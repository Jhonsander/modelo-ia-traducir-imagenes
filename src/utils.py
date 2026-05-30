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
    # Crear una copia de la imagen para no modificar la original
    imagen_anotada = imagen.copy()
    
    # Dibujar un rectángulo para cada región
    for region in regiones:
        # Convertir los 4 puntos del bbox a un array de puntos para cv2.polylines
        puntos = np.array(region.bbox, dtype=np.int32)
        
        # Dibujar el polígono (rectángulo) con líneas verdes de grosor 2
        cv2.polylines(imagen_anotada, [puntos], isClosed=True, 
                     color=(0, 255, 0), thickness=2)
    
    return imagen_anotada


def detectar_idioma(texto: str) -> str:
    """Detecta el idioma del texto usando langdetect.

    Args:
        texto: Texto del que se quiere detectar el idioma.

    Returns:
        Código ISO 639-1 del idioma detectado (p. ej. "en", "es", "ja").
    """
    return detect(texto)


def configurar_logging(nivel: str = "INFO") -> logging.Logger:
    """Configura y retorna el logger del sistema con el formato estándar.

    Formato: [TIMESTAMP] [NIVEL] [ETAPA] mensaje

    Args:
        nivel: Nivel de logging como cadena (p. ej. "INFO", "DEBUG", "ERROR").

    Returns:
        Instancia de logging.Logger configurada.
    """
    # Crear o obtener el logger del sistema
    logger = logging.getLogger("multimodal_image_translation")
    
    # Configurar el nivel de logging
    logger.setLevel(getattr(logging, nivel.upper()))
    
    # Evitar duplicar handlers si ya está configurado
    if not logger.handlers:
        # Crear un handler para la consola
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, nivel.upper()))
        
        # Definir el formato: [TIMESTAMP] [NIVEL] [ETAPA] mensaje
        # Nota: [ETAPA] se añadirá en el mensaje por el código que llama al logger
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Añadir el handler al logger
        logger.addHandler(handler)
    
    return logger


def milisegundos_actuales() -> int:
    """Retorna el timestamp actual en milisegundos.

    Returns:
        Timestamp Unix en milisegundos como entero.
    """
    return int(time.time() * 1000)
