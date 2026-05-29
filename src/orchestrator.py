"""
Orquestador – Coordina el pipeline de traducción de imágenes.

Responsabilidades:
- Gestionar el estado de la sesión (imagen cargada, módulos inicializados).
- Ejecutar el pipeline en orden: cargar_imagen → detectar_texto → traducir →
  construcción de ResultadoTraduccion.
- Capturar excepciones de los módulos, registrarlas en log y retornar un
  ResultadoTraduccion con el campo error poblado.
- Registrar el tiempo de ejecución en ms de cada etapa del pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from src.ocr_module import ModuloOCR, RegionTexto
from src.translation_module import ModuloTraduccion
from src import utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ResultadoTraduccion:
    """Resultado completo del pipeline de traducción de imágenes.

    Attributes:
        texto_original: Texto consolidado extraído de la imagen por OCR.
        idioma_origen: Código ISO 639-1 del idioma detectado (p. ej. "en").
        texto_traducido: Texto traducido al idioma destino.
        idioma_destino: Código ISO 639-1 del idioma destino (p. ej. "es").
        regiones: Lista de RegionTexto detectadas (para dibujar bboxes).
        error: Cadena vacía si no hay error; descripción del fallo si lo hay.
        tiempos_ms: Tiempos de ejecución en ms por etapa
                    {"ocr": int, "traduccion": int, "construccion": int}.
    """

    texto_original: str = ""
    idioma_origen: str = ""
    texto_traducido: str = ""
    idioma_destino: str = ""
    regiones: list = field(default_factory=list)  # list[RegionTexto]
    error: str = ""
    tiempos_ms: dict = field(default_factory=dict)  # {"ocr": int, ...}


# ---------------------------------------------------------------------------
# TraductorDeImagenes
# ---------------------------------------------------------------------------

class TraductorDeImagenes:
    """Orquestador central del pipeline de traducción de imágenes.

    Coordina la ejecución secuencial de los módulos OCR y Traducción,
    gestiona el estado de la sesión y registra tiempos y errores.

    Attributes:
        ocr_reader: Instancia del módulo OCR (ModuloOCR).
        translation_models: Caché de modelos de traducción {par_idiomas: pipeline}.
        target_language: Idioma destino seleccionado por el usuario.
        confidence_threshold: Umbral mínimo de confianza OCR (default 0.3).
        use_gpu: Indica si se usa aceleración GPU.
    """

    def __init__(self, target_language: str, use_gpu: bool = False) -> None:
        """Inicializa el orquestador con los módulos OCR y Traducción.

        Args:
            target_language: Código ISO 639-1 del idioma destino.
            use_gpu: Activar aceleración GPU si está disponible.
        """
        self.target_language = target_language
        self.use_gpu = use_gpu or torch.cuda.is_available()
        self.confidence_threshold: float = 0.3
        self._imagen_cargada: bool = False
        self._imagen_actual: Optional[np.ndarray] = None

        # Inicializar módulos
        from src.translation_module import IDIOMAS_SOPORTADOS
        all_langs = [v["easyocr"] for v in IDIOMAS_SOPORTADOS.values()]
        self.ocr_reader = ModuloOCR(
            languages=all_langs,
            use_gpu=self.use_gpu,
            confidence_threshold=self.confidence_threshold,
        )
        self.modulo_traduccion = ModuloTraduccion(use_gpu=self.use_gpu)
        self.translation_models: dict = self.modulo_traduccion._model_cache

    def cargar_imagen(self, image_path: str) -> np.ndarray:
        """Carga una imagen desde disco y la retorna como np.ndarray.

        Args:
            image_path: Ruta al archivo de imagen (JPEG, PNG o BMP).

        Returns:
            Imagen cargada como np.ndarray en formato BGR.

        Raises:
            FileNotFoundError: Si el archivo no existe en la ruta indicada.
            ValueError: Si el archivo no puede leerse como imagen válida.
        """
        raise NotImplementedError

    def detectar_texto(self, image: np.ndarray) -> list:
        """Detecta regiones de texto en la imagen usando el módulo OCR.

        Args:
            image: Imagen como np.ndarray (debe haberse cargado previamente
                   con cargar_imagen).

        Returns:
            Lista de dicts con campos 'bbox', 'texto' y 'confianza'.

        Raises:
            RuntimeError: Si cargar_imagen no fue invocado previamente.
        """
        raise NotImplementedError

    def traducir(self, text: str, source_lang: str) -> str:
        """Traduce el texto del idioma origen al idioma destino configurado.

        Args:
            text: Texto a traducir.
            source_lang: Código ISO 639-1 del idioma origen.

        Returns:
            Texto traducido al idioma destino.

        Raises:
            RuntimeError: Si cargar_imagen no fue invocado previamente.
        """
        raise NotImplementedError

    def procesar(self, image_path: str) -> dict:
        """Ejecuta el pipeline completo sobre la imagen indicada.

        Orden de ejecución: cargar_imagen → detectar_texto → traducir →
        construcción de ResultadoTraduccion. Registra el tiempo en ms de cada
        etapa. Captura cualquier excepción, la registra en log e incluye la
        descripción del error en el ResultadoTraduccion retornado.

        Args:
            image_path: Ruta al archivo de imagen a procesar.

        Returns:
            Diccionario con los campos de ResultadoTraduccion:
            texto_original, idioma_origen, texto_traducido, idioma_destino,
            regiones, error, tiempos_ms.
        """
        raise NotImplementedError
