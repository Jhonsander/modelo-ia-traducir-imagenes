"""
Módulo OCR – Detección y extracción de texto en imágenes mediante EasyOCR.

Responsabilidades:
- Detectar regiones de texto en una imagen (np.ndarray) usando EasyOCR.
- Filtrar regiones por umbral de confianza (default 0.3).
- Consolidar el texto de todas las regiones en un único bloque ordenado
  espacialmente (arriba-abajo, izquierda-derecha).
- Propagar excepciones con prefijo "[OCR] " para identificar la etapa.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import easyocr
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RegionTexto:
    """Región de texto detectada por OCR con su bounding box y confianza.

    Attributes:
        bbox: Lista de 4 puntos [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] que delimitan
              el rectángulo de la región.
        texto: Texto reconocido en la región.
        confianza: Valor de confianza OCR en el rango [0.0, 1.0].
    """

    bbox: list  # list[list[int]]
    texto: str
    confianza: float

    @property
    def y_min(self) -> int:
        """Coordenada Y mínima del bounding box."""
        return int(min(punto[1] for punto in self.bbox))

    @property
    def y_max(self) -> int:
        """Coordenada Y máxima del bounding box."""
        return int(max(punto[1] for punto in self.bbox))

    @property
    def x_min(self) -> int:
        """Coordenada X mínima del bounding box."""
        return int(min(punto[0] for punto in self.bbox))

    @property
    def altura(self) -> int:
        """Altura del bounding box (y_max - y_min)."""
        return self.y_max - self.y_min


# ---------------------------------------------------------------------------
# ModuloOCR
# ---------------------------------------------------------------------------

class ModuloOCR:
    """Encapsula la interacción con EasyOCR para detección y extracción de texto.

    Args:
        languages: Lista de códigos de idioma EasyOCR (p. ej. ["en", "es"]).
        use_gpu: Si es True, usa aceleración GPU cuando esté disponible.
        confidence_threshold: Umbral mínimo de confianza para incluir una región.
    """

    def __init__(
        self,
        languages: list,
        use_gpu: bool = False,
        confidence_threshold: float = 0.3,
    ) -> None:
        """Inicializa el lector EasyOCR con los idiomas y configuración dados.

        Args:
            languages: Códigos de idioma EasyOCR.
            use_gpu: Activar aceleración GPU.
            confidence_threshold: Umbral mínimo de confianza OCR.
        """
        self.languages = languages
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.reader = easyocr.Reader(languages, gpu=use_gpu)

    def detectar_regiones(self, image: np.ndarray) -> list:
        """Detecta regiones de texto en la imagen y filtra por confianza.

        Invoca ``easyocr.Reader.readtext()`` sobre la imagen, descarta las
        regiones cuya confianza sea estrictamente inferior al umbral configurado
        y retorna el resto como objetos ``RegionTexto``.

        Args:
            image: Imagen de entrada como np.ndarray (BGR o RGB).

        Returns:
            Lista de RegionTexto con confianza >= confidence_threshold.

        Raises:
            Exception: Propaga cualquier excepción con prefijo "[OCR] ".
        """
        try:
            resultados = self.reader.readtext(image)
            regiones: List[RegionTexto] = []
            for bbox, texto, confianza in resultados:
                if confianza >= self.confidence_threshold:
                    # EasyOCR devuelve bbox como lista de 4 pares [x, y]
                    bbox_normalizado = [[int(p[0]), int(p[1])] for p in bbox]
                    regiones.append(RegionTexto(
                        bbox=bbox_normalizado,
                        texto=texto,
                        confianza=float(confianza),
                    ))
            return regiones
        except Exception as exc:
            mensaje_original = str(exc)
            raise type(exc)(f"[OCR] {mensaje_original}") from exc

    def consolidar_texto(self, regiones: list) -> str:
        """Consolida el texto de todas las regiones en un único bloque ordenado.

        Ordena las regiones de arriba a abajo (y_min ascendente) y, dentro de
        la misma fila, de izquierda a derecha (x_min ascendente). Inserta '\\n'
        cuando la separación vertical entre regiones consecutivas supera el 50%
        de la altura media de las regiones de esa fila.

        Args:
            regiones: Lista de RegionTexto a consolidar.

        Returns:
            Texto consolidado como cadena. Cadena vacía si todas las regiones
            tienen texto vacío o solo espacios.
        """
        if not regiones:
            return ""

        # Descartar regiones con texto vacío o solo espacios para la comprobación
        # de si hay contenido útil, pero conservarlas para el ordenamiento.
        textos_utiles = [r.texto for r in regiones if r.texto.strip()]
        if not textos_utiles:
            return ""

        # Ordenar: primero por y_min ascendente, luego por x_min ascendente
        ordenadas = sorted(regiones, key=lambda r: (r.y_min, r.x_min))

        # Calcular la altura media global de todas las regiones (para el umbral
        # de salto de línea). Usamos las regiones con altura > 0 para evitar
        # divisiones problemáticas.
        alturas = [r.altura for r in ordenadas if r.altura > 0]
        altura_media = sum(alturas) / len(alturas) if alturas else 0.0

        partes: List[str] = []
        for i, region in enumerate(ordenadas):
            if i == 0:
                partes.append(region.texto)
                continue

            anterior = ordenadas[i - 1]
            separacion_vertical = region.y_min - anterior.y_max

            # Insertar salto de línea si la separación supera el 50% de la
            # altura media de las regiones de la fila actual.
            if altura_media > 0 and separacion_vertical > 0.5 * altura_media:
                partes.append("\n")
            else:
                partes.append(" ")

            partes.append(region.texto)

        return "".join(partes)
