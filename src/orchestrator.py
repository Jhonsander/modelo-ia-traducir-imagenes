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
        # Detectar GPU automáticamente con torch.cuda.is_available()
        self.use_gpu = use_gpu if use_gpu else torch.cuda.is_available()
        self.confidence_threshold: float = 0.3
        self._imagen_cargada: bool = False
        self._imagen_actual: Optional[np.ndarray] = None

        # Inicializar módulos
        from src.translation_module import IDIOMAS_SOPORTADOS
        # EasyOCR tiene requisitos de compatibilidad entre idiomas.
        # Según la documentación de EasyOCR, ciertos idiomas asiáticos (ch_sim, ja, ko)
        # solo son compatibles con inglés cuando se usan juntos.
        # Para soportar todos los idiomas requeridos, usamos un enfoque que incluye
        # los idiomas latinos y el inglés, que son mutuamente compatibles.
        # Los idiomas asiáticos se detectarán cuando sea necesario mediante
        # reinicialización dinámica del reader o mediante readers separados.
        # Por ahora, inicializamos con los idiomas más comunes y compatibles.
        all_langs_raw = [v["easyocr"] for v in IDIOMAS_SOPORTADOS.values()]
        
        # Separar idiomas en grupos compatibles según EasyOCR
        # Grupo 1: Idiomas latinos + inglés (mutuamente compatibles)
        latin_langs = ['en', 'es', 'fr', 'de', 'pt', 'it']
        # Grupo 2: Idiomas asiáticos (cada uno solo compatible con inglés)
        asian_langs = ['ch_sim', 'ja', 'ko']
        
        # Para la inicialización, usamos solo los idiomas latinos que son
        # mutuamente compatibles. El sistema podrá detectar texto en estos idiomas.
        # Nota: En una implementación completa, se podría usar detección de script
        # para seleccionar dinámicamente el reader apropiado (latino vs asiático).
        all_langs = [lang for lang in all_langs_raw if lang in latin_langs]
        
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
        import cv2
        import os
        
        # Verificar que el archivo existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"El archivo no existe: {image_path}")
        
        # Leer la imagen con OpenCV
        imagen = cv2.imread(image_path)
        
        # Verificar que la imagen se cargó correctamente
        if imagen is None:
            raise ValueError(f"No se pudo leer la imagen como archivo válido: {image_path}")
        
        # Registrar estado interno de imagen cargada
        self._imagen_cargada = True
        self._imagen_actual = imagen
        
        logger.info(f"Imagen cargada exitosamente: {image_path}, shape={imagen.shape}")
        
        return imagen

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
        # Verificar que cargar_imagen fue invocado
        if not self._imagen_cargada:
            raise RuntimeError(
                "Imagen no cargada. Invoque cargar_imagen antes de continuar."
            )
        
        # Invocar ModuloOCR.detectar_regiones()
        regiones = self.ocr_reader.detectar_regiones(image)
        
        # Consolidar texto
        texto_consolidado = self.ocr_reader.consolidar_texto(regiones)
        
        # Convertir lista de RegionTexto a lista de dicts
        resultado = []
        for region in regiones:
            resultado.append({
                "bbox": region.bbox,
                "texto": region.texto,
                "confianza": region.confianza,
            })
        
        logger.info(
            f"Detección de texto completada: {len(resultado)} regiones detectadas, "
            f"{len(texto_consolidado)} caracteres consolidados"
        )
        
        return resultado

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
        # Verificar que cargar_imagen fue invocado
        if not self._imagen_cargada:
            raise RuntimeError(
                "Imagen no cargada. Invoque cargar_imagen antes de continuar."
            )
        
        # Detectar idioma del texto usando utils.detectar_idioma()
        idioma_detectado = utils.detectar_idioma(text)
        logger.info(f"Idioma detectado: {idioma_detectado}")
        
        # Invocar ModuloTraduccion.traducir()
        texto_traducido = self.modulo_traduccion.traducir(
            texto=text,
            idioma_origen=idioma_detectado,
            idioma_destino=self.target_language
        )
        
        logger.info(
            f"Traducción completada: {len(text)} caracteres → {len(texto_traducido)} caracteres"
        )
        
        return texto_traducido

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
        resultado = ResultadoTraduccion()
        resultado.idioma_destino = self.target_language
        tiempos = {}
        
        try:
            # Etapa 1: Cargar imagen
            inicio_carga = utils.milisegundos_actuales()
            imagen = self.cargar_imagen(image_path)
            tiempo_carga = utils.milisegundos_actuales() - inicio_carga
            tiempos["carga"] = tiempo_carga
            logger.info(f"[TIMING] carga={tiempo_carga}ms")
            
            # Etapa 2: Detectar texto (OCR)
            inicio_ocr = utils.milisegundos_actuales()
            regiones_dict = self.detectar_texto(imagen)
            tiempo_ocr = utils.milisegundos_actuales() - inicio_ocr
            tiempos["ocr"] = tiempo_ocr
            logger.info(f"[TIMING] ocr={tiempo_ocr}ms")
            
            # Verificar si se detectaron regiones
            if not regiones_dict:
                resultado.error = "No se detectó texto en la imagen"
                resultado.tiempos_ms = tiempos
                logger.warning("[OCR] No se detectó texto en la imagen")
                return resultado.__dict__
            
            # Convertir regiones_dict a objetos RegionTexto para consolidar texto
            regiones_obj = []
            for r in regiones_dict:
                regiones_obj.append(RegionTexto(
                    bbox=r["bbox"],
                    texto=r["texto"],
                    confianza=r["confianza"]
                ))
            
            # Consolidar texto de las regiones
            texto_consolidado = self.ocr_reader.consolidar_texto(regiones_obj)
            
            # Verificar si el texto consolidado está vacío
            if not texto_consolidado or texto_consolidado.strip() == "":
                resultado.error = "No se detectó texto en la imagen"
                resultado.tiempos_ms = tiempos
                logger.warning("[OCR] Texto consolidado vacío")
                return resultado.__dict__
            
            resultado.texto_original = texto_consolidado
            resultado.regiones = regiones_dict
            
            # Etapa 3: Traducir
            inicio_traduccion = utils.milisegundos_actuales()
            
            # Detectar idioma del texto
            idioma_detectado = utils.detectar_idioma(texto_consolidado)
            resultado.idioma_origen = idioma_detectado
            logger.info(f"Idioma detectado: {idioma_detectado}")
            
            # Traducir el texto
            texto_traducido = self.modulo_traduccion.traducir(
                texto=texto_consolidado,
                idioma_origen=idioma_detectado,
                idioma_destino=self.target_language
            )
            resultado.texto_traducido = texto_traducido
            
            tiempo_traduccion = utils.milisegundos_actuales() - inicio_traduccion
            tiempos["traduccion"] = tiempo_traduccion
            logger.info(f"[TIMING] traduccion={tiempo_traduccion}ms")
            
            # Etapa 4: Construcción del ResultadoTraduccion
            inicio_construccion = utils.milisegundos_actuales()
            resultado.tiempos_ms = tiempos
            tiempo_construccion = utils.milisegundos_actuales() - inicio_construccion
            tiempos["construccion"] = tiempo_construccion
            logger.info(f"[TIMING] construccion={tiempo_construccion}ms")
            
            # Log resumen de tiempos
            logger.info(
                f"[TIMING] Pipeline completado: "
                f"carga={tiempos.get('carga', 0)}ms "
                f"ocr={tiempos.get('ocr', 0)}ms "
                f"traduccion={tiempos.get('traduccion', 0)}ms "
                f"construccion={tiempos.get('construccion', 0)}ms"
            )
            
        except Exception as exc:
            # Capturar cualquier excepción, registrar en log y poblar campo error
            tipo_excepcion = type(exc).__name__
            mensaje_error = str(exc)
            
            # Determinar la etapa donde ocurrió el error
            etapa = "desconocida"
            if "[OCR]" in mensaje_error or "OCR" in tipo_excepcion:
                etapa = "OCR"
            elif "traducción" in mensaje_error.lower() or "traduccion" in mensaje_error.lower():
                etapa = "traducción"
            elif "imagen" in mensaje_error.lower() or "cargar" in mensaje_error.lower():
                etapa = "carga"
            else:
                # Intentar inferir de la pila de llamadas o del contexto
                if not self._imagen_cargada:
                    etapa = "carga"
                elif not resultado.texto_original:
                    etapa = "OCR"
                else:
                    etapa = "traducción"
            
            # Registrar en log
            logger.error(f"[{etapa}] {tipo_excepcion}: {mensaje_error}")
            
            # Poblar campo error del resultado
            resultado.error = f"Error en {etapa}: {mensaje_error}"
            resultado.tiempos_ms = tiempos
        
        return resultado.__dict__
