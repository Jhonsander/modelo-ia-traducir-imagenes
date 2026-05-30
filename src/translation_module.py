"""
Módulo de Traducción – Traducción de texto mediante modelos Hugging Face Transformers.

Responsabilidades:
- Seleccionar el modelo de traducción más adecuado para un par de idiomas
  (Helsinki-NLP/opus-mt-{src}-{tgt} o facebook/mbart-large-50-many-to-many-mmt).
- Cargar y cachear los modelos para evitar recargas innecesarias.
- Validar el texto de entrada (no vacío, no supera 50 000 caracteres).
- Retornar el texto sin modificar cuando idioma_origen == idioma_destino.
- Lanzar excepciones con mensajes exactos definidos en los requisitos.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mapeo de idiomas soportados
# ---------------------------------------------------------------------------

IDIOMAS_SOPORTADOS: Dict[str, Dict[str, str]] = {
    "Español":            {"easyocr": "es", "iso": "es", "mbart": "es_XX"},
    "Inglés":             {"easyocr": "en", "iso": "en", "mbart": "en_XX"},
    "Francés":            {"easyocr": "fr", "iso": "fr", "mbart": "fr_XX"},
    "Alemán":             {"easyocr": "de", "iso": "de", "mbart": "de_DE"},
    "Portugués":          {"easyocr": "pt", "iso": "pt", "mbart": "pt_XX"},
    "Italiano":           {"easyocr": "it", "iso": "it", "mbart": "it_IT"},
    "Chino simplificado": {"easyocr": "ch_sim", "iso": "zh", "mbart": "zh_CN"},
    "Japonés":            {"easyocr": "ja",     "iso": "ja", "mbart": "ja_XX"},
    "Coreano":            {"easyocr": "ko",     "iso": "ko", "mbart": "ko_KR"},
}

# Lookup: ISO code -> mbart code
_ISO_TO_MBART: Dict[str, str] = {
    info["iso"]: info["mbart"] for info in IDIOMAS_SOPORTADOS.values()
}

# Set of all supported ISO codes
_ISO_SOPORTADOS = {info["iso"] for info in IDIOMAS_SOPORTADOS.values()}

# Fallback multilingual model
_MBART_MODEL = "facebook/mbart-large-50-many-to-many-mmt"


# ---------------------------------------------------------------------------
# ModuloTraduccion
# ---------------------------------------------------------------------------

class ModuloTraduccion:
    """Encapsula la carga y uso de modelos Hugging Face para traducción de texto.

    Mantiene un caché interno de pipelines indexado por par de idiomas para
    evitar recargas en invocaciones sucesivas.

    Args:
        use_gpu: Si es True, usa aceleración GPU cuando esté disponible.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        """Inicializa el módulo de traducción.

        Args:
            use_gpu: Activar aceleración GPU para los modelos de traducción.
        """
        self.use_gpu = use_gpu
        self._model_cache: dict = {}  # {(src_lang, tgt_lang): pipeline}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def traducir(
        self,
        texto: str,
        idioma_origen: str,
        idioma_destino: str,
    ) -> str:
        """Traduce el texto del idioma origen al idioma destino.

        Valida el texto de entrada, gestiona el caso de mismo idioma y reutiliza
        el modelo en caché si ya fue cargado para el par solicitado.

        Args:
            texto: Texto a traducir.
            idioma_origen: Código ISO 639-1 del idioma origen (p. ej. "en").
            idioma_destino: Código ISO 639-1 del idioma destino (p. ej. "es").

        Returns:
            Texto traducido al idioma destino.

        Raises:
            ValueError: Si el texto está vacío o es solo espacios en blanco.
            ValueError: Si el texto supera los 50 000 caracteres.
            ValueError: Si el par de idiomas no está soportado.
        """
        # 1. Validar texto vacío
        if texto.strip() == "":
            raise ValueError("El texto de entrada para traducción está vacío")

        # 2. Validar longitud máxima
        if len(texto) > 50_000:
            raise ValueError(
                "El texto supera el límite de 50.000 caracteres permitido para traducción"
            )

        # 3. Mismo idioma → devolver sin modificar
        if idioma_origen == idioma_destino:
            return texto

        # 4. Verificar que ambos códigos ISO están soportados
        if idioma_origen not in _ISO_SOPORTADOS or idioma_destino not in _ISO_SOPORTADOS:
            raise ValueError(
                f"Par de idiomas no soportado: {idioma_origen} → {idioma_destino}"
            )

        # 5. Obtener o cargar el pipeline
        clave = (idioma_origen, idioma_destino)
        if clave in self._model_cache:
            traductor = self._model_cache[clave]
        else:
            traductor = self._cargar_modelo(idioma_origen, idioma_destino)
            self._model_cache[clave] = traductor

        # 6. Ejecutar la traducción
        # Detect if this is an mbart pipeline by checking the model config
        es_mbart = False
        try:
            # Try to get the model name from the pipeline
            model_obj = getattr(traductor, "model", None)
            if model_obj is not None:
                config_obj = getattr(model_obj, "config", None)
                if config_obj is not None:
                    model_name = getattr(config_obj, "_name_or_path", "")
                    es_mbart = _MBART_MODEL in (model_name or "")
        except AttributeError:
            # If we can't access the model config, assume it's not mbart
            es_mbart = False

        if es_mbart:
            mbart_tgt = _ISO_TO_MBART[idioma_destino]
            forced_bos_token_id = traductor.tokenizer.lang_code_to_id[mbart_tgt]
            resultado = traductor(texto, forced_bos_token_id=forced_bos_token_id)
        else:
            resultado = traductor(texto)

        # resultado should be a list from the pipeline
        return resultado[0]["translation_text"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _seleccionar_modelo(self, idioma_origen: str, idioma_destino: str) -> str:
        """Selecciona el identificador del modelo de Hugging Face más adecuado.

        Intenta primero Helsinki-NLP/opus-mt-{src}-{tgt}; si no existe en el
        Hub, usa facebook/mbart-large-50-many-to-many-mmt.

        Args:
            idioma_origen: Código ISO 639-1 del idioma origen.
            idioma_destino: Código ISO 639-1 del idioma destino.

        Returns:
            Identificador del modelo en Hugging Face Hub.
        """
        opus_model = f"Helsinki-NLP/opus-mt-{idioma_origen}-{idioma_destino}"
        try:
            from huggingface_hub import model_info
            from huggingface_hub.utils import RepositoryNotFoundError
            model_info(opus_model)
            logger.info("Modelo seleccionado (opus-mt): %s", opus_model)
            return opus_model
        except Exception as exc:
            # Covers RepositoryNotFoundError and any network/auth issues
            logger.info(
                "Modelo opus-mt no disponible (%s: %s). Usando mbart.",
                type(exc).__name__,
                exc,
            )
            return _MBART_MODEL

    def _cargar_modelo(self, idioma_origen: str, idioma_destino: str):
        """Carga el pipeline de traducción y lo almacena en el caché.

        Selecciona el modelo mediante ``_seleccionar_modelo()``, construye el
        pipeline de Hugging Face con la configuración de dispositivo adecuada y
        lo guarda en ``_model_cache`` para reutilización futura.

        Args:
            idioma_origen: Código ISO 639-1 del idioma origen.
            idioma_destino: Código ISO 639-1 del idioma destino.

        Returns:
            Pipeline de traducción de Hugging Face listo para usar.
        """
        modelo_nombre = self._seleccionar_modelo(idioma_origen, idioma_destino)
        device = 0 if (self.use_gpu and torch.cuda.is_available()) else -1

        logger.info(
            "Cargando modelo '%s' para %s→%s (device=%d).",
            modelo_nombre,
            idioma_origen,
            idioma_destino,
            device,
        )

        if modelo_nombre == _MBART_MODEL:
            mbart_src = _ISO_TO_MBART[idioma_origen]
            traductor = pipeline(
                "translation",
                model=modelo_nombre,
                src_lang=mbart_src,
                tgt_lang=_ISO_TO_MBART[idioma_destino],
                device=device,
            )
        else:
            traductor = pipeline(
                "translation",
                model=modelo_nombre,
                device=device,
            )

        clave = (idioma_origen, idioma_destino)
        self._model_cache[clave] = traductor
        logger.info("Modelo cargado y almacenado en caché para clave %s.", clave)
        return traductor
