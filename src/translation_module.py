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
# Translation pipeline factory
# ---------------------------------------------------------------------------

class _TranslationPipeline:
    """Pipeline de traducción que encapsula un modelo Marian o mBART.

    Actúa como callable: ``pipe(texto)`` retorna ``[{"translation_text": ...}]``.
    Expone los atributos ``model`` y ``tokenizer`` para compatibilidad con los
    tests que inspeccionan el pipeline.
    """

    def __init__(self, model_name: str, device: int = -1):
        """Carga el modelo y el tokenizer.

        Args:
            model_name: Identificador del modelo en Hugging Face Hub.
            device: -1 para CPU, 0 para GPU.
        """
        self._model_name = model_name
        self._device_id = device
        torch_device = torch.device("cuda" if device >= 0 else "cpu")

        if model_name == _MBART_MODEL:
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
            self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
            self.model = MBartForConditionalGeneration.from_pretrained(model_name).to(torch_device)
        else:
            from transformers import MarianMTModel, MarianTokenizer
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(torch_device)

        self._torch_device = torch_device

    def __call__(self, texto: str, **kwargs) -> list:
        """Traduce el texto y retorna el resultado en formato pipeline.

        Args:
            texto: Texto a traducir.
            **kwargs: Argumentos adicionales (p. ej. forced_bos_token_id para mBART).

        Returns:
            Lista con un dict ``{"translation_text": texto_traducido}``.
        """
        if self._model_name == _MBART_MODEL:
            inputs = self.tokenizer(texto, return_tensors="pt", padding=True).to(self._torch_device)
            forced_bos_token_id = kwargs.get("forced_bos_token_id")
            if forced_bos_token_id is not None:
                translated = self.model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
            else:
                translated = self.model.generate(**inputs)
        else:
            inputs = self.tokenizer([texto], return_tensors="pt", padding=True).to(self._torch_device)
            translated = self.model.generate(**inputs)

        texto_traducido = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return [{"translation_text": texto_traducido}]


def pipeline(task: str = "translation", model: str = "", device: int = -1, **kwargs) -> _TranslationPipeline:
    """Crea un pipeline de traducción compatible con la interfaz de Hugging Face.

    Esta función reemplaza ``transformers.pipeline`` para la tarea de traducción,
    usando directamente los modelos Marian o mBART sin depender de la API de
    ``pipeline`` de transformers (que eliminó el soporte para la tarea
    ``"translation"`` en versiones recientes).

    Args:
        task: Tarea del pipeline (ignorada; siempre se usa traducción).
        model: Identificador del modelo en Hugging Face Hub.
        device: -1 para CPU, 0 para GPU.
        **kwargs: Argumentos adicionales (ignorados).

    Returns:
        Instancia de ``_TranslationPipeline`` lista para usar.
    """
    return _TranslationPipeline(model_name=model, device=device)


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
        self._model_cache: dict = {}  # {(src_lang, tgt_lang): pipeline_instance}

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

        # 5. Obtener o cargar el modelo (pipeline)
        clave = (idioma_origen, idioma_destino)
        if clave not in self._model_cache:
            pipe = self._cargar_modelo(idioma_origen, idioma_destino)
            self._model_cache[clave] = pipe

        pipe = self._model_cache[clave]

        # 6. Ejecutar la traducción
        # Detect if this is an mbart pipeline by checking the model name
        model_name = ""
        try:
            model_name = pipe.model.config._name_or_path
        except Exception:
            pass

        if _MBART_MODEL in model_name:
            # mbart pipeline: set src_lang and forced_bos_token_id
            try:
                pipe.tokenizer.src_lang = _ISO_TO_MBART[idioma_origen]
                forced_bos_token_id = pipe.tokenizer.lang_code_to_id[
                    _ISO_TO_MBART[idioma_destino]
                ]
                result = pipe(texto, forced_bos_token_id=forced_bos_token_id)
            except Exception:
                result = pipe(texto)
        else:
            result = pipe(texto)

        if isinstance(result, list) and len(result) > 0:
            return result[0].get("translation_text", texto)
        return texto

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
            model_info(opus_model)
            logger.info("Modelo seleccionado (opus-mt): %s", opus_model)
            return opus_model
        except Exception as exc:
            logger.info(
                "Modelo opus-mt no disponible (%s: %s). Usando mbart.",
                type(exc).__name__,
                exc,
            )
            return _MBART_MODEL

    def _cargar_modelo(self, idioma_origen: str, idioma_destino: str):
        """Carga el pipeline de traducción y lo retorna.

        Selecciona el modelo mediante ``_seleccionar_modelo()``, crea el
        pipeline de traducción con la configuración de dispositivo adecuada
        y lo retorna para que el llamador lo almacene en caché.

        Args:
            idioma_origen: Código ISO 639-1 del idioma origen.
            idioma_destino: Código ISO 639-1 del idioma destino.

        Returns:
            Pipeline de traducción (callable).
        """
        modelo_nombre = self._seleccionar_modelo(idioma_origen, idioma_destino)
        device = 0 if (self.use_gpu and torch.cuda.is_available()) else -1

        logger.info(
            "Cargando modelo '%s' para %s→%s (device=%s).",
            modelo_nombre,
            idioma_origen,
            idioma_destino,
            device,
        )

        pipe = pipeline(
            "translation",
            model=modelo_nombre,
            device=device,
        )

        logger.info("Pipeline cargado para %s→%s.", idioma_origen, idioma_destino)
        return pipe
