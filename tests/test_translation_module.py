"""
Tests para src/translation_module.py

Cubre:
- ModuloTraduccion.traducir() con mocks de transformers
- Validación de texto vacío/espacios en blanco
- Validación de longitud máxima (50.000 caracteres)
- Caso de mismo idioma (identidad)
- Validación de pares de idiomas soportados
- Caché de modelos
- Pruebas basadas en propiedades con Hypothesis (P3, P4, P5)
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub de transformers para evitar instalar el paquete real en el entorno de tests.
# Se inyecta ANTES de importar src.translation_module.
# ---------------------------------------------------------------------------
_transformers_stub = types.ModuleType("transformers")
_transformers_stub.pipeline = MagicMock  # pipeline es una función; MagicMock la simula
sys.modules.setdefault("transformers", _transformers_stub)

_torch_stub = types.ModuleType("torch")
_torch_stub.cuda = MagicMock()
_torch_stub.cuda.is_available = MagicMock(return_value=False)
sys.modules.setdefault("torch", _torch_stub)

import pytest

# Try to import hypothesis, but skip if not available
try:
    from hypothesis import given, settings, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators for when hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    HealthCheck = type('HealthCheck', (), {'too_slow': 'too_slow', 'filter_too_much': 'filter_too_much'})
    
    class st:
        @staticmethod
        def text(**kwargs):
            return None
        
        @staticmethod
        def sampled_from(items):
            return None
        
        @staticmethod
        def integers(**kwargs):
            return None
        
        @staticmethod
        def floats(**kwargs):
            return None
        
        @staticmethod
        def lists(**kwargs):
            return None
        
        @staticmethod
        def characters(**kwargs):
            return None
        
        @staticmethod
        def composite(**kwargs):
            def decorator(func):
                return func
            return decorator

from src.translation_module import ModuloTraduccion, IDIOMAS_SOPORTADOS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_translation_module_with_mock(use_gpu=False):
    """Crea un ModuloTraduccion con _cargar_modelo mockeado."""
    modulo = ModuloTraduccion(use_gpu=use_gpu)
    
    # Crear un mock de pipeline que sea callable
    mock_pipeline = MagicMock()
    mock_pipeline.model = MagicMock()
    mock_pipeline.model.config = MagicMock()
    mock_pipeline.model.config._name_or_path = "Helsinki-NLP/opus-mt-en-es"
    mock_pipeline.tokenizer = MagicMock()
    mock_pipeline.tokenizer.lang_code_to_id = {"es_XX": 250004}
    
    # Mockear _cargar_modelo directamente en la instancia
    mock_cargar_modelo = MagicMock(return_value=mock_pipeline)
    modulo._cargar_modelo = mock_cargar_modelo
    
    return modulo, mock_cargar_modelo, mock_pipeline


def get_iso_codes():
    """Retorna lista de códigos ISO soportados."""
    return [info["iso"] for info in IDIOMAS_SOPORTADOS.values()]


# ---------------------------------------------------------------------------
# Tests unitarios – ModuloTraduccion.traducir()
# ---------------------------------------------------------------------------

class TestModuloTraduccion:
    """Pruebas de la clase ModuloTraduccion."""

    def test_empty_text_raises_exception(self):
        """Texto vacío debe lanzar ValueError con mensaje exacto."""
        modulo, _, _ = make_translation_module_with_mock()
        with pytest.raises(ValueError, match="El texto de entrada para traducción está vacío"):
            modulo.traducir("", "en", "es")

    def test_whitespace_only_text_raises_exception(self):
        """Texto con solo espacios en blanco debe lanzar ValueError."""
        modulo, _, _ = make_translation_module_with_mock()
        with pytest.raises(ValueError, match="El texto de entrada para traducción está vacío"):
            modulo.traducir("   \t\n  ", "en", "es")

    def test_text_exceeds_50000_chars_raises_exception(self):
        """Texto > 50.000 caracteres debe lanzar ValueError."""
        modulo, _, _ = make_translation_module_with_mock()
        long_text = "a" * 50001
        with pytest.raises(ValueError, match="El texto supera el límite de 50.000 caracteres permitido para traducción"):
            modulo.traducir(long_text, "en", "es")

    def test_text_exactly_50000_chars_does_not_raise(self):
        """Texto de exactamente 50.000 caracteres debe ser aceptado."""
        modulo, mock_cargar_modelo, mock_pipeline = make_translation_module_with_mock()
        long_text = "a" * 50000
        # Mockear la traducción para que no falle
        mock_pipeline.return_value = [{"translation_text": "translated"}]
        # Esto debería funcionar sin excepción
        result = modulo.traducir(long_text, "en", "es")
        assert result == "translated"

    def test_same_source_target_language_returns_original(self):
        """Mismo idioma origen y destino debe retornar texto sin modificar."""
        modulo, mock_cargar_modelo, _ = make_translation_module_with_mock()
        texto = "Hello world"
        result = modulo.traducir(texto, "en", "en")
        assert result == texto
        # No debe llamar a _cargar_modelo
        mock_cargar_modelo.assert_not_called()

    def test_unsupported_source_language_raises_exception(self):
        """Idioma origen no soportado debe lanzar ValueError."""
        modulo, _, _ = make_translation_module_with_mock()
        with pytest.raises(ValueError, match="Par de idiomas no soportado: xx → es"):
            modulo.traducir("texto", "xx", "es")

    def test_unsupported_target_language_raises_exception(self):
        """Idioma destino no soportado debe lanzar ValueError."""
        modulo, _, _ = make_translation_module_with_mock()
        with pytest.raises(ValueError, match="Par de idiomas no soportado: en → xx"):
            modulo.traducir("texto", "en", "xx")

    def test_supported_language_pair_calls_pipeline(self):
        """Par de idiomas soportado debe invocar pipeline."""
        modulo, mock_cargar_modelo, mock_pipeline = make_translation_module_with_mock()
        texto = "Hello world"
        mock_pipeline.return_value = [{"translation_text": "Hola mundo"}]
        
        result = modulo.traducir(texto, "en", "es")
        
        assert result == "Hola mundo"
        # _cargar_modelo debe haberse llamado una vez
        mock_cargar_modelo.assert_called_once_with("en", "es")

    def test_model_cache_reuses_pipeline(self):
        """Segunda llamada para mismo par debe reutilizar pipeline en caché."""
        modulo, mock_cargar_modelo, mock_pipeline = make_translation_module_with_mock()
        texto = "Hello world"
        mock_pipeline.return_value = [{"translation_text": "Hola mundo"}]
        
        # Primera llamada
        result1 = modulo.traducir(texto, "en", "es")
        assert result1 == "Hola mundo"
        assert mock_cargar_modelo.call_count == 1
        
        # Segunda llamada
        result2 = modulo.traducir(texto, "en", "es")
        assert result2 == "Hola mundo"
        # _cargar_modelo no debe llamarse de nuevo
        assert mock_cargar_modelo.call_count == 1
        
        # Verificar que el modelo está en caché
        cache_key = ("en", "es")
        assert cache_key in modulo._model_cache

    def test_different_language_pair_loads_new_model(self):
        """Par de idiomas diferente debe cargar nuevo modelo."""
        modulo, mock_cargar_modelo, mock_pipeline = make_translation_module_with_mock()
        texto = "Hello world"
        mock_pipeline.return_value = [{"translation_text": "Hola mundo"}]
        
        # Primera llamada para en→es
        result1 = modulo.traducir(texto, "en", "es")
        assert result1 == "Hola mundo"
        assert mock_cargar_modelo.call_count == 1
        
        # Resetear el mock para contar llamadas
        mock_cargar_modelo.reset_mock()
        mock_pipeline.return_value = [{"translation_text": "Bonjour le monde"}]
        
        # Segunda llamada para en→fr (diferente par)
        result2 = modulo.traducir(texto, "en", "fr")
        assert result2 == "Bonjour le monde"
        # _cargar_modelo debe llamarse de nuevo
        assert mock_cargar_modelo.call_count == 1

    def test_use_gpu_flag_passed_to_pipeline(self):
        """Flag use_gpu debe pasarse a pipeline."""
        with patch("src.translation_module.pipeline") as MockPipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.return_value = [{"translation_text": "translated"}]
            mock_pipeline_instance.model = MagicMock()
            mock_pipeline_instance.model.config = MagicMock()
            mock_pipeline_instance.model.config._name_or_path = "Helsinki-NLP/opus-mt-en-es"
            mock_pipeline_instance.tokenizer = MagicMock()
            mock_pipeline_instance.tokenizer.lang_code_to_id = {"es_XX": 250004}
            MockPipeline.return_value = mock_pipeline_instance
            
            # Con use_gpu=True
            modulo = ModuloTraduccion(use_gpu=True)
            # Mockear torch.cuda.is_available para que devuelva True
            with patch("src.translation_module.torch.cuda.is_available", return_value=True):
                modulo.traducir("text", "en", "es")
            
            # Verificar que se pasó device=0 (GPU)
            MockPipeline.assert_called_once()
            call_kwargs = MockPipeline.call_args[1]
            assert call_kwargs.get("device") == 0

    def test_use_gpu_false_passes_device_minus_one(self):
        """use_gpu=False debe pasar device=-1 a pipeline."""
        with patch("src.translation_module.pipeline") as MockPipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.return_value = [{"translation_text": "translated"}]
            mock_pipeline_instance.model = MagicMock()
            mock_pipeline_instance.model.config = MagicMock()
            mock_pipeline_instance.model.config._name_or_path = "Helsinki-NLP/opus-mt-en-es"
            mock_pipeline_instance.tokenizer = MagicMock()
            mock_pipeline_instance.tokenizer.lang_code_to_id = {"es_XX": 250004}
            MockPipeline.return_value = mock_pipeline_instance
            
            # Con use_gpu=False
            modulo = ModuloTraduccion(use_gpu=False)
            modulo.traducir("text", "en", "es")
            
            # Verificar que se pasó device=-1 (CPU)
            MockPipeline.assert_called_once()
            call_kwargs = MockPipeline.call_args[1]
            assert call_kwargs.get("device") == -1


# ---------------------------------------------------------------------------
# Pruebas basadas en propiedades – Hypothesis
# ---------------------------------------------------------------------------

# Estrategia para generar texto no vacío
texto_strategy = st.text(
    min_size=1,
    max_size=1000,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
)

# Estrategia para generar texto con solo espacios en blanco
whitespace_text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Zs", "Cc")),  # Espacios y controles
    min_size=0,
    max_size=100,
).filter(lambda s: s.strip() == "")

# Estrategia para generar códigos ISO soportados
iso_code_strategy = st.sampled_from(get_iso_codes())


# ---------------------------------------------------------------------------
# P3: Identidad en traducción mismo idioma
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    texto=texto_strategy,
    idioma=iso_code_strategy,
)
def test_p3_identity_same_language(texto, idioma):
    """
    **Validates: Requirements 4.3**
    
    P3: Para cualquier texto no vacío y cualquier idioma soportado,
    cuando idioma_origen == idioma_destino, el texto retornado debe ser
    idéntico al texto de entrada, sin invocar ningún modelo de traducción.
    """
    # Usamos un mock para verificar que no se llama a pipeline
    with patch("transformers.pipeline") as MockPipeline:
        modulo = ModuloTraduccion(use_gpu=False)
        result = modulo.traducir(texto, idioma, idioma)
        
        # El resultado debe ser idéntico al texto original
        assert result == texto
        
        # No debe haberse llamado a pipeline
        MockPipeline.assert_not_called()


# ---------------------------------------------------------------------------
# P4: Caché de modelos reduce tiempo de traducción
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    texto=st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
    src_lang=iso_code_strategy,
    tgt_lang=iso_code_strategy,
)
def test_p4_cache_reduces_translation_time(texto, src_lang, tgt_lang):
    """
    **Validates: Requirements 4.5**
    
    P4: Para cualquier par de idiomas (origen, destino) y cualquier texto
    de hasta 5.000 caracteres, la segunda invocación debe reutilizar el
    modelo ya cargado en caché.
    
    Nota: No medimos tiempo real en tests, pero verificamos que el modelo
    se cachea y se reutiliza.
    """
    # Saltar si es el mismo idioma (ya probado en P3)
    if src_lang == tgt_lang:
        return
    
    with patch("src.translation_module.pipeline") as MockPipeline:
        mock_pipeline_instance = MagicMock()
        mock_pipeline_instance.return_value = [{"translation_text": "translated"}]
        mock_pipeline_instance.model = MagicMock()
        mock_pipeline_instance.model.config = MagicMock()
        mock_pipeline_instance.model.config._name_or_path = "Helsinki-NLP/opus-mt-en-es"
        mock_pipeline_instance.tokenizer = MagicMock()
        mock_pipeline_instance.tokenizer.lang_code_to_id = {"es_XX": 250004}
        MockPipeline.return_value = mock_pipeline_instance
        
        modulo = ModuloTraduccion(use_gpu=False)
        
        # Primera invocación
        result1 = modulo.traducir(texto, src_lang, tgt_lang)
        assert result1 == "translated"
        assert MockPipeline.call_count == 1
        
        # Segunda invocación
        result2 = modulo.traducir(texto, src_lang, tgt_lang)
        assert result2 == "translated"
        
        # Pipeline no debe llamarse de nuevo (mismo call_count)
        assert MockPipeline.call_count == 1
        
        # Verificar que el modelo está en caché
        cache_key = (src_lang, tgt_lang)
        assert cache_key in modulo._model_cache


# ---------------------------------------------------------------------------
# P5: Rechazo de texto vacío o solo espacios en blanco
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    texto=whitespace_text_strategy,
    src_lang=iso_code_strategy,
    tgt_lang=iso_code_strategy,
)
def test_p5_rejects_empty_or_whitespace_text(texto, src_lang, tgt_lang):
    """
    **Validates: Requirements 4.6**
    
    P5: Para cualquier cadena compuesta únicamente de espacios en blanco
    (incluyendo cadena vacía, tabulaciones y saltos de línea), el módulo
    debe lanzar una excepción con el mensaje exacto y no invocar ningún modelo.
    """
    with patch("transformers.pipeline") as MockPipeline:
        modulo = ModuloTraduccion(use_gpu=False)
        
        with pytest.raises(ValueError, match="El texto de entrada para traducción está vacío"):
            modulo.traducir(texto, src_lang, tgt_lang)
        
        # No debe haberse llamado a pipeline
        MockPipeline.assert_not_called()


# ---------------------------------------------------------------------------
# Propiedad adicional: Validación de longitud máxima
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    length=st.integers(min_value=50001, max_value=100000),
    src_lang=iso_code_strategy,
    tgt_lang=iso_code_strategy,
)
def test_property_text_length_validation(length, src_lang, tgt_lang):
    """
    Propiedad: Para cualquier texto que supere los 50.000 caracteres,
    el módulo debe lanzar una excepción con el mensaje exacto.
    """
    # Saltar si es el mismo idioma
    if src_lang == tgt_lang:
        return
    
    texto = "a" * length
    with patch("transformers.pipeline") as MockPipeline:
        modulo = ModuloTraduccion(use_gpu=False)
        
        with pytest.raises(ValueError, match="El texto supera el límite de 50.000 caracteres permitido para traducción"):
            modulo.traducir(texto, src_lang, tgt_lang)
        
        # No debe haberse llamado a pipeline
        MockPipeline.assert_not_called()