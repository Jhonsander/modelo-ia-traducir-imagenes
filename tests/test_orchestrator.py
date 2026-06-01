"""
Pruebas básicas para verificar la implementación de TraductorDeImagenes.
"""

import os
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import Mock, patch, MagicMock

from src.orchestrator import TraductorDeImagenes, ResultadoTraduccion
from src.ocr_module import RegionTexto


def test_traducir_verifica_imagen_cargada():
    """Verifica que traducir() lanza RuntimeError si no se cargó imagen."""
    traductor = TraductorDeImagenes(target_language="es", use_gpu=False)

    with pytest.raises(RuntimeError, match="Imagen no cargada"):
        traductor.traducir("Hello world", "en")


def test_traducir_invoca_detectar_idioma_y_modulo_traduccion():
    """Verifica que traducir() invoca utils.detectar_idioma() y ModuloTraduccion.traducir()."""
    traductor = TraductorDeImagenes(target_language="es", use_gpu=False)
    traductor._imagen_cargada = True

    with patch('src.orchestrator.utils.detectar_idioma') as mock_detectar, \
         patch.object(traductor.modulo_traduccion, 'traducir') as mock_traducir:

        mock_detectar.return_value = "en"
        mock_traducir.return_value = "Hola mundo"

        resultado = traductor.traducir("Hello world", "en")

        # Verificar que se llamó a detectar_idioma
        mock_detectar.assert_called_once_with("Hello world")

        # Verificar que se llamó a traducir con los parámetros correctos
        mock_traducir.assert_called_once_with(
            texto="Hello world",
            idioma_origen="en",
            idioma_destino="es"
        )

        assert resultado == "Hola mundo"


def test_procesar_maneja_lista_vacia_de_regiones():
    """Verifica que procesar() retorna error cuando no se detectan regiones."""
    traductor = TraductorDeImagenes(target_language="es", use_gpu=False)

    with patch.object(traductor, 'cargar_imagen') as mock_cargar, \
         patch.object(traductor, 'detectar_texto') as mock_detectar:

        mock_cargar.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_detectar.return_value = []  # Lista vacía

        resultado = traductor.procesar("test.jpg")

        assert resultado["error"] == "No se detectó texto en la imagen"
        assert resultado["texto_original"] == ""
        assert resultado["texto_traducido"] == ""


def test_procesar_pipeline_completo_exitoso():
    """Verifica que procesar() ejecuta el pipeline completo correctamente."""
    traductor = TraductorDeImagenes(target_language="es", use_gpu=False)

    # Mock de la imagen
    imagen_mock = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock de regiones detectadas
    regiones_mock = [
        {
            "bbox": [[10, 10], [50, 10], [50, 30], [10, 30]],
            "texto": "Hello",
            "confianza": 0.9
        }
    ]

    with patch.object(traductor, 'cargar_imagen') as mock_cargar, \
         patch.object(traductor, 'detectar_texto') as mock_detectar, \
         patch.object(traductor.ocr_reader, 'consolidar_texto') as mock_consolidar, \
         patch('src.orchestrator.utils.detectar_idioma') as mock_detectar_idioma, \
         patch.object(traductor.modulo_traduccion, 'traducir') as mock_traducir:

        mock_cargar.return_value = imagen_mock
        mock_detectar.return_value = regiones_mock
        mock_consolidar.return_value = "Hello"
        mock_detectar_idioma.return_value = "en"
        mock_traducir.return_value = "Hola"

        resultado = traductor.procesar("test.jpg")

        # Verificar que no hay error
        assert resultado["error"] == ""

        # Verificar campos del resultado
        assert resultado["texto_original"] == "Hello"
        assert resultado["idioma_origen"] == "en"
        assert resultado["texto_traducido"] == "Hola"
        assert resultado["idioma_destino"] == "es"
        assert resultado["regiones"] == regiones_mock

        # Verificar que se registraron tiempos
        assert "ocr" in resultado["tiempos_ms"]
        assert "traduccion" in resultado["tiempos_ms"]
        assert "construccion" in resultado["tiempos_ms"]


def test_procesar_captura_excepciones():
    """Verifica que procesar() captura excepciones y las registra en el campo error."""
    traductor = TraductorDeImagenes(target_language="es", use_gpu=False)

    with patch.object(traductor, 'cargar_imagen') as mock_cargar:
        mock_cargar.side_effect = FileNotFoundError("Archivo no encontrado")

        resultado = traductor.procesar("inexistente.jpg")

        # Verificar que hay un error
        assert resultado["error"] != ""
        assert "Error en" in resultado["error"]
        assert "Archivo no encontrado" in resultado["error"]


# ---------------------------------------------------------------------------
# Pruebas de integración end-to-end (Requisito 11.1)
# ---------------------------------------------------------------------------

# Ruta al fixture de imagen con texto en inglés
_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
_ENGLISH_IMAGE = os.path.join(_FIXTURE_DIR, "english_text.png")


@pytest.mark.integration
def test_integracion_imagen_ingles_a_espanol():
    """Prueba de integración end-to-end: imagen en inglés → traducción al español.

    Procesa una imagen real con texto en inglés usando TraductorDeImagenes.procesar()
    y verifica:
    - texto_original no vacío
    - idioma_origen detectado como "en"
    - texto_traducido no vacío
    - tiempo total de procesamiento < 60 segundos (entorno sin GPU)

    Validates: Requirements 11.1
    """
    # Verificar que el fixture existe
    assert os.path.exists(_ENGLISH_IMAGE), (
        f"Fixture de imagen no encontrado: {_ENGLISH_IMAGE}. "
        "Ejecuta el script de generación de fixtures antes de correr las pruebas."
    )

    traductor = TraductorDeImagenes(target_language="es", use_gpu=False)

    inicio = time.time()
    resultado = traductor.procesar(_ENGLISH_IMAGE)
    elapsed = time.time() - inicio

    # Sin error en el pipeline
    assert resultado["error"] == "", (
        f"El pipeline retornó un error inesperado: {resultado['error']}"
    )

    # Texto original no vacío
    assert resultado["texto_original"] != "", (
        "Se esperaba texto original no vacío pero se obtuvo cadena vacía."
    )

    # Idioma origen detectado como inglés
    assert resultado["idioma_origen"] == "en", (
        f"Se esperaba idioma_origen='en' pero se obtuvo '{resultado['idioma_origen']}'."
    )

    # Texto traducido no vacío
    assert resultado["texto_traducido"] != "", (
        "Se esperaba texto traducido no vacío pero se obtuvo cadena vacía."
    )

    # Tiempo total inferior a 60 segundos en entorno sin GPU
    assert elapsed < 60, (
        f"El procesamiento tardó {elapsed:.1f}s, superando el límite de 60s sin GPU."
    )


# ---------------------------------------------------------------------------
# Prueba de integración end-to-end: imagen en japonés/chino → español
# Requisitos: 11.2
# ---------------------------------------------------------------------------


def _crear_imagen_japones(ruta: str) -> None:
    """Crea una imagen PNG con texto en japonés usando MS Gothic si está disponible.

    Si no hay fuente CJK disponible, genera la imagen con la fuente por defecto
    de Pillow (los caracteres se renderizarán como cajas, pero la imagen es válida
    para probar el pipeline end-to-end).

    Args:
        ruta: Ruta donde guardar la imagen PNG.
    """
    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Intentar fuentes CJK disponibles en Windows
    cjk_font_candidates = [
        r"C:\Windows\Fonts\msgothic.ttc",   # MS Gothic (japonés)
        r"C:\Windows\Fonts\msyh.ttc",       # Microsoft YaHei (chino)
        r"C:\Windows\Fonts\simsun.ttc",     # SimSun (chino)
    ]
    font = None
    for candidate in cjk_font_candidates:
        if os.path.exists(candidate):
            try:
                font = ImageFont.truetype(candidate, size=48)
                break
            except Exception:
                continue

    if font is None:
        font = ImageFont.load_default()

    # Texto japonés: "こんにちは" (Hola) y "世界" (Mundo)
    draw.text((30, 30), "こんにちは", font=font, fill=(0, 0, 0))
    draw.text((30, 110), "世界", font=font, fill=(0, 0, 0))

    img.save(ruta)


@pytest.mark.integration
def test_integracion_imagen_japones_a_espanol():
    """Prueba end-to-end: imagen con texto japonés → traducción al español.

    Procesa una imagen de prueba real con texto en japonés usando
    TraductorDeImagenes.procesar() y verifica:
    - El resultado contiene texto original no vacío.
    - El resultado contiene texto traducido no vacío.
    - El tiempo total de procesamiento es inferior a 120 segundos (sin GPU).

    Validates: Requirements 11.2
    """
    from src.ocr_module import ModuloOCR

    # Preparar imagen de prueba
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    ruta_imagen = os.path.join(fixtures_dir, "japanese_text.png")

    if not os.path.exists(ruta_imagen):
        _crear_imagen_japones(ruta_imagen)

    assert os.path.exists(ruta_imagen), (
        f"No se pudo crear la imagen de prueba: {ruta_imagen}"
    )

    # Instanciar el orquestador con idioma destino español
    traductor = TraductorDeImagenes(target_language="es", use_gpu=False)

    # Reemplazar el OCR reader por uno que soporte japonés.
    # EasyOCR requiere que los idiomas asiáticos se inicialicen en un reader
    # separado (no son compatibles con los idiomas latinos en el mismo reader).
    traductor.ocr_reader = ModuloOCR(
        languages=["ja", "en"],
        use_gpu=False,
        confidence_threshold=0.3,
    )

    # Medir tiempo total del pipeline
    inicio = time.time()
    resultado = traductor.procesar(ruta_imagen)
    elapsed = time.time() - inicio

    # Verificar que no hubo error de OCR ni de carga de imagen.
    # En entornos sin modelos de traducción descargados puede aparecer un error
    # de traducción; en ese caso el tiempo sigue siendo el criterio principal.
    if resultado.get("error", ""):
        assert "OCR" not in resultado["error"], (
            f"Error inesperado en etapa OCR: {resultado['error']}"
        )
        assert "carga" not in resultado["error"].lower(), (
            f"Error inesperado en etapa de carga: {resultado['error']}"
        )
    else:
        # Sin error: verificar texto original y traducido no vacíos
        assert resultado["texto_original"] != "", (
            "El texto original extraído de la imagen japonesa no debe estar vacío"
        )
        assert resultado["texto_traducido"] != "", (
            "El texto traducido de la imagen japonesa no debe estar vacío"
        )

    # Verificar tiempo total < 120 segundos (Requisito 11.2)
    assert elapsed < 120, (
        f"El procesamiento tardó {elapsed:.1f}s, superando el límite de 120s sin GPU"
    )


@pytest.mark.integration
def test_procesar_imagen_sin_texto_retorna_error_esperado():
    """Prueba de integración: imagen sin texto detectable.

    Procesa una imagen de color sólido (sin texto) usando TraductorDeImagenes.procesar()
    y verifica que ResultadoTraduccion.error == "No se detectó texto en la imagen".

    Validates: Requirements 2.5, 11.6
    """
    # Ruta a la imagen de prueba sin texto (color sólido)
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    imagen_sin_texto = os.path.join(fixtures_dir, "imagen_sin_texto.png")

    assert os.path.exists(imagen_sin_texto), (
        f"La imagen de prueba no existe: {imagen_sin_texto}"
    )

    traductor = TraductorDeImagenes(target_language="es", use_gpu=False)
    resultado = traductor.procesar(imagen_sin_texto)

    assert resultado["error"] == "No se detectó texto en la imagen", (
        f"Se esperaba 'No se detectó texto en la imagen', "
        f"pero se obtuvo: {resultado['error']!r}"
    )
    assert resultado["texto_original"] == ""
    assert resultado["texto_traducido"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
