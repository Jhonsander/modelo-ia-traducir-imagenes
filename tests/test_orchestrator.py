"""
Pruebas básicas para verificar la implementación de TraductorDeImagenes.
"""

import pytest
import numpy as np
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
