"""
conftest.py – Configuración global de pytest para el proyecto.

Garantiza que el stub de easyocr inyectado por test_ocr_module.py no
contamine los tests de integración que necesitan el módulo real.
"""

import sys
import importlib
import pytest


@pytest.fixture(autouse=True)
def restore_easyocr_for_integration(request):
    """Restaura el módulo easyocr real antes de cada test de integración.

    test_ocr_module.py inyecta un stub de easyocr en sys.modules para evitar
    instalar el paquete real durante los tests unitarios. Sin embargo, cuando
    pytest recolecta todos los tests juntos, ese stub persiste y rompe los
    tests de integración que necesitan el easyocr real.

    Este fixture detecta si el test actual está marcado como 'integration' y,
    en ese caso, elimina el stub de sys.modules y los módulos src que lo
    importaron, para forzar su reimportación con el módulo real.
    """
    is_integration = request.node.get_closest_marker("integration") is not None
    if not is_integration:
        yield
        return

    # Verificar si easyocr en sys.modules es un stub (sin __file__)
    easyocr_mod = sys.modules.get("easyocr")
    stub_active = easyocr_mod is not None and not hasattr(easyocr_mod, "__file__")

    removed_modules = {}

    if stub_active:
        # Guardar y eliminar el stub de easyocr
        removed_modules["easyocr"] = sys.modules.pop("easyocr")

        # Eliminar submódulos de easyocr
        for k in list(sys.modules.keys()):
            if k.startswith("easyocr."):
                removed_modules[k] = sys.modules.pop(k)

        # Eliminar src.ocr_module y src.orchestrator para forzar reimportación
        # con el easyocr real
        for mod_name in ("src.ocr_module", "src.orchestrator"):
            if mod_name in sys.modules:
                removed_modules[mod_name] = sys.modules.pop(mod_name)

        # Reimportar el easyocr real y los módulos src que dependen de él
        try:
            importlib.import_module("easyocr")
            importlib.import_module("src.ocr_module")
            importlib.import_module("src.orchestrator")
        except ImportError:
            # Si easyocr real no está instalado, restaurar el stub
            for k, v in removed_modules.items():
                sys.modules[k] = v
            yield
            return

        # Actualizar las referencias en el módulo test_orchestrator ya importado
        # para que use las clases recién reimportadas
        test_orch_mod = sys.modules.get("tests.test_orchestrator")
        if test_orch_mod is not None:
            src_orch = sys.modules.get("src.orchestrator")
            src_ocr = sys.modules.get("src.ocr_module")
            if src_orch is not None:
                test_orch_mod.TraductorDeImagenes = src_orch.TraductorDeImagenes
                test_orch_mod.ResultadoTraduccion = src_orch.ResultadoTraduccion
            if src_ocr is not None:
                test_orch_mod.RegionTexto = src_ocr.RegionTexto

    yield

    # Restaurar los módulos eliminados para que los tests posteriores
    # (si los hubiera) puedan seguir usando el stub
    if stub_active:
        # Eliminar los módulos reales que cargamos para la integración
        for mod_name in ("src.ocr_module", "src.orchestrator"):
            if mod_name in sys.modules:
                sys.modules.pop(mod_name)
        # Eliminar easyocr real y sus submódulos
        for k in list(sys.modules.keys()):
            if k == "easyocr" or k.startswith("easyocr."):
                sys.modules.pop(k)
        # Restaurar los módulos originales (stub)
        for k, v in removed_modules.items():
            if k not in sys.modules:
                sys.modules[k] = v
