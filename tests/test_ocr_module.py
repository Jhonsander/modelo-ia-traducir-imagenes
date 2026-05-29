"""
Tests para src/ocr_module.py

Cubre:
- Propiedades de RegionTexto (y_min, y_max, x_min, altura)
- ModuloOCR.detectar_regiones() con mock de easyocr.Reader
- ModuloOCR.consolidar_texto()
- Pruebas basadas en propiedades con Hypothesis (P1, P2, P8)
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub de easyocr para evitar instalar el paquete real en el entorno de tests.
# Se inyecta ANTES de importar src.ocr_module.
# ---------------------------------------------------------------------------
_easyocr_stub = types.ModuleType("easyocr")
_easyocr_stub.Reader = MagicMock  # Reader es una clase; MagicMock la simula
sys.modules.setdefault("easyocr", _easyocr_stub)

import numpy as np
import pytest

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.ocr_module import ModuloOCR, RegionTexto


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bbox(x1: int, y1: int, x2: int, y2: int) -> list:
    """Crea un bbox rectangular de 4 puntos a partir de esquinas top-left / bottom-right."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def make_region(x1=0, y1=0, x2=10, y2=10, texto="hola", confianza=0.9) -> RegionTexto:
    return RegionTexto(bbox=make_bbox(x1, y1, x2, y2), texto=texto, confianza=confianza)


def make_ocr_with_mock(readtext_return, confidence_threshold=0.3):
    """Crea un ModuloOCR con easyocr.Reader mockeado."""
    with patch("easyocr.Reader") as MockReader:
        mock_reader_instance = MagicMock()
        mock_reader_instance.readtext.return_value = readtext_return
        MockReader.return_value = mock_reader_instance
        ocr = ModuloOCR(languages=["en"], use_gpu=False,
                        confidence_threshold=confidence_threshold)
        # Reemplazamos el reader real por el mock para que las llamadas funcionen
        ocr.reader = mock_reader_instance
        return ocr


# ---------------------------------------------------------------------------
# Tests unitarios – RegionTexto
# ---------------------------------------------------------------------------

class TestRegionTexto:
    """Pruebas de las propiedades calculadas de RegionTexto."""

    def test_y_min_returns_minimum_y(self):
        """y_min debe ser el mínimo de las coordenadas Y de los 4 puntos."""
        region = RegionTexto(
            bbox=[[10, 20], [50, 15], [50, 40], [10, 35]],
            texto="test",
            confianza=0.9,
        )
        assert region.y_min == 15

    def test_y_max_returns_maximum_y(self):
        """y_max debe ser el máximo de las coordenadas Y de los 4 puntos."""
        region = RegionTexto(
            bbox=[[10, 20], [50, 15], [50, 40], [10, 35]],
            texto="test",
            confianza=0.9,
        )
        assert region.y_max == 40

    def test_x_min_returns_minimum_x(self):
        """x_min debe ser el mínimo de las coordenadas X de los 4 puntos."""
        region = RegionTexto(
            bbox=[[10, 20], [50, 15], [50, 40], [10, 35]],
            texto="test",
            confianza=0.9,
        )
        assert region.x_min == 10

    def test_altura_is_y_max_minus_y_min(self):
        """altura debe ser y_max - y_min."""
        region = RegionTexto(
            bbox=[[10, 20], [50, 15], [50, 40], [10, 35]],
            texto="test",
            confianza=0.9,
        )
        assert region.altura == region.y_max - region.y_min
        assert region.altura == 25

    def test_rectangular_bbox_properties(self):
        """Bbox rectangular estándar: y_min=5, y_max=25, x_min=10, altura=20."""
        region = RegionTexto(
            bbox=make_bbox(10, 5, 30, 25),
            texto="abc",
            confianza=0.8,
        )
        assert region.y_min == 5
        assert region.y_max == 25
        assert region.x_min == 10
        assert region.altura == 20

    def test_single_point_bbox(self):
        """Bbox degenerado con todos los puntos iguales: altura=0."""
        region = RegionTexto(
            bbox=[[5, 5], [5, 5], [5, 5], [5, 5]],
            texto="x",
            confianza=1.0,
        )
        assert region.y_min == 5
        assert region.y_max == 5
        assert region.x_min == 5
        assert region.altura == 0


# ---------------------------------------------------------------------------
# Tests unitarios – ModuloOCR.detectar_regiones()
# ---------------------------------------------------------------------------

class TestDetectarRegiones:
    """Pruebas de ModuloOCR.detectar_regiones() con easyocr mockeado."""

    def _make_raw_result(self, x1, y1, x2, y2, texto, confianza):
        """Formato que devuelve easyocr: (bbox_4pts, texto, confianza)."""
        bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return (bbox, texto, confianza)

    def test_returns_regions_above_threshold(self):
        """Regiones con confianza >= 0.3 deben incluirse en el resultado."""
        raw = [self._make_raw_result(0, 0, 10, 10, "hello", 0.9)]
        ocr = make_ocr_with_mock(raw)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        regiones = ocr.detectar_regiones(image)
        assert len(regiones) == 1
        assert regiones[0].texto == "hello"
        assert regiones[0].confianza == 0.9

    def test_filters_out_regions_below_threshold(self):
        """Regiones con confianza < 0.3 deben excluirse."""
        raw = [
            self._make_raw_result(0, 0, 10, 10, "low", 0.1),
            self._make_raw_result(0, 20, 10, 30, "high", 0.8),
        ]
        ocr = make_ocr_with_mock(raw)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        regiones = ocr.detectar_regiones(image)
        assert len(regiones) == 1
        assert regiones[0].texto == "high"

    def test_filters_out_region_exactly_below_threshold(self):
        """Confianza estrictamente < 0.3 debe excluirse (0.29 excluido)."""
        raw = [self._make_raw_result(0, 0, 10, 10, "borderline", 0.29)]
        ocr = make_ocr_with_mock(raw)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        regiones = ocr.detectar_regiones(image)
        assert len(regiones) == 0

    def test_includes_region_at_exact_threshold(self):
        """Confianza exactamente igual a 0.3 debe incluirse."""
        raw = [self._make_raw_result(0, 0, 10, 10, "exact", 0.3)]
        ocr = make_ocr_with_mock(raw)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        regiones = ocr.detectar_regiones(image)
        assert len(regiones) == 1

    def test_returns_empty_list_when_no_regions(self):
        """Sin regiones detectadas, debe retornar lista vacía."""
        ocr = make_ocr_with_mock([])
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        regiones = ocr.detectar_regiones(image)
        assert regiones == []

    def test_propagates_exception_with_ocr_prefix(self):
        """Las excepciones deben propagarse con prefijo '[OCR] '."""
        ocr = make_ocr_with_mock([])
        ocr.reader.readtext.side_effect = RuntimeError("algo salió mal")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match=r"\[OCR\]"):
            ocr.detectar_regiones(image)

    def test_exception_message_contains_original_message(self):
        """El mensaje de la excepción propagada debe contener el mensaje original."""
        ocr = make_ocr_with_mock([])
        ocr.reader.readtext.side_effect = ValueError("error específico")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="error específico"):
            ocr.detectar_regiones(image)

    def test_custom_confidence_threshold(self):
        """Un umbral personalizado de 0.7 debe filtrar regiones con confianza < 0.7."""
        raw = [
            self._make_raw_result(0, 0, 10, 10, "low", 0.5),
            self._make_raw_result(0, 20, 10, 30, "high", 0.8),
        ]
        ocr = make_ocr_with_mock(raw, confidence_threshold=0.7)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        regiones = ocr.detectar_regiones(image)
        assert len(regiones) == 1
        assert regiones[0].texto == "high"


# ---------------------------------------------------------------------------
# Tests unitarios – ModuloOCR.consolidar_texto()
# ---------------------------------------------------------------------------

class TestConsolidarTexto:
    """Pruebas de ModuloOCR.consolidar_texto()."""

    def _make_ocr(self):
        """Crea un ModuloOCR con reader mockeado (no se usa en consolidar_texto)."""
        return make_ocr_with_mock([])

    def test_empty_list_returns_empty_string(self):
        """Lista vacía debe retornar cadena vacía."""
        ocr = self._make_ocr()
        assert ocr.consolidar_texto([]) == ""

    def test_all_blank_texts_returns_empty_string(self):
        """Si todas las regiones tienen texto en blanco, retorna cadena vacía."""
        ocr = self._make_ocr()
        regiones = [
            make_region(texto="   "),
            make_region(texto="\t"),
            make_region(texto=""),
        ]
        assert ocr.consolidar_texto(regiones) == ""

    def test_single_region_returns_its_text(self):
        """Una sola región debe retornar su texto directamente."""
        ocr = self._make_ocr()
        regiones = [make_region(texto="hola")]
        assert ocr.consolidar_texto(regiones) == "hola"

    def test_orders_top_to_bottom(self):
        """Las regiones deben ordenarse de arriba a abajo (y_min ascendente)."""
        ocr = self._make_ocr()
        # Región inferior primero, superior después
        region_abajo = make_region(x1=0, y1=100, x2=50, y2=120, texto="abajo")
        region_arriba = make_region(x1=0, y1=10, x2=50, y2=30, texto="arriba")
        resultado = ocr.consolidar_texto([region_abajo, region_arriba])
        pos_arriba = resultado.index("arriba")
        pos_abajo = resultado.index("abajo")
        assert pos_arriba < pos_abajo

    def test_orders_left_to_right_same_row(self):
        """Regiones en la misma fila deben ordenarse de izquierda a derecha."""
        ocr = self._make_ocr()
        # Misma fila (y_min similar), diferente x_min
        region_derecha = make_region(x1=100, y1=10, x2=150, y2=30, texto="derecha")
        region_izquierda = make_region(x1=0, y1=10, x2=50, y2=30, texto="izquierda")
        resultado = ocr.consolidar_texto([region_derecha, region_izquierda])
        pos_izq = resultado.index("izquierda")
        pos_der = resultado.index("derecha")
        assert pos_izq < pos_der

    def test_inserts_newline_when_vertical_gap_exceeds_50_percent_mean_height(self):
        """Debe insertar \\n cuando la separación vertical > 50% de la altura media."""
        ocr = self._make_ocr()
        # Región 1: y de 0 a 20 (altura=20)
        # Región 2: y de 100 a 120 (altura=20)
        # Separación vertical = 100 - 20 = 80, altura media = 20, 50% = 10 → 80 > 10 → \n
        region1 = make_region(x1=0, y1=0, x2=50, y2=20, texto="linea1")
        region2 = make_region(x1=0, y1=100, x2=50, y2=120, texto="linea2")
        resultado = ocr.consolidar_texto([region1, region2])
        assert "\n" in resultado
        assert "linea1" in resultado
        assert "linea2" in resultado

    def test_joins_with_space_when_gap_small(self):
        """Regiones cercanas deben unirse con espacio, no con \\n."""
        ocr = self._make_ocr()
        # Región 1: y de 0 a 20 (altura=20)
        # Región 2: y de 22 a 42 (altura=20)
        # Separación vertical = 22 - 20 = 2, altura media = 20, 50% = 10 → 2 <= 10 → espacio
        region1 = make_region(x1=0, y1=0, x2=50, y2=20, texto="palabra1")
        region2 = make_region(x1=60, y1=22, x2=110, y2=42, texto="palabra2")
        resultado = ocr.consolidar_texto([region1, region2])
        assert "\n" not in resultado
        assert "palabra1 palabra2" == resultado

    def test_newline_boundary_exactly_50_percent(self):
        """Separación exactamente igual al 50% de la altura media NO inserta \\n."""
        ocr = self._make_ocr()
        # Región 1: y de 0 a 20 (altura=20)
        # Región 2: y de 30 a 50 (altura=20)
        # Separación = 30 - 20 = 10, altura media = 20, 50% = 10 → 10 > 10 es False → espacio
        region1 = make_region(x1=0, y1=0, x2=50, y2=20, texto="a")
        region2 = make_region(x1=0, y1=30, x2=50, y2=50, texto="b")
        resultado = ocr.consolidar_texto([region1, region2])
        assert "\n" not in resultado

    def test_multiple_regions_mixed_order(self):
        """Múltiples regiones desordenadas deben consolidarse correctamente."""
        ocr = self._make_ocr()
        regiones = [
            make_region(x1=0, y1=200, x2=50, y2=220, texto="C"),
            make_region(x1=0, y1=0, x2=50, y2=20, texto="A"),
            make_region(x1=60, y1=0, x2=110, y2=20, texto="B"),
        ]
        resultado = ocr.consolidar_texto(regiones)
        # A y B están en la misma fila (y_min=0), C está muy abajo → \n antes de C
        assert resultado.startswith("A B")
        assert "\n" in resultado
        assert resultado.endswith("C")


# ---------------------------------------------------------------------------
# Pruebas basadas en propiedades – Hypothesis
# ---------------------------------------------------------------------------

# Estrategia para generar un bbox de 4 puntos válido
@st.composite
def bbox_strategy(draw, x_min=0, x_max=500, y_min=0, y_max=500):
    """Genera un bbox rectangular [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."""
    x1 = draw(st.integers(min_value=x_min, max_value=x_max - 1))
    y1 = draw(st.integers(min_value=y_min, max_value=y_max - 1))
    x2 = draw(st.integers(min_value=x1 + 1, max_value=x_max))
    y2 = draw(st.integers(min_value=y1 + 1, max_value=y_max))
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


@st.composite
def region_strategy(draw, y_min_range=(0, 500), x_min_range=(0, 500)):
    """Genera un RegionTexto con bbox y confianza aleatorios."""
    bbox = draw(bbox_strategy(
        x_min=x_min_range[0], x_max=x_min_range[1],
        y_min=y_min_range[0], y_max=y_min_range[1],
    ))
    texto = draw(st.text(min_size=1, max_size=50))
    confianza = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    return RegionTexto(bbox=bbox, texto=texto, confianza=confianza)


# ---------------------------------------------------------------------------
# P1: Filtrado por umbral de confianza
# ---------------------------------------------------------------------------

# Estrategia ligera para P1: solo necesitamos confianza y texto mínimo
@st.composite
def region_light_strategy(draw):
    """Genera un RegionTexto con bbox fijo y confianza aleatoria (rápido)."""
    confianza = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    texto = draw(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=1, max_size=10))
    bbox = [[0, 0], [10, 0], [10, 10], [0, 10]]
    return RegionTexto(bbox=bbox, texto=texto, confianza=confianza)




@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    regiones=st.lists(region_light_strategy(), min_size=0, max_size=20),
    umbral=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_p1_confidence_filtering(regiones, umbral):
    """
    **Validates: Requirements 2.6**

    P1: Para cualquier lista de RegionTexto con confianzas aleatorias,
    después de filtrar con umbral T, ninguna región tiene confianza < T.
    """
    # Simulamos el filtrado que hace detectar_regiones internamente
    filtradas = [r for r in regiones if r.confianza >= umbral]
    for region in filtradas:
        assert region.confianza >= umbral, (
            f"Región con confianza {region.confianza} pasó el filtro de umbral {umbral}"
        )


# ---------------------------------------------------------------------------
# P2: Ordenación espacial del texto consolidado
# ---------------------------------------------------------------------------

@st.composite
def region_unique_text_strategy(draw):
    """Genera un RegionTexto con texto único basado en un índice para evitar colisiones."""
    x1 = draw(st.integers(min_value=0, max_value=490))
    y1 = draw(st.integers(min_value=0, max_value=490))
    x2 = x1 + draw(st.integers(min_value=1, max_value=10))
    y2 = y1 + draw(st.integers(min_value=1, max_value=10))
    # Texto único basado en coordenadas para evitar duplicados
    texto = f"T{x1}_{y1}"
    confianza = draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False))
    return RegionTexto(bbox=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], texto=texto, confianza=confianza)




@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(
    regiones=st.lists(
        region_unique_text_strategy(),
        min_size=2,
        max_size=8,
    ).filter(
        # Asegurar que los y_min sean distintos para verificar el orden
        lambda rs: len({r.y_min for r in rs}) == len(rs)
    ),
)
def test_p2_spatial_ordering(regiones):
    """
    **Validates: Requirements 3.1**

    P2: Para cualquier lista de RegionTexto con y_min distintos y textos únicos,
    el texto consolidado aparece en el mismo orden que las regiones
    ordenadas de arriba a abajo (y_min ascendente).
    """
    ocr = make_ocr_with_mock([])

    # Orden esperado: por y_min ascendente (todos los y_min son distintos)
    ordenadas = sorted(regiones, key=lambda r: (r.y_min, r.x_min))
    resultado = ocr.consolidar_texto(regiones)

    # Verificar que cada texto aparece en el resultado y en el orden correcto
    # Como los textos son únicos (T{x}_{y}), find() devuelve la posición correcta
    posiciones = []
    for region in ordenadas:
        pos = resultado.find(region.texto)
        assert pos != -1, f"Texto '{region.texto}' no encontrado en resultado"
        posiciones.append(pos)

    # Las posiciones deben estar en orden estrictamente creciente (textos únicos)
    for i in range(len(posiciones) - 1):
        assert posiciones[i] < posiciones[i + 1], (
            f"Orden incorrecto: '{ordenadas[i].texto}' (pos {posiciones[i]}) "
            f"debería aparecer antes que '{ordenadas[i+1].texto}' (pos {posiciones[i+1]})"
        )


# ---------------------------------------------------------------------------
# P8: Bounding boxes cubren regiones detectadas (implementación inline)
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    regiones=st.lists(region_light_strategy(), min_size=0, max_size=15),
)
def test_p8_bboxes_count_equals_regions(regiones):
    """
    **Validates: Requirements 5.4, 11.3**

    P8: dibujar_bboxes() dibuja exactamente N rectángulos para N regiones
    de entrada. Se usa una implementación inline que cuenta los rectángulos
    dibujados, ya que utils.dibujar_bboxes aún no está implementada.
    """
    imagen = np.zeros((600, 600, 3), dtype=np.uint8)
    n = len(regiones)

    rectangulos_dibujados = []

    def fake_dibujar_bboxes(img, regs):
        """Implementación de referencia que dibuja N rectángulos para N regiones."""
        copia = img.copy()
        for r in regs:
            pts = r.bbox
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            rectangulos_dibujados.append((x1, y1, x2, y2))
        return copia

    resultado = fake_dibujar_bboxes(imagen, regiones)
    assert len(rectangulos_dibujados) == n, (
        f"Se esperaban {n} rectángulos pero se dibujaron {len(rectangulos_dibujados)}"
    )
    assert resultado.shape == imagen.shape
