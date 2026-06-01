# Implementation Plan: Sistema de Traducción Multimodal de Imágenes

## Overview

Implementación incremental del pipeline de traducción de imágenes en seis fases: configuración del entorno, módulo OCR, módulo de traducción, orquestador, interfaz Gradio y pruebas. Cada fase produce código ejecutable de forma independiente antes de integrarse con las demás.

---

## Tasks

- [x] 1. Fase 1 – Configuración del entorno y estructura del proyecto
  - [x] 1.1 Crear la estructura de directorios y archivos base del proyecto
    - Crear los directorios `src/`, `models/`, `tests/`
    - Crear los archivos vacíos `src/__init__.py`, `tests/__init__.py`, `models/.gitkeep`
    - Crear `app.py`, `src/orchestrator.py`, `src/ocr_module.py`, `src/translation_module.py`, `src/utils.py` con esqueletos mínimos (imports y docstring de módulo)
    - _Requisitos: 8.1, 8.2_

  - [x] 1.2 Crear `requirements.txt` con todas las dependencias fijadas con `==`
    - Incluir: `easyocr`, `torch`, `torchvision`, `transformers`, `gradio`, `langdetect`, `opencv-python`, `Pillow`, `hypothesis`, `pytest`, `numpy`
    - Fijar versiones compatibles entre sí para Python 3.8+
    - _Requisitos: 7.4, 7.5_

- [x] 2. Fase 2 – Módulo OCR (`src/ocr_module.py`)
  - [x] 2.1 Implementar el dataclass `RegionTexto` en `src/ocr_module.py`
    - Definir campos `bbox`, `texto`, `confianza` con sus tipos
    - Implementar las propiedades calculadas `y_min`, `y_max`, `x_min`, `altura`
    - _Requisitos: 2.2, 9.1_

  - [x] 2.2 Implementar la clase `ModuloOCR` con `__init__` y `detectar_regiones()`
    - Inicializar `easyocr.Reader` con los idiomas configurados y el flag `use_gpu`
    - En `detectar_regiones()`: invocar `reader.readtext()`, filtrar por `confidence_threshold`, retornar lista de `RegionTexto`
    - Propagar excepciones con prefijo `"[OCR] "`
    - _Requisitos: 2.1, 2.2, 2.3, 2.4, 2.6, 2.7_

  - [x] 2.3 Implementar `consolidar_texto()` en `ModuloOCR`
    - Ordenar regiones por `y_min` ascendente y, dentro de la misma fila, por `x_min` ascendente
    - Insertar `\n` cuando la separación vertical entre regiones supere el 50% de la altura media de la fila
    - Retornar cadena vacía si todas las regiones tienen texto vacío o solo espacios
    - _Requisitos: 3.1, 3.2, 3.3_

- [x] 3. Checkpoint – Validar Módulo OCR
  - Asegurarse de que todos los tests de `tests/test_ocr_module.py` pasan con `pytest tests/test_ocr_module.py`. Consultar al usuario si surgen dudas.

- [x] 4. Fase 3 – Módulo de Traducción (`src/translation_module.py`)
  - [x] 4.1 Implementar la constante `IDIOMAS_SOPORTADOS` y la clase `ModuloTraduccion` con `__init__`
    - Definir el diccionario `IDIOMAS_SOPORTADOS` con los 9 idiomas y sus códigos `easyocr`, `iso` y `mbart`
    - Inicializar `_model_cache: dict` vacío y el flag `use_gpu`
    - _Requisitos: 4.2, 7.2_

  - [x] 4.2 Implementar `_seleccionar_modelo()` y `_cargar_modelo()` en `ModuloTraduccion`
    - En `_seleccionar_modelo()`: intentar primero `Helsinki-NLP/opus-mt-{src}-{tgt}`; si no existe, usar `facebook/mbart-large-50-many-to-many-mmt`
    - En `_cargar_modelo()`: cargar el pipeline de Hugging Face, almacenarlo en `_model_cache[(src, tgt)]` y retornarlo
    - _Requisitos: 7.2, 4.5_

  - [x] 4.3 Implementar `traducir()` en `ModuloTraduccion`
    - Validar que el texto no esté vacío ni sea solo espacios; lanzar excepción con mensaje exacto si lo está
    - Validar que el texto no supere 50.000 caracteres; lanzar excepción con mensaje exacto si lo supera
    - Si `idioma_origen == idioma_destino`, retornar el texto sin modificar sin cargar modelo
    - Verificar que el par de idiomas está soportado; lanzar excepción con mensaje exacto si no lo está
    - Reutilizar modelo de caché si ya fue cargado; cargarlo si no
    - _Requisitos: 4.1, 4.3, 4.4, 4.5, 4.6, 4.7_

- [x] 5. Checkpoint – Validar Módulo de Traducción
  - Asegurarse de que todos los tests de `tests/test_translation_module.py` pasan con `pytest tests/test_translation_module.py`. Consultar al usuario si surgen dudas.

- [x] 6. Fase 4 – Orquestador (`src/orchestrator.py`)
  - [x] 6.1 Implementar los dataclasses `RegionTexto` y `ResultadoTraduccion` en `src/orchestrator.py` (o importarlos desde `src/ocr_module.py`)
    - Definir `ResultadoTraduccion` con campos: `texto_original`, `idioma_origen`, `texto_traducido`, `idioma_destino`, `regiones`, `error`, `tiempos_ms`
    - _Requisitos: 6.1, 6.3_

  - [x] 6.2 Implementar `TraductorDeImagenes.__init__()`, `cargar_imagen()` y `detectar_texto()`
    - En `__init__()`: inicializar `ModuloOCR`, `ModuloTraduccion`, `target_language`, `confidence_threshold=0.3`, `use_gpu`; detectar GPU con `torch.cuda.is_available()`
    - En `cargar_imagen()`: leer la imagen con OpenCV/Pillow, retornar `np.ndarray`; registrar estado interno de imagen cargada
    - En `detectar_texto()`: verificar que `cargar_imagen` fue invocado; invocar `ModuloOCR.detectar_regiones()` y `consolidar_texto()`; retornar lista de dicts con campos `bbox`, `texto`, `confianza`
    - _Requisitos: 6.2, 6.5, 9.1, 7.6_

  - [x] 6.3 Implementar `TraductorDeImagenes.traducir()` y `procesar()`
    - En `traducir()`: verificar estado de imagen cargada; invocar `utils.detectar_idioma()` sobre el texto; invocar `ModuloTraduccion.traducir()`; retornar texto traducido
    - En `procesar()`: ejecutar el pipeline completo en orden (`cargar_imagen` → `detectar_texto` → `traducir` → construcción de `ResultadoTraduccion`); registrar tiempo en ms de cada etapa con `utils.milisegundos_actuales()`; capturar cualquier excepción, registrar en log (etapa + tipo + mensaje) y retornar `ResultadoTraduccion` con campo `error` poblado
    - Manejar el caso de lista vacía de regiones retornando `ResultadoTraduccion` con `error = "No se detectó texto en la imagen"`
    - _Requisitos: 6.1, 6.2, 6.3, 6.4, 6.5, 2.5_

- [x] 7. Checkpoint – Validar Orquestador
  - Asegurarse de que todos los tests de `tests/test_orchestrator.py` pasan con `pytest tests/test_orchestrator.py`. Consultar al usuario si surgen dudas.

- [x] 8. Fase 4b – Utilidades (`src/utils.py`)
  - [x] 8.1 Implementar `dibujar_bboxes()`, `detectar_idioma()`, `configurar_logging()` y `milisegundos_actuales()` en `src/utils.py`
    - `dibujar_bboxes()`: recibir `np.ndarray` y lista de `RegionTexto`; dibujar rectángulos con OpenCV sobre una copia de la imagen; retornar la copia anotada
    - `detectar_idioma()`: invocar `langdetect.detect()`; retornar código ISO 639-1
    - `configurar_logging()`: configurar el logger con formato `[TIMESTAMP] [NIVEL] [ETAPA] mensaje`; retornar instancia de `logging.Logger`
    - `milisegundos_actuales()`: retornar `int(time.time() * 1000)`
    - _Requisitos: 5.4, 6.4_

- [x] 9. Fase 5 – Interfaz Gradio (`app.py`)
  - [x] 9.1 Implementar `validar_imagen()` en `app.py`
    - Verificar tipo MIME del archivo (no solo extensión) para JPEG, PNG y BMP
    - Verificar que el tamaño no supera 10 MB
    - Retornar `(True, "")` si es válida o `(False, mensaje_error)` con el mensaje exacto definido en los requisitos
    - _Requisitos: 1.1, 1.3, 1.4_

  - [x] 9.2 Implementar `procesar_imagen()` como callback principal de Gradio en `app.py`
    - Recibir `imagen: np.ndarray` e `idioma_destino: str`
    - Validar que se ha seleccionado un idioma destino; mostrar mensaje si no
    - Instanciar `TraductorDeImagenes` e invocar `procesar()`
    - Invocar `utils.dibujar_bboxes()` con las regiones del resultado
    - Retornar la tupla `(imagen_anotada, texto_original, idioma_origen, texto_traducido, mensaje_error)`
    - _Requisitos: 1.2, 1.6, 5.1, 5.2, 5.3, 5.6_

  - [x] 9.3 Implementar `construir_interfaz()` con `gradio.Blocks` en `app.py`
    - Añadir componente de carga de imagen (`gr.Image`)
    - Añadir selector de `Idioma_Destino` con los 9 idiomas soportados, disponible antes de cargar la imagen
    - Añadir indicador de progreso visible durante el procesamiento
    - Añadir componentes de salida: imagen anotada, texto original con etiqueta e idioma detectado, texto traducido con etiqueta e idioma destino
    - Conectar el botón de envío al callback `procesar_imagen()`
    - _Requisitos: 1.5, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 9.4 Añadir el bloque `if __name__ == "__main__"` en `app.py` para lanzar la interfaz
    - Llamar a `construir_interfaz().launch()` con los parámetros adecuados para entorno local y Colab
    - _Requisitos: 7.4, 11.5_

- [x] 10. Checkpoint – Validar Interfaz Gradio
  - Asegurarse de que `app.py` arranca sin excepciones no controladas ejecutando `python app.py` (o la celda equivalente en Colab). Consultar al usuario si surgen dudas.

- [x] 11. Fase 6 – Pruebas de integración y ajuste final
  - [x] 11.1 Escribir prueba de integración end-to-end: imagen en inglés → español
    - Procesar una imagen de prueba real con texto en inglés usando `TraductorDeImagenes.procesar()`
    - Verificar que el resultado contiene texto original no vacío, idioma origen detectado como `"en"` y texto traducido no vacío
    - Verificar que el tiempo total de procesamiento es inferior a 60 segundos en entorno sin GPU
    - Archivo: `tests/test_orchestrator.py`
    - _Requisitos: 11.1_

  - [x] 11.2 Escribir prueba de integración end-to-end: imagen en japonés/chino → español
    - Procesar una imagen de prueba real con texto en japonés o chino usando `TraductorDeImagenes.procesar()`
    - Verificar que el resultado contiene texto original no vacío y texto traducido no vacío
    - Verificar que el tiempo total de procesamiento es inferior a 120 segundos en entorno sin GPU
    - Archivo: `tests/test_orchestrator.py`
    - _Requisitos: 11.2_

  - [x] 11.3 Escribir prueba de integración: imagen sin texto detectable
    - Procesar una imagen sin texto usando `TraductorDeImagenes.procesar()`
    - Verificar que `ResultadoTraduccion.error == "No se detectó texto en la imagen"`
    - Archivo: `tests/test_orchestrator.py`
    - _Requisitos: 2.5, 11.6_

- [x] 12. Checkpoint final – Validar el sistema completo
  - Ejecutar `pytest tests/` y confirmar que al menos el 80% de los casos pasan. Verificar que `python app.py` arranca sin errores. Consultar al usuario si surgen dudas.

---

## Notes

- Las tareas marcadas con `*` son opcionales y pueden omitirse para una entrega MVP más rápida.
- Cada tarea referencia los requisitos específicos que implementa para trazabilidad completa.
- Los checkpoints garantizan validación incremental antes de avanzar a la siguiente fase.
- Las pruebas de propiedad (Hypothesis) validan invariantes universales; las pruebas unitarias validan casos concretos y condiciones de error.
- El directorio `models/` actúa como caché local para los modelos de Hugging Face descargados; no debe incluirse en el repositorio (añadir a `.gitignore`).
- La detección de GPU se realiza automáticamente mediante `torch.cuda.is_available()`; no se requiere configuración manual.

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "4.2"] },
    { "id": 3, "tasks": ["2.3", "2.4", "4.3"] },
    { "id": 4, "tasks": ["2.5", "2.6", "4.4", "4.5", "4.6", "4.7"] },
    { "id": 5, "tasks": ["6.1", "8.1"] },
    { "id": 6, "tasks": ["6.2"] },
    { "id": 7, "tasks": ["6.3", "8.2"] },
    { "id": 8, "tasks": ["6.4", "6.5", "6.6"] },
    { "id": 9, "tasks": ["9.1"] },
    { "id": 10, "tasks": ["9.2", "9.3"] },
    { "id": 11, "tasks": ["9.4"] },
    { "id": 12, "tasks": ["11.1", "11.2", "11.3"] },
    { "id": 13, "tasks": ["11.4"] }
  ]
}
```
