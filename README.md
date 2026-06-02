# Sistema de Traducción Multimodal de Imágenes

Aplicación Python que detecta texto en imágenes mediante OCR, identifica automáticamente el idioma y lo traduce al idioma que el usuario elija, todo desde una interfaz web simple construida con Gradio.

---

## Demostración

1. Sube una imagen (JPEG, PNG o BMP) que contenga texto
2. Selecciona el idioma destino
3. Pulsa **Traducir**
4. Obtén el texto original, la traducción y la imagen anotada con los bloques de texto detectados

---

## Características principales

- **OCR multilingüe** — detecta texto en inglés, español, francés, alemán, portugués, italiano, chino simplificado, japonés y coreano
- **Traducción automática** — usa modelos locales de Hugging Face sin llamadas a APIs externas de pago
- **Detección de idioma origen** — identifica automáticamente el idioma del texto extraído
- **Visualización de resultados** — muestra la imagen original con rectángulos sobre cada región de texto detectada
- **GPU opcional** — detecta automáticamente CUDA; funciona perfectamente en CPU
- **Portabilidad** — ejecutable en entorno local (Python 3.10.11)

---

## Arquitectura

El sistema sigue una arquitectura de **pipeline secuencial** con un orquestador central que coordina módulos independientes.

```
┌─────────────────────────────────────────────────────┐
│                  Módulo Interfaz                      │
│                    (app.py)                           │
│         Gradio Blocks · Validación MIME/tamaño        │
└────────────────────┬────────────────────────────────┘
                     │  imagen + idioma destino
                     ▼
┌─────────────────────────────────────────────────────┐
│               Módulo Orquestador                      │
│           (src/orchestrator.py)                       │
│         TraductorDeImagenes · Control de flujo        │
│         Registro de tiempos · Manejo de errores       │
└──────┬──────────────────────────────────┬────────────┘
       │                                  │
       ▼                                  ▼
┌──────────────────┐            ┌─────────────────────┐
│   Módulo OCR     │            │  Módulo Traducción   │
│ (ocr_module.py)  │            │(translation_module.py│
│ EasyOCR          │            │ Helsinki-NLP opus-mt  │
│ Filtrado conf.   │            │ facebook/mbart-large  │
│ Ordenación bbox  │            │ Pivot vía inglés      │
└──────────────────┘            └─────────────────────┘
       │                                  │
       └──────────────┬───────────────────┘
                      ▼
              ┌───────────────┐
              │   utils.py    │
              │ Dibuja bboxes │
              │ Detecta idioma│
              │ Logging       │
              └───────────────┘
```

### Flujo de datos

```
Imagen (JPEG/PNG/BMP)
  → Validación formato MIME y tamaño (≤ 10 MB)
  → EasyOCR: detección de regiones de texto
  → Filtrado por confianza OCR (umbral ≥ 0.3)
  → Consolidación y ordenación del texto (arriba→abajo, izquierda→derecha)
  → langdetect: identificación del idioma origen
  → Modelo Hugging Face: traducción al idioma destino
  → OpenCV: dibujo de bounding boxes sobre la imagen original
  → Presentación: imagen anotada + texto original + traducción
```

---

## Modelos de IA utilizados

### OCR — EasyOCR

| Propiedad | Detalle |
|---|---|
| Librería | [EasyOCR 1.7.1](https://github.com/JaidedAI/EasyOCR) |
| Backend | PyTorch |
| Idiomas configurados | `en`, `es`, `fr`, `de`, `pt`, `it` (grupo latino) |
| Salida | Lista de regiones con `bbox`, `texto` y `confianza` |

> Los idiomas asiáticos (japonés, chino, coreano) se detectan en las imágenes pero su OCR se gestiona con readers separados porque EasyOCR no mezcla idiomas latinos y asiáticos en el mismo reader.

### Traducción — Hugging Face Transformers

El sistema usa una **estrategia de selección de modelo en dos niveles**:

#### Nivel 1 — Helsinki-NLP opus-mt (modelos directos, ~300 MB)

Modelos Marian MT especializados por par de idiomas. Más ligeros y rápidos.

| Par de idiomas | Modelo |
|---|---|
| Inglés → Español | `Helsinki-NLP/opus-mt-en-es` |
| Inglés → Francés | `Helsinki-NLP/opus-mt-en-fr` |
| Inglés → Alemán | `Helsinki-NLP/opus-mt-en-de` |
| Inglés → Chino | `Helsinki-NLP/opus-mt-en-zh` |
| Inglés → Japonés | `Helsinki-NLP/opus-mt-en-jap` |
| Inglés → Coreano | `Helsinki-NLP/opus-mt-en-ko` |
| Japonés → Español | `Helsinki-NLP/opus-mt-ja-es` |
| Coreano → Español | `Helsinki-NLP/opus-mt-ko-es` |
| ... y más pares | `Helsinki-NLP/opus-mt-{src}-{tgt}` |

#### Nivel 2 — facebook/mbart-large-50 (modelo multilingüe, ~2.4 GB)

Modelo usado como fallback cuando no existe un opus-mt de calidad suficiente para el par solicitado.

| Propiedad | Detalle |
|---|---|
| Modelo | `facebook/mbart-large-50-many-to-many-mmt` |
| Cobertura | 50 idiomas en cualquier dirección |
| Tamaño | ~2.4 GB |
| Uso | Fallback cuando opus-mt no existe en el Hub |

#### Traducción con pivot (inglés como idioma intermedio)

Para pares asiáticos sin ruta directa de calidad (por ejemplo chino→español), el sistema usa inglés como idioma puente:

```
中文 (zh) ──→ opus-mt-zh-en ──→ English ──→ opus-mt-en-es ──→ Español
```

Pares que usan pivot: `zh→es/fr/de/pt/it/ja/ko`, `ja→fr/de/pt/it/ko/zh`, `ko→fr/de/pt/it/ja/zh`.

### Detección de idioma — langdetect

Detecta automáticamente el idioma del texto extraído por OCR y devuelve el código ISO 639-1 (`en`, `es`, `ja`, etc.).

---

## Estructura del proyecto

```
modelo_ia_traductor/
├── app.py                      # Punto de entrada — interfaz Gradio
├── requirements.txt            # Dependencias con versiones fijadas
├── README.md                   # Este archivo
├── src/
│   ├── __init__.py
│   ├── orchestrator.py         # TraductorDeImagenes — coordina el pipeline
│   ├── ocr_module.py           # ModuloOCR — EasyOCR + filtrado + consolidación
│   ├── translation_module.py   # ModuloTraduccion — Hugging Face + caché + pivot
│   └── utils.py                # Utilidades: bboxes, logging, detección de idioma
├── models/                     # Caché local de modelos descargados (no en git)
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_ocr_module.py       # Tests unitarios y de propiedad del OCR
    ├── test_translation_module.py
    └── test_orchestrator.py    # Tests de integración end-to-end
```

---

## Idiomas soportados

| Idioma | OCR (EasyOCR) | ISO | mBART |
|---|---|---|---|
| Español | `es` | `es` | `es_XX` |
| Inglés | `en` | `en` | `en_XX` |
| Francés | `fr` | `fr` | `fr_XX` |
| Alemán | `de` | `de` | `de_DE` |
| Portugués | `pt` | `pt` | `pt_XX` |
| Italiano | `it` | `it` | `it_IT` |
| Chino simplificado | `ch_sim` | `zh` | `zh_CN` |
| Japonés | `ja` | `ja` | `ja_XX` |
| Coreano | `ko` | `ko` | `ko_KR` |

---

## Instalación y uso

### Requisitos

- Python 3.10.11
- 4 GB de RAM mínimo (8 GB recomendado para mBART)
- GPU con CUDA (opcional, mejora la velocidad 5-10x)

### Instalación

```bash
pip install -r requirements.txt
```

### Ejecución

```bash
python app.py
```

Abre el navegador en **http://localhost:7860**

---

## Dependencias principales

| Paquete | Versión | Rol |
|---|---|---|
| `torch` | 2.0.1 | Backend de deep learning |
| `easyocr` | 1.7.1 | OCR multilingüe |
| `transformers` | 4.35.2 | Modelos de traducción Hugging Face |
| `gradio` | 3.50.2 | Interfaz web |
| `langdetect` | 1.0.9 | Detección de idioma origen |
| `opencv-python` | 4.8.1.78 | Dibujo de bounding boxes |
| `sentencepiece` | 0.1.99 | Tokenizer para mBART |
| `sacremoses` | 0.1.1 | Tokenizer para Marian MT |
| `Pillow` | 10.1.0 | Procesamiento de imágenes |
| `numpy` | 1.24.4 | Operaciones matriciales |

---

## Notas de rendimiento

| Situación | Tiempo aproximado |
|---|---|
| Primera traducción (descarga modelo opus-mt) | 1–3 min |
| Primera traducción con mBART (descarga modelo) | 5–10 min |
| Traducciones posteriores (mismo par, modelo en caché) | 2–15 seg |
| Con GPU CUDA | 5–10x más rápido |

Los modelos se descargan una sola vez y quedan en `~/.cache/huggingface/hub/`.

---

## Tests

```bash
# Ejecutar todos los tests
pytest tests/

# Solo tests unitarios de OCR
pytest tests/test_ocr_module.py

# Solo tests de traducción
pytest tests/test_translation_module.py

# Solo tests de integración
pytest tests/test_orchestrator.py
```

El sistema supera el 80% de los casos de prueba definidos (criterio de aceptación del proyecto).

---

## Manejo de errores

El sistema usa propagación controlada: los módulos lanzan excepciones tipadas, el orquestador las captura, las registra en log y las convierte en un resultado con campo `error` que la interfaz muestra al usuario sin exponer stack traces ni rutas internas.

| Situación | Mensaje mostrado al usuario |
|---|---|
| Formato de imagen no soportado | "Formato no soportado. Por favor, sube una imagen en formato JPEG, PNG o BMP." |
| Imagen mayor a 10 MB | "La imagen supera el límite de 10 MB. Por favor, sube una imagen más pequeña." |
| Sin idioma destino seleccionado | "Por favor, selecciona un idioma destino antes de continuar." |
| Sin texto detectable en la imagen | "No se detectó texto en la imagen" |
| Error en OCR | "Error en OCR: \<descripción\>" |
| Error en traducción | "Error en traducción: \<descripción\>" |

---

## Licencia

Este proyecto usa exclusivamente tecnologías open-source:
- EasyOCR — Apache 2.0
- Hugging Face Transformers — Apache 2.0
- Gradio — Apache 2.0
- PyTorch — BSD 3-Clause
