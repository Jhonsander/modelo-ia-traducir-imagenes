# Requirements Document

## Introduction

El **Sistema de Traducción Multimodal de Imágenes** es una aplicación Python que permite al usuario cargar una imagen, foto o captura de pantalla que contiene texto en cualquier idioma. El sistema detecta automáticamente las regiones de texto presentes en la imagen, extrae el contenido mediante OCR, lo traduce al idioma elegido por el usuario y presenta tanto el texto original como su traducción en una interfaz web simple.

El sistema se construye con un enfoque modular basado en un pipeline de componentes independientes, utilizando exclusivamente tecnologías open-source (EasyOCR, Hugging Face Transformers, Gradio) y está diseñado para ejecutarse en entornos locales o en Google Colab.

---

## Glossary

- **Sistema**: La aplicación completa de Traducción Multimodal de Imágenes.
- **Pipeline**: Secuencia ordenada de módulos que procesan la imagen de entrada hasta producir la traducción final.
- **Módulo_OCR**: Componente responsable de detectar regiones de texto en la imagen y extraer el texto mediante reconocimiento óptico de caracteres, implementado con EasyOCR.
- **Módulo_Traducción**: Componente responsable de traducir el texto extraído al idioma destino elegido por el usuario, implementado con modelos de Hugging Face Transformers (Helsinki-NLP/opus-mt o facebook/mbart-large-50).
- **Módulo_Interfaz**: Componente de interfaz de usuario web implementado con Gradio que permite la interacción del usuario con el Sistema.
- **Módulo_Orquestador**: Componente central (clase `TraductorDeImagenes`) que coordina la ejecución secuencial del Pipeline.
- **Región_de_Texto**: Área rectangular de la imagen identificada por el Módulo_OCR como contenedora de texto.
- **Idioma_Origen**: Idioma detectado automáticamente en el texto extraído de la imagen.
- **Idioma_Destino**: Idioma al que el usuario desea traducir el texto extraído.
- **Resultado_de_Traducción**: Estructura de datos que contiene el texto original, el idioma detectado, el texto traducido, las coordenadas de las Regiones_de_Texto y, opcionalmente, un campo de error.
- **Imagen_de_Entrada**: Archivo de imagen en formato JPEG, PNG o BMP proporcionado por el usuario.
- **Confianza_OCR**: Valor numérico en el rango [0.0, 1.0] que indica la certeza del Módulo_OCR sobre el texto reconocido en una Región_de_Texto.

---

## Requirements

### Requisito 1: Carga de Imagen

**User Story:** Como usuario, quiero subir una imagen que contiene texto, para que el sistema pueda procesarla y traducir su contenido.

#### Criterios de Aceptación

1. THE Módulo_Interfaz SHALL aceptar Imágenes_de_Entrada en los formatos JPEG, PNG y BMP, verificando el tipo MIME del archivo además de la extensión.
2. WHEN el usuario carga una Imagen_de_Entrada válida, THE Módulo_Interfaz SHALL transmitir la imagen al Módulo_Orquestador para iniciar el Pipeline sin requerir ninguna acción adicional del usuario.
3. IF el usuario carga un archivo cuyo tipo MIME no corresponde a JPEG, PNG ni BMP, THEN THE Módulo_Interfaz SHALL mostrar el mensaje "Formato no soportado. Por favor, sube una imagen en formato JPEG, PNG o BMP." y no iniciará el Pipeline.
4. IF el usuario carga una Imagen_de_Entrada cuyo tamaño supera los 10 MB, THEN THE Módulo_Interfaz SHALL mostrar el mensaje "La imagen supera el límite de 10 MB. Por favor, sube una imagen más pequeña." y no iniciará el Pipeline.
5. THE Módulo_Interfaz SHALL presentar al usuario un selector de Idioma_Destino con al menos los idiomas: español, inglés, francés, alemán, portugués, italiano, chino simplificado, japonés y coreano; y este selector SHALL estar disponible antes de que el usuario cargue la imagen.
6. IF el usuario intenta iniciar el Pipeline sin haber seleccionado un Idioma_Destino, THEN THE Módulo_Interfaz SHALL mostrar un mensaje indicando que debe seleccionar un idioma destino antes de continuar.

---

### Requisito 2: Detección de Regiones de Texto mediante OCR

**User Story:** Como usuario, quiero que el sistema detecte automáticamente las regiones de la imagen que contienen texto, para no tener que marcarlas manualmente.

#### Criterios de Aceptación

1. WHEN el Módulo_Orquestador invoca el Módulo_OCR con una Imagen_de_Entrada, THE Módulo_OCR SHALL detectar todas las Regiones_de_Texto presentes en la imagen y retornar una lista (posiblemente vacía) de regiones detectadas.
2. THE Módulo_OCR SHALL retornar, para cada Región_de_Texto detectada, las coordenadas del rectángulo delimitador como cuatro puntos (x, y), el texto reconocido como cadena de caracteres y el valor de Confianza_OCR como número en el rango [0.0, 1.0].
3. WHEN el Módulo_OCR procesa una Imagen_de_Entrada, THE Módulo_OCR SHALL ser capaz de detectar texto en al menos los idiomas: inglés, español, francés, alemán, portugués, italiano, chino simplificado, japonés y coreano, configurando EasyOCR con los códigos de idioma correspondientes.
4. IF la Imagen_de_Entrada no contiene ninguna Región_de_Texto detectable, THEN THE Módulo_OCR SHALL retornar una lista vacía de regiones.
5. WHEN el Módulo_Orquestador recibe una lista vacía de regiones del Módulo_OCR, THE Módulo_Orquestador SHALL construir un Resultado_de_Traducción con el campo de texto original vacío y notificar al Módulo_Interfaz mediante el campo de error con el valor "No se detectó texto en la imagen".
6. THE Módulo_OCR SHALL excluir del resultado retornado al Módulo_Orquestador todas las Regiones_de_Texto cuyo valor de Confianza_OCR sea estrictamente inferior a 0.3.
7. IF el Módulo_OCR lanza una excepción durante el procesamiento de la Imagen_de_Entrada, THEN THE Módulo_OCR SHALL propagar la excepción al Módulo_Orquestador con un mensaje que identifique la etapa "OCR" como origen del error.

---

### Requisito 3: Extracción de Texto

**User Story:** Como usuario, quiero que el sistema extraiga el texto de las regiones detectadas, para obtener el contenido textual de la imagen.

#### Criterios de Aceptación

1. WHEN el Módulo_OCR completa la detección de Regiones_de_Texto y la lista de regiones no está vacía, THE Módulo_OCR SHALL consolidar el texto extraído de todas las regiones no vacías en un único bloque de texto, ordenado de arriba a abajo según la coordenada Y mínima de cada región y, dentro de la misma fila, de izquierda a derecha según la coordenada X mínima.
2. THE Módulo_OCR SHALL insertar un carácter de salto de línea (`\n`) entre dos Regiones_de_Texto consecutivas cuando la diferencia entre la coordenada Y mínima de la segunda región y la coordenada Y máxima de la primera región sea mayor que el 50% de la altura media de las regiones de esa fila.
3. IF todas las Regiones_de_Texto detectadas tienen texto vacío o contienen únicamente espacios en blanco tras el filtrado por Confianza_OCR, THEN THE Módulo_OCR SHALL retornar una cadena vacía como texto consolidado.
4. THE Módulo_Orquestador SHALL incluir el texto consolidado retornado por el Módulo_OCR en el campo `texto_original` del Resultado_de_Traducción final.

---

### Requisito 4: Traducción del Texto Extraído

**User Story:** Como usuario, quiero que el sistema traduzca el texto extraído al idioma que yo elija, para entender el contenido de la imagen.

#### Criterios de Aceptación

1. WHEN el Módulo_Orquestador invoca el Módulo_Traducción con el texto consolidado, el Idioma_Origen detectado y el Idioma_Destino seleccionado, THE Módulo_Traducción SHALL retornar el texto traducido al Idioma_Destino en un tiempo no superior a 30 segundos para textos de hasta 5.000 caracteres.
2. THE Módulo_Traducción SHALL soportar como Idioma_Destino al menos los siguientes idiomas: español, inglés, francés, alemán, portugués, italiano, chino simplificado, japonés y coreano.
3. WHEN el Módulo_Traducción recibe texto cuyo Idioma_Origen es igual al Idioma_Destino, THE Módulo_Traducción SHALL retornar el texto original sin modificaciones y sin invocar ningún modelo de traducción.
4. IF el Módulo_Traducción no dispone de un modelo para el par Idioma_Origen–Idioma_Destino solicitado, THEN THE Módulo_Traducción SHALL lanzar una excepción con el mensaje "Par de idiomas no soportado: {Idioma_Origen} → {Idioma_Destino}".
5. WHEN el Módulo_Traducción ejecuta una traducción para un par Idioma_Origen–Idioma_Destino por segunda vez o más en la misma sesión, THE Módulo_Traducción SHALL completar la traducción en un tiempo no superior a 10 segundos para textos de hasta 5.000 caracteres, reutilizando el modelo ya cargado.
6. IF el texto consolidado recibido por el Módulo_Traducción es una cadena vacía o contiene únicamente espacios en blanco, THEN THE Módulo_Traducción SHALL lanzar una excepción con el mensaje "El texto de entrada para traducción está vacío".
7. IF el texto consolidado recibido por el Módulo_Traducción supera los 50.000 caracteres, THEN THE Módulo_Traducción SHALL lanzar una excepción con el mensaje "El texto supera el límite de 50.000 caracteres permitido para traducción".

---

### Requisito 5: Presentación de Resultados

**User Story:** Como usuario, quiero ver tanto el texto original detectado como su traducción, para comparar el contenido original con la versión traducida.

#### Criterios de Aceptación

1. WHEN el Pipeline completa el procesamiento de una Imagen_de_Entrada, THE Módulo_Interfaz SHALL mostrar el texto original extraído bajo una etiqueta visible con el texto "Texto original" y el texto traducido bajo una etiqueta visible con el texto "Traducción".
2. WHEN el Pipeline completa el procesamiento de una Imagen_de_Entrada, THE Módulo_Interfaz SHALL mostrar el código o nombre del Idioma_Origen detectado adyacente a la etiqueta "Texto original".
3. WHEN el Pipeline completa el procesamiento de una Imagen_de_Entrada, THE Módulo_Interfaz SHALL mostrar el código o nombre del Idioma_Destino seleccionado adyacente a la etiqueta "Traducción".
4. WHEN el Pipeline completa el procesamiento de una Imagen_de_Entrada, THE Módulo_Interfaz SHALL mostrar la imagen original con cada Región_de_Texto resaltada mediante un rectángulo delimitador de color distinto al fondo de la imagen.
5. WHEN el Pipeline está procesando una Imagen_de_Entrada, THE Módulo_Interfaz SHALL mostrar un indicador de progreso visible; y WHEN el Pipeline completa el procesamiento o produce un error, THE Módulo_Interfaz SHALL ocultar o desactivar dicho indicador.
6. IF el Pipeline produce un error en cualquier etapa, THEN THE Módulo_Interfaz SHALL mostrar un mensaje de error que identifique la etapa donde ocurrió el fallo (por ejemplo, "Error en OCR", "Error en traducción") sin incluir trazas de pila, rutas de archivo ni nombres de variables internas del sistema.

---

### Requisito 6: Orquestación del Pipeline

**User Story:** Como desarrollador, quiero que el sistema coordine los módulos de forma ordenada y desacoplada, para facilitar el mantenimiento y la extensión del pipeline.

#### Criterios de Aceptación

1. THE Módulo_Orquestador SHALL ejecutar el Pipeline en el siguiente orden: (1) Módulo_OCR, (2) Módulo_Traducción, (3) construcción del Resultado_de_Traducción; y no SHALL ejecutar una etapa posterior si la etapa anterior no ha completado exitosamente.
2. THE Módulo_Orquestador SHALL encapsular la lógica de coordinación en la clase `TraductorDeImagenes` con los métodos `cargar_imagen`, `detectar_texto`, `traducir` y `procesar`.
3. IF cualquier módulo del Pipeline lanza una excepción, THEN THE Módulo_Orquestador SHALL capturar la excepción, registrar en el log del sistema una entrada que incluya el nombre de la etapa, el tipo de excepción y el mensaje de error, y retornar un Resultado_de_Traducción cuyo campo de error contenga una descripción no vacía del fallo.
4. THE Módulo_Orquestador SHALL registrar en el log del sistema el tiempo de ejecución en milisegundos de cada una de las tres etapas del Pipeline: Módulo_OCR, Módulo_Traducción y construcción del Resultado_de_Traducción.
5. IF el método `detectar_texto` o `traducir` de `TraductorDeImagenes` es invocado antes de que `cargar_imagen` haya completado exitosamente en la misma sesión de procesamiento, THEN THE Módulo_Orquestador SHALL retornar un Resultado_de_Traducción con el campo de error poblado con el mensaje "Imagen no cargada. Invoque cargar_imagen antes de continuar." sin ejecutar etapas adicionales del Pipeline.

---

### Requisito 7: Arquitectura Modular y Tecnologías Open-Source

**User Story:** Como desarrollador, quiero que el sistema use exclusivamente tecnologías open-source y tenga una arquitectura modular, para garantizar la reproducibilidad y facilitar la sustitución de componentes.

#### Criterios de Aceptación

1. THE Sistema SHALL implementar el Módulo_OCR utilizando únicamente la biblioteca EasyOCR sin dependencias de APIs externas de pago.
2. THE Sistema SHALL implementar el Módulo_Traducción utilizando únicamente modelos de Hugging Face Transformers de la familia Helsinki-NLP/opus-mt o facebook/mbart-large-50, descargados y ejecutados localmente.
3. THE Sistema SHALL implementar el Módulo_Interfaz utilizando únicamente la biblioteca Gradio.
4. THE Sistema SHALL ser ejecutable en un entorno local con Python 3.8 o superior y en Google Colab ejecutando únicamente `pip install -r requirements.txt` seguido de `python app.py` (o la celda equivalente en Colab), sin modificaciones al código fuente.
5. THE Sistema SHALL declarar todas sus dependencias en un archivo `requirements.txt` con versiones fijadas usando el operador `==` para cada paquete.
6. WHERE el entorno de ejecución dispone de una GPU compatible con CUDA, THE Sistema SHALL utilizar aceleración GPU para el Módulo_OCR y el Módulo_Traducción, detectable mediante la variable de entorno o la API de PyTorch `torch.cuda.is_available()`.

---

### Requisito 8: Estructura del Proyecto

**User Story:** Como desarrollador, quiero que el proyecto tenga una estructura de carpetas clara y predecible, para facilitar la navegación y el mantenimiento del código.

#### Criterios de Aceptación

1. THE Sistema SHALL organizar el código fuente en la siguiente estructura de directorios:

```
multimodal-image-translation/
├── app.py                        # Punto de entrada principal (Gradio)
├── requirements.txt              # Dependencias con versiones fijadas
├── README.md                     # Documentación de uso
├── src/
│   ├── __init__.py
│   ├── orchestrator.py           # Clase TraductorDeImagenes
│   ├── ocr_module.py             # Módulo_OCR (EasyOCR)
│   ├── translation_module.py     # Módulo_Traducción (Hugging Face)
│   └── utils.py                  # Funciones auxiliares (dibujo de bboxes, logging)
├── models/                       # Directorio para caché de modelos descargados
│   └── .gitkeep
└── tests/
    ├── __init__.py
    ├── test_ocr_module.py
    ├── test_translation_module.py
    └── test_orchestrator.py
```

2. THE Sistema SHALL mantener cada módulo en su propio archivo Python dentro del directorio `src/`, de forma que ningún archivo en `src/` importe directamente de otro archivo en `src/` que no sea `utils.py`, garantizando que las dependencias entre módulos fluyan únicamente a través del Módulo_Orquestador.

---

### Requisito 9: Esquema de la Clase Principal

**User Story:** Como desarrollador, quiero que la clase `TraductorDeImagenes` tenga una interfaz clara y documentada, para poder integrarla y extenderla fácilmente.

#### Criterios de Aceptación

1. THE Módulo_Orquestador SHALL implementar la clase `TraductorDeImagenes` con los siguientes atributos y métodos:

```python
class TraductorDeImagenes:
    # Atributos
    ocr_reader: easyocr.Reader          # Instancia del lector OCR
    translation_models: dict            # Caché de modelos de traducción cargados {par_idiomas: modelo}
    target_language: str                # Idioma destino seleccionado por el usuario
    confidence_threshold: float         # Umbral mínimo de Confianza_OCR (default: 0.3)
    use_gpu: bool                       # Indica si se usa aceleración GPU

    # Métodos
    def __init__(self, target_language: str, use_gpu: bool = False) -> None
    def cargar_imagen(self, image_path: str) -> np.ndarray
    def detectar_texto(self, image: np.ndarray) -> list[dict]
    def traducir(self, text: str, source_lang: str) -> str
    def procesar(self, image_path: str) -> dict
```

2. THE Módulo_Orquestador SHALL documentar cada método de `TraductorDeImagenes` con docstrings que incluyan: (a) una descripción de una línea del propósito del método, (b) la sección `Args:` con el nombre, tipo y descripción de cada parámetro, (c) la sección `Returns:` con el tipo y descripción del valor retornado, y (d) la sección `Raises:` con el tipo y condición de cada excepción que el método puede lanzar.

---

### Requisito 10: Plan de Implementación por Fases

**User Story:** Como desarrollador, quiero que el proyecto se implemente en fases incrementales, para poder validar cada componente antes de integrarlo con los demás.

#### Criterios de Aceptación

1. THE Sistema SHALL implementarse siguiendo las fases en el orden indicado:

   - **Fase 1 – Configuración del entorno**: Crear la estructura de directorios, el archivo `requirements.txt` e instalar las dependencias.
   - **Fase 2 – Módulo OCR**: Implementar y validar `ocr_module.py` de forma aislada con imágenes de prueba.
   - **Fase 3 – Módulo de Traducción**: Implementar y validar `translation_module.py` de forma aislada con textos de prueba.
   - **Fase 4 – Orquestador**: Implementar `orchestrator.py` integrando los módulos de las fases 2 y 3.
   - **Fase 5 – Interfaz Gradio**: Implementar `app.py` conectando el Módulo_Orquestador con el Módulo_Interfaz.
   - **Fase 6 – Pruebas y ajuste**: Ejecutar los tests unitarios e integración, corregir errores y ajustar umbrales.

2. WHEN se completa cada fase, THE Sistema SHALL ser ejecutable de forma independiente, entendiéndose por "ejecutable de forma independiente" que el módulo o conjunto de módulos de esa fase puede invocarse desde la línea de comandos o un notebook sin errores de importación ni excepciones no controladas, produciendo una salida observable (texto en consola, archivo generado o interfaz activa).

---

### Requisito 11: Criterios de Aceptación del Proyecto

**User Story:** Como usuario y desarrollador, quiero tener criterios claros para determinar cuándo el proyecto está completo y funcional.

#### Criterios de Aceptación

1. THE Sistema SHALL procesar correctamente una imagen de prueba que contenga texto en inglés y producir su traducción al español en menos de 60 segundos medidos desde que el Módulo_Orquestador recibe la imagen hasta que el Resultado_de_Traducción está disponible, en un entorno sin GPU.
2. THE Sistema SHALL procesar correctamente una imagen de prueba que contenga texto en japonés o chino y producir su traducción al español en menos de 120 segundos en un entorno sin GPU.
3. WHEN el Pipeline completa el procesamiento, THE Módulo_Interfaz SHALL mostrar la imagen original con al menos un rectángulo delimitador visible sobre cada Región_de_Texto detectada.
4. WHEN el Pipeline completa el procesamiento, THE Módulo_Interfaz SHALL mostrar el texto original y el texto traducido en componentes de texto separados, cada uno con su etiqueta identificadora ("Texto original" y "Traducción" respectivamente).
5. THE Sistema SHALL completar la instalación de dependencias y el inicio de la interfaz Gradio en Google Colab ejecutando únicamente `!pip install -r requirements.txt` seguido de `!python app.py`, sin que se produzca ninguna excepción no controlada durante el arranque.
6. IF se proporciona una imagen sin texto detectable, THEN THE Módulo_Interfaz SHALL mostrar el mensaje "No se detectó texto en la imagen" en el área de resultados de la interfaz Gradio.
7. THE Sistema SHALL superar al menos el 80% de los casos de prueba definidos en los archivos del directorio `tests/`, donde "superar" significa que el caso de prueba termina con estado `PASSED` al ejecutar `pytest tests/`.
