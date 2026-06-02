"""
Punto de entrada principal – Interfaz web Gradio para el Sistema de Traducción
Multimodal de Imágenes.

Responsabilidades:
- Validar el formato MIME y el tamaño de la imagen cargada por el usuario.
- Presentar el selector de idioma destino (disponible antes de cargar imagen).
- Invocar el orquestador (TraductorDeImagenes) y mostrar los resultados.
- Mostrar indicador de progreso durante el procesamiento.
- Mostrar mensajes de error amigables sin stack traces.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

# Necesario para compatibilidad con protobuf >= 4.x y el tokenizer de mBART.
# Sin esto, sentencepiece lanza "Descriptors cannot be created directly."
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gradio as gr
import numpy as np

from src.orchestrator import TraductorDeImagenes
from src.translation_module import IDIOMAS_SOPORTADOS
from src import utils

logger = logging.getLogger(__name__)

# Mensajes de error definidos en los requisitos
_MSG_FORMATO_NO_SOPORTADO = (
    "Formato no soportado. Por favor, sube una imagen en formato JPEG, PNG o BMP."
)
_MSG_TAMANO_EXCEDIDO = (
    "La imagen supera el límite de 10 MB. Por favor, sube una imagen más pequeña."
)
_LIMITE_BYTES = 10 * 1024 * 1024  # 10 MB


def validar_imagen(archivo) -> Tuple[bool, str]:
    """Valida el formato MIME y el tamaño del archivo de imagen.

    Verifica el tipo MIME real del archivo (no solo la extensión) para
    JPEG, PNG y BMP, y comprueba que el tamaño no supere los 10 MB.

    Args:
        archivo: Ruta al archivo (str) o un objeto de archivo con atributo
                 'name' que apunta a la ruta del archivo temporal de Gradio.

    Returns:
        Tupla (es_valida, mensaje_error). Si es válida, mensaje_error es "".
    """
    import os

    # Resolver la ruta del archivo
    if isinstance(archivo, str):
        ruta = archivo
    elif hasattr(archivo, "name"):
        ruta = archivo.name
    else:
        # Intentar convertir a str como último recurso
        ruta = str(archivo)

    # Verificar tamaño del archivo (antes de leer magic bytes para eficiencia)
    try:
        tamano = os.path.getsize(ruta)
    except OSError as exc:
        logger.warning("No se pudo obtener el tamaño del archivo: %s", exc)
        return False, _MSG_FORMATO_NO_SOPORTADO

    if tamano > _LIMITE_BYTES:
        return False, _MSG_TAMANO_EXCEDIDO

    # Leer los primeros bytes para detectar el tipo MIME por magic bytes
    # JPEG: FF D8 FF  (3 bytes)
    # PNG:  89 50 4E 47 0D 0A 1A 0A  (8 bytes)
    # BMP:  42 4D  (2 bytes)
    try:
        with open(ruta, "rb") as f:
            cabecera = f.read(8)
    except OSError as exc:
        logger.warning("No se pudo leer el archivo para validación MIME: %s", exc)
        return False, _MSG_FORMATO_NO_SOPORTADO

    if len(cabecera) < 2:
        return False, _MSG_FORMATO_NO_SOPORTADO

    es_jpeg = cabecera[:3] == b"\xff\xd8\xff"
    es_png = cabecera[:8] == b"\x89PNG\r\n\x1a\n"
    es_bmp = cabecera[:2] == b"BM"

    if not (es_jpeg or es_png or es_bmp):
        return False, _MSG_FORMATO_NO_SOPORTADO

    return True, ""


def procesar_imagen(
    imagen: np.ndarray,
    idioma_destino: str,
) -> Tuple[Optional[np.ndarray], str, str, str, str]:
    """Callback principal de Gradio. Procesa la imagen y retorna los resultados.

    Valida que se haya seleccionado un idioma destino, instancia
    TraductorDeImagenes, invoca procesar() y dibuja los bounding boxes.

    Args:
        imagen: Imagen cargada por el usuario como np.ndarray.
        idioma_destino: Nombre del idioma destino seleccionado en el selector.

    Returns:
        Tupla (imagen_anotada, texto_original, idioma_origen,
               texto_traducido, mensaje_error).
    """
    import tempfile
    import os
    import cv2
    from src.ocr_module import RegionTexto

    # Validar que se ha proporcionado una imagen
    if imagen is None:
        return (
            None,
            "",
            "",
            "",
            "Por favor, carga una imagen antes de traducir.",
        )

    # Req 1.6: Validar que se ha seleccionado un idioma destino
    if not idioma_destino:
        return (
            None,
            "",
            "",
            "",
            "Por favor, selecciona un idioma destino antes de continuar.",
        )

    # Obtener el código ISO del idioma destino a partir del nombre de display
    info_idioma = IDIOMAS_SOPORTADOS.get(idioma_destino)
    if info_idioma is None:
        return (
            None,
            "",
            "",
            "",
            "Por favor, selecciona un idioma destino antes de continuar.",
        )
    codigo_iso = info_idioma["iso"]

    # Guardar el np.ndarray en un archivo temporal para pasarlo al orquestador
    # (TraductorDeImagenes.procesar() espera una ruta de archivo)
    archivo_temp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            archivo_temp = f.name

        # Gradio entrega imágenes en RGB; OpenCV trabaja en BGR para guardar
        imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
        cv2.imwrite(archivo_temp, imagen_bgr)

        # Instanciar el orquestador e invocar el pipeline completo
        traductor = TraductorDeImagenes(target_language=codigo_iso)
        resultado = traductor.procesar(archivo_temp)

    except Exception as exc:
        logger.error("Error inesperado en procesar_imagen: %s", exc)
        return (None, "", "", "", f"Error inesperado: {exc}")
    finally:
        # Limpiar el archivo temporal
        if archivo_temp and os.path.exists(archivo_temp):
            try:
                os.remove(archivo_temp)
            except OSError:
                pass

    # Extraer campos del resultado (procesar() retorna un dict)
    texto_original: str = resultado.get("texto_original", "")
    idioma_origen: str = resultado.get("idioma_origen", "")
    texto_traducido: str = resultado.get("texto_traducido", "")
    regiones_raw: list = resultado.get("regiones", [])
    error: str = resultado.get("error", "")

    # Req 5.6: Si hay error, devolverlo sin stack traces ni detalles internos
    mensaje_error = error if error else ""

    # Convertir las regiones (dicts) a objetos RegionTexto para dibujar_bboxes
    regiones_obj = [
        RegionTexto(bbox=r["bbox"], texto=r["texto"], confianza=r["confianza"])
        for r in regiones_raw
        if isinstance(r, dict)
    ]

    # Req 5.4: Dibujar bounding boxes sobre la imagen original
    # dibujar_bboxes espera BGR; la imagen de Gradio llega en RGB
    imagen_bgr_anotada = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    imagen_anotada_bgr = utils.dibujar_bboxes(imagen_bgr_anotada, regiones_obj)
    # Convertir de vuelta a RGB para que Gradio la muestre correctamente
    imagen_anotada = cv2.cvtColor(imagen_anotada_bgr, cv2.COLOR_BGR2RGB)

    # Req 5.1, 5.2, 5.3: Retornar la tupla con todos los campos
    return (imagen_anotada, texto_original, idioma_origen, texto_traducido, mensaje_error)


def construir_interfaz() -> gr.Blocks:
    """Construye y retorna el objeto gr.Blocks con todos los componentes Gradio.

    Componentes incluidos:
    - Carga de imagen (gr.Image).
    - Selector de idioma destino con los 9 idiomas soportados.
    - Indicador de progreso durante el procesamiento.
    - Imagen anotada con bounding boxes.
    - Texto original con etiqueta e idioma detectado.
    - Texto traducido con etiqueta e idioma destino.

    Returns:
        Objeto gr.Blocks configurado y listo para lanzar.
    """
    with gr.Blocks(title="Traductor Multimodal de Imágenes") as interfaz:
        gr.Markdown("# Traductor Multimodal de Imágenes")
        gr.Markdown(
            "Carga una imagen con texto, selecciona el idioma destino y pulsa **Traducir**."
        )

        with gr.Row():
            with gr.Column():
                # Selector de idioma destino — disponible antes de cargar imagen (Req 1.5)
                idioma_destino = gr.Dropdown(
                    choices=list(IDIOMAS_SOPORTADOS.keys()),
                    value="Español",
                    label="Idioma destino",
                )
                # Componente de carga de imagen
                imagen_entrada = gr.Image(
                    type="numpy",
                    label="Imagen de entrada",
                )
                boton_traducir = gr.Button("Traducir", variant="primary")

            with gr.Column():
                # Imagen anotada con bounding boxes (Req 5.4)
                imagen_anotada = gr.Image(
                    label="Imagen anotada",
                    interactive=False,
                )

        with gr.Row():
            with gr.Column():
                # Texto original con idioma detectado (Req 5.1, 5.2)
                texto_original = gr.Textbox(
                    label="Texto original",
                    lines=5,
                    interactive=False,
                )
                idioma_detectado = gr.Textbox(
                    label="Idioma detectado",
                    interactive=False,
                )
            with gr.Column():
                # Texto traducido con idioma destino (Req 5.1, 5.3)
                texto_traducido = gr.Textbox(
                    label="Traducción",
                    lines=5,
                    interactive=False,
                )
                mensajes = gr.Textbox(
                    label="Mensajes",
                    interactive=False,
                )

        # Conectar el botón al callback procesar_imagen.
        # Gradio Blocks muestra automáticamente un indicador de carga (spinner)
        # en el botón mientras el evento está en curso (Req 5.5).
        boton_traducir.click(
            fn=procesar_imagen,
            inputs=[imagen_entrada, idioma_destino],
            outputs=[imagen_anotada, texto_original, idioma_detectado, texto_traducido, mensajes],
        )

    return interfaz


if __name__ == "__main__":
    utils.configurar_logging()
    interfaz = construir_interfaz()
    interfaz.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
    )
