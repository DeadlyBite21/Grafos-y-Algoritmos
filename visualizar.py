#!/usr/bin/env python3
"""
Generador de Vista de Horario (Versión 2.0)
---------------------------------------------
Lee 'horario_generado.json' y crea 'horario_generado.png'
usando Matplotlib para control total sobre el estilo.
- Columnas con días completos (Lunes, Martes, etc.)
- Celdas de tamaño uniforme.
- Contenido de celda: Sala, Profesor y Asignatura.

Uso:
$ python generar_imagen.py horario_generado.json
"""

import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import webbrowser
import os
from collections import defaultdict

# --- Definiciones de Formato ---

# 1. Mapeo de iniciales a días completos
DIAS_MAP = {
    'L': 'Lunes',
    'M': 'Martes',
    'W': 'Miércoles',
    'J': 'Jueves',
    'V': 'Viernes'
}
# 2. Orden de los días
DIAS_ORDEN = ['L', 'M', 'W', 'J', 'V']

def format_event(vertex_data: dict, room_type: str) -> str:
    """
    Crea la etiqueta para una celda con el formato solicitado:
    Sala, Profesor y Asignatura (Curso-Sección-Actividad).
    """
    prof = vertex_data.get('teacher', 'N/A')
    course = vertex_data.get('course', 'N/A')
    sec = vertex_data.get('section', 'N/A')
    act = vertex_data.get('activity', 'N/A')

    # Limitar el largo del nombre del profesor para que quepa
    if len(prof) > 20:
        prof = prof.split(' ')[0] + " " + prof.split(' ')[-1][0] + "."

    return (
        f"Sala: {room_type}\n"
        f"Prof: {prof}\n"
        f"Asig: {course}-S{sec}-{act}"
    )

def main():
    if len(sys.argv) != 2:
        print(f"Uso: python {sys.argv[0]} <archivo_json_horario>")
        return

    input_file = sys.argv[1]
    output_image = "horario_generado.png"

    # --- 1. Cargar datos ---
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{input_file}'")
        return

    slots = data['slots']
    vertices = data['vertices']
    assignment = {int(k): v for k, v in data['assignment'].items()}

    # --- 2. Definir Estructura y Mapear Datos ---
    try:
        bloques_set = set(int(s[1:]) for s in slots)
    except Exception:
        print("Error: El formato de 'slots' no es el esperado (ej. 'L1').")
        return
        
    BLOQUES = sorted(list(bloques_set))
    bloques_labels = [f"Bloque {b}" for b in BLOQUES]
    dias_labels = [DIAS_MAP[d] for d in DIAS_ORDEN]

    # Mapea (Día, Bloque) -> Lista de strings formateados
    schedule_map = defaultdict(list)
    for vidx, (slot_idx, room_type) in assignment.items():
        slot_name = slots[slot_idx]
        dia = slot_name[0]
        bloque = int(slot_name[1:])
        
        vertex_data = vertices[vidx]
        event_str = format_event(vertex_data, room_type)
        schedule_map[(dia, bloque)].append(event_str)

    # --- 3. Preparar Matriz de Texto para Matplotlib ---
    cell_text = []
    for b in BLOQUES:
        row = []
        for d in DIAS_ORDEN:
            events = schedule_map.get((d, b), [])
            
            if len(events) == 0:
                # Celda vacía (con 3 líneas para altura uniforme)
                row.append('\n\n')
            else:
                # **CAMBIO**: En lugar de marcar un error si hay más de 1 evento,
                # los unimos con un separador. Esto es correcto porque
                # el algoritmo los asignó a diferentes tipos de sala.
                row.append("\n------------------\n".join(events))
        cell_text.append(row)

    # --- 4. Generar la Imagen con Matplotlib ---
    print(f"Generando imagen del horario en '{output_image}'...")
    
    # Ajustar el tamaño de la figura (ancho, alto en pulgadas)
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis('off') # Ocultar ejes x/y

    # Crear la tabla
    table = ax.table(
        cellText=cell_text,
        rowLabels=bloques_labels,
        colLabels=dias_labels,
        loc='center',
        cellLoc='center'
    )
    
    # --- 5. Estilizar la Tabla ---
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Aplicar estilos a cada celda
    cells = table.get_celld()
    for (row, col), cell in cells.items():
        # **Esta es la clave para el tamaño uniforme**
        cell.set_height(0.12) # Forzar altura de celda
        
        if row == 0: # Fila de cabecera (Días)
            cell.set_facecolor('#E0E0E0') # Gris claro
            cell.set_text_props(weight='bold')
            cell.set_height(0.06)
        elif col == -1: # Columna de cabecera (Bloques)
            cell.set_facecolor('#E0E0E0')
            cell.set_text_props(weight='bold')
            cell.set_width(0.08) # Hacerla más angosta
        else:
            # **Fondo blanco para celdas de datos**
            cell.set_facecolor('white')
            cell.set_width(0.18) # Ancho uniforme para columnas
            
            # **CAMBIO**: Eliminamos la lógica que pintaba de rojo
            # la celda, ya que no es un error.
            # if '[MÚLTIPLES ASIGNACIONES]' in cell.get_text().get_text():
            #     cell.set_text_props(color='red')
    
    try:
        # Guardar la figura
        plt.savefig(output_image, dpi=200, bbox_inches='tight', pad_inches=0.1)
        print("¡Imagen generada exitosamente!")

        # --- 6. Abrir la Imagen ---
        filepath = os.path.abspath(output_image)
        webbrowser.open_new_tab('file://' + filepath)
        print(f"Abriendo '{output_image}' en tu visor...")

    except Exception as e:
        print(f"Error al generar o guardar la imagen: {e}")
        print("Asegúrate de tener 'matplotlib' y 'pandas' instalados.")

if __name__ == '__main__':
    main()