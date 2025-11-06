#!/usr/bin/env python3
"""
Demo: Asignación de horarios FIC-UDP con grafos y DSATUR + Búsqueda Tabú
-----------------------------------------------------------------------------
• Modela vértices como (curso, sección, actividad) con docente, capacidad, requisitos
  de sala y pertenencia a grupos de conflicto curricular (para evitar topes).
• Colores = (franja_horaria, tipo_sala). Cada color tiene capacidad (número de salas
  disponibles de ese tipo) y cada tipo de sala tiene una capacidad mínima en cupos.
• Algoritmo: 
  1. DSATUR factible para encontrar una solución inicial que respete
     restricciones duras (docentes, grupos, capacidad de salas, disponibilidad docente).
  2. Búsqueda Tabú (Tabu Search) para optimizar restricciones blandas
     (reducir “ventanas” y clases en días consecutivos).

Cómo usar
---------
1) Ejecutar en modo demo (datos embebidos):
   $ python demo_horarios_udp.py --demo

2) Ejecutar con un JSON de entrada:
   $ python demo_horarios_udp.py --input data.json

   Formato JSON (mínimo):
   {
     "slots": ["L1", "L2", ..., "V7"],                 // 35 bloques (etiquetas libres)
     "room_types": {
       "Sala_>60": {"count": 2, "capacity": 80},      // número de salas y cupos
       "Lab_PC":   {"count": 3, "capacity": 40},
       "Sala_<=40": {"count": 4, "capacity": 40}
     },
     "groups": {                                        // grupos de conflicto (listas de cursos)
       "PlanComun_3": ["CalculoIII", "CalorOndas", "Resistencia"],
       "Rezagados_A": ["Algebra", "EstructurasDatos"]
     },
     "teacher_constraints": {                           // [NUEVO] Slots (índices) donde el docente NO está disponible
       "Soto": [0, 1, 34],                              // Ej: Soto no puede L1, L2, V7
       "Diaz": []
     },
     "sections": [                                      // vértices
       {
         "course": "CalculoIII", "section": "1", "activity": "Catedra",
         "teacher": "Soto", "room_type": "Sala_>60", "seats": 60
       },
       ...
     ]
   }

Salida
------
• Imprime el horario agrupado por (slot, tipo sala) y métricas:
  - Cumplimiento de restricciones duras
  - Penalización por ventanas (total y promedio por grupo)
  - Penalización total (incluyendo días consecutivos)

Nota: Este es un prototipo didáctico. Para instancias grandes, conviene persistir y
      ampliar las heurísticas/metaheurísticas.
"""

from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

# ------------------------------
# Datos / Estructuras de dominio
# ------------------------------

@dataclass(frozen=True)
class Vertex:
    course: str
    section: str
    activity: str  # Catedra / Lab / Ayudantia
    teacher: str
    room_type_req: str
    seats: int

    def key(self) -> str:
        return f"{self.course}-S{self.section}-{self.activity}"


@dataclass
class RoomType:
    count: int       # cuántas salas disponibles de este tipo
    capacity: int    # cupos que soporta el tipo de sala


# ------------------------------
# Construcción del grafo
# ------------------------------

class SchedulingModel:
    def __init__(self,
                 slots: List[str],
                 room_types: Dict[str, RoomType],
                 groups: Dict[str, List[str]],  # grupo -> lista de cursos que no pueden toparse
                 vertices: List[Vertex],
                 teacher_constraints: Dict[str, Set[int]]): # Docente -> Set de slot_idx prohibidos
        self.slots = slots
        self.room_types = room_types
        self.groups = groups
        self.vertices = vertices
        self.teacher_constraints = teacher_constraints

        # Colores = (slot_idx, room_type)
        self.colors: List[Tuple[int, str]] = [
            (i, rt) for i in range(len(slots)) for rt in room_types.keys()
        ]

        # Grafo de conflictos duros entre vértices
        self.adj: Dict[int, Set[int]] = defaultdict(set)
        self._build_conflict_graph()

    def _build_conflict_graph(self):
        n = len(self.vertices)
        # Índices por docente
        by_teacher: Dict[str, List[int]] = defaultdict(list)
        for i, v in enumerate(self.vertices):
            by_teacher[v.teacher].append(i)

        # 1) Conflictos por mismo docente
        for teacher, idxs in by_teacher.items():
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    self.adj[a].add(b)
                    self.adj[b].add(a)

        # 2) Conflictos por grupos curriculares (cursos en el mismo grupo no pueden toparse)
        #    Conectamos todos los vértices cuyos 'course' estén dentro del mismo grupo.
        by_course: Dict[str, List[int]] = defaultdict(list)
        for i, v in enumerate(self.vertices):
            by_course[v.course].append(i)

        for gname, course_list in self.groups.items():
            # list of vertex indices that belong to any of the courses in this group
            group_vertices: List[int] = []
            for c in course_list:
                group_vertices.extend(by_course.get(c, []))
            # clique entre ellos
            for i in range(len(group_vertices)):
                for j in range(i + 1, len(group_vertices)):
                    a, b = group_vertices[i], group_vertices[j]
                    self.adj[a].add(b)
                    self.adj[b].add(a)

    # ------------------------------
    # Restricciones duras de factibilidad
    # ------------------------------
    def room_type_compatible(self, v: Vertex, room_type: str) -> bool:
        rt = self.room_types[room_type]
        return rt.capacity >= v.seats

    def teacher_slot_compatible(self, v: Vertex, slot_idx: int) -> bool:
        # [NUEVO] Verifica la restricción de disponibilidad docente
        forbidden_slots = self.teacher_constraints.get(v.teacher, set())
        return slot_idx not in forbidden_slots

    # ------------------------------
    # DSATUR con capacidad por color
    # ------------------------------
    def dsatur_schedule(self, seed: int = 0) -> Optional[Dict[int, Tuple[int, str]]]:
        random.seed(seed)
        n = len(self.vertices)

        # Orden DSATUR: se eligen vértices por mayor grado de saturación
        uncolored: Set[int] = set(range(n))
        color_of: Dict[int, Tuple[int, str]] = {}  # v -> (slot_idx, room_type)

        # capacidad usada por color (slot_idx, room_type) -> usado
        used_color_count: Counter = Counter()

        # Precompute grado
        degrees = {i: len(self.adj[i]) for i in range(n)}

        # Saturation sets
        sat_colors: Dict[int, Set[int]] = {i: set() for i in range(n)}  # set de slots usados por vecinos

        def select_vertex() -> int:
            # elige vértice no coloreado con mayor (|sat_colors|, degree)
            best = None
            best_key = None
            for v in uncolored:
                key = (len(sat_colors[v]), degrees[v], random.random())
                if best_key is None or key > best_key:
                    best_key = key
                    best = v
            return best  # type: ignore

        def feasible_colors(v: int) -> List[Tuple[int, str]]:
            # Un color es factible si:
            # 1) Ningún vecino tiene la misma franja (slot_idx) (conflicto duro)
            # 2) RoomType compatible y no supera la capacidad (count)
            # 3) [NUEVO] Docente disponible en esa franja (slot_idx)
            vtx = self.vertices[v]

            # slots prohibidos por vecinos ya coloreados
            neighbor_slots = set()
            for u in self.adj[v]:
                if u in color_of:
                    neighbor_slots.add(color_of[u][0])

            feasible = []
            for slot_idx, rt in self.colors:
                if slot_idx in neighbor_slots:
                    continue
                if not self.room_type_compatible(vtx, rt):
                    continue
                # [NUEVO] Chequeo de disponibilidad docente
                if not self.teacher_slot_compatible(vtx, slot_idx):
                    continue
                # capacidad: ¿cuántos vértices ya usan (slot, rt)?
                if used_color_count[(slot_idx, rt)] >= self.room_types[rt].count:
                    continue
                feasible.append((slot_idx, rt))

            # Heurística: preferir colores con menor uso para balancear
            feasible.sort(key=lambda c: used_color_count[c])
            return feasible

        while uncolored:
            v = select_vertex()
            options = feasible_colors(v)
            if not options:
                print(f"FALLO DSATUR: No hay color factible para {self.vertices[v].key()}")
                return None  # fallo: no hay color factible

            chosen = options[0]
            color_of[v] = chosen
            used_color_count[chosen] += 1
            uncolored.remove(v)

            # Actualiza saturación de vecinos (slot usado)
            for u in self.adj[v]:
                if u in uncolored:
                    sat_colors[u].add(chosen[0])  # slot_idx

        return color_of

    # ------------------------------
    # Métricas soft: Penalizaciones
    # ------------------------------
    
    def _calculate_windows_penalty(self, assignment: Dict[int, Tuple[int, str]]) -> int:
        # Penaliza “huecos” por grupo de conflicto a nivel de día.
        # slots están en orden, asumimos 7 por día (L1..L7, M1..M7, ...)
        # Definimos día = slot_idx // 7, bloque = slot_idx % 7
        by_course_slots: Dict[str, List[int]] = defaultdict(list)
        for vidx, (slot_idx, _) in assignment.items():
            v = self.vertices[vidx]
            by_course_slots[v.course].append(slot_idx)

        total = 0
        for gname, course_list in self.groups.items():
            # recolectar slots de cursos del grupo
            day_to_blocks: Dict[int, List[int]] = defaultdict(list)
            for c in course_list:
                for s in by_course_slots.get(c, []):
                    day_to_blocks[s // 7].append(s % 7)
            # por día, contar huecos dentro del intervalo [min, max]
            for blocks in day_to_blocks.values():
                blocks = sorted(set(blocks))
                if len(blocks) >= 2:
                    span = blocks[-1] - blocks[0] + 1
                    gaps = span - len(blocks)
                    total += gaps
        return total

    def _calculate_consecutive_days_penalty(self, assignment: Dict[int, Tuple[int, str]]) -> int:
        # [NUEVO] Penaliza si un mismo curso tiene clases en días consecutivos
        by_course_slots: Dict[str, List[int]] = defaultdict(list)
        for vidx, (slot_idx, _) in assignment.items():
            v = self.vertices[vidx]
            by_course_slots[v.course].append(slot_idx)

        total = 0
        num_slots_per_day = 7 # Asumimos 7 bloques por día
        for course, slots in by_course_slots.items():
            if not slots:
                continue
            # Obtenemos los días únicos en que se imparte el curso
            days = sorted(list(set(s // num_slots_per_day for s in slots)))
            if len(days) >= 2:
                for i in range(len(days) - 1):
                    if days[i+1] == days[i] + 1:
                        total += 1 # Penalización por día consecutivo
        return total

    def calculate_total_penalty(self, assignment: Dict[int, Tuple[int, str]]) -> Tuple[int, int]:
        """
        Calcula la penalización total ponderada (función objetivo a minimizar).
        Retorna (penalización_total, penalización_solo_ventanas)
        """
        # Pesos de las restricciones blandas
        W_WINDOWS = 1
        W_CONSECUTIVE = 5 # Damos más peso a no tener días consecutivos

        pen_windows = self._calculate_windows_penalty(assignment)
        pen_consecutive = self._calculate_consecutive_days_penalty(assignment)

        total_penalty = (W_WINDOWS * pen_windows) + (W_CONSECUTIVE * pen_consecutive)
        
        return total_penalty, pen_windows


    # ------------------------------
    # [NUEVO] Búsqueda Tabú para optimización
    # ------------------------------
    def tabu_search_improve(self,
                            assignment: Dict[int, Tuple[int, str]],
                            iters: int = 200,
                            tabu_size: int = 10) -> Dict[int, Tuple[int, str]]:
        """
        Optimiza la asignación usando Búsqueda Tabú para minimizar la penalización total.
        """
        current_assignment = dict(assignment)
        best_assignment = dict(assignment)
        (best_penalty, _) = self.calculate_total_penalty(best_assignment)
        
        used_color_count: Counter = Counter(current_assignment.values())
        
        # Lista Tabú: almacena movimientos (vertice, old_color)
        # Un movimiento (v, c) significa que 'v' no puede volver a 'c'
        tabu_list = deque(maxlen=tabu_size)

        print(f"Iniciando Búsqueda Tabú (Penalización inicial: {best_penalty})...")

        for k in range(iters):
            best_move = None # (v, new_color, new_penalty)
            best_move_penalty = float('inf')

            # Explorar vecinos (1-movimiento)
            vids = list(range(len(self.vertices)))
            random.shuffle(vids)

            for v in vids[:50]: # Explorar un subconjunto aleatorio de vértices por iteración
                old_color = current_assignment[v]
                vtx = self.vertices[v]
                
                # Slots prohibidos por vecinos (hard constraint)
                neighbor_slots = {current_assignment[u][0] for u in self.adj[v] if u in current_assignment}

                # Probar colores alternativos
                candidates = []
                for color in self.colors:
                    slot_idx, rt = color
                    if color == old_color:
                        continue
                    # --- Chequeo de restricciones duras ---
                    if slot_idx in neighbor_slots:
                        continue
                    if not self.room_type_compatible(vtx, rt):
                        continue
                    if not self.teacher_slot_compatible(vtx, slot_idx):
                        continue
                    if used_color_count[color] >= self.room_types[rt].count:
                        continue
                    
                    candidates.append(color)
                
                random.shuffle(candidates)

                for new_color in candidates[:10]: # Limitar branching
                    # --- Evaluar el movimiento ---
                    # 1. Aplicar movimiento temporal
                    current_assignment[v] = new_color
                    (new_penalty, _) = self.calculate_total_penalty(current_assignment)
                    
                    # 2. Revertir
                    current_assignment[v] = old_color 

                    is_tabu = (v, new_color) in tabu_list
                    
                    # Criterio de Aspiración: Si es el mejor global, lo aceptamos aunque sea tabú
                    if new_penalty < best_penalty:
                        best_move = (v, new_color, new_penalty)
                        best_move_penalty = new_penalty
                        break # Encontramos un nuevo óptimo global

                    # Movimiento No Tabú: Lo consideramos si es el mejor de esta iteración
                    elif not is_tabu:
                        if new_penalty < best_move_penalty:
                            best_move = (v, new_color, new_penalty)
                            best_move_penalty = new_penalty
                
                if best_move and best_move_penalty < best_penalty:
                    break # Salir del bucle de vértices si encontramos un nuevo global
            
            if not best_move:
                # No se encontraron movimientos válidos
                break

            # --- Realizar el mejor movimiento encontrado (Tabú o no) ---
            v, new_color, new_penalty = best_move
            old_color = current_assignment[v]

            # Aplicar el movimiento
            current_assignment[v] = new_color
            used_color_count[old_color] -= 1
            used_color_count[new_color] += 1
            
            # Añadir el movimiento *inverso* a la lista tabú
            # (No mover 'v' de vuelta a 'old_color' por un tiempo)
            tabu_list.append((v, old_color))

            # Actualizar el mejor global si es necesario
            if new_penalty < best_penalty:
                best_penalty = new_penalty
                best_assignment = dict(current_assignment)
                print(f"  Iter {k+1}/{iters}: Nueva mejor penalización = {best_penalty}")

        print(f"Búsqueda Tabú finalizada. Penalización final: {best_penalty}")
        return best_assignment


# ------------------------------
# Utilidades de E/S y demo
# ------------------------------

def demo_instance() -> Tuple[List[str], Dict[str, RoomType], Dict[str, List[str]], List[Vertex], Dict[str, Set[int]]]:
    # 35 bloques: L1..L7, M1..M7, W1..W7, J1..J7, V1..V7
    days = ['L', 'M', 'W', 'J', 'V']
    slots = [f"{d}{i}" for d in days for i in range(1, 8)]

    room_types = {
        'Sala_>60': RoomType(count=2, capacity=80),
        'Sala_<=40': RoomType(count=3, capacity=40),
        'Lab_PC': RoomType(count=2, capacity=40),
    }

    groups = {
        'PlanComun_3': ['CalculoIII', 'CalorOndas', 'Resistencia'],
        'Rezagados_A': ['Algebra', 'EstructurasDatos'],
    }

    sections = [
        Vertex('CalculoIII', '1', 'Catedra', 'Soto', 'Sala_>60', 60),
        Vertex('CalculoIII', '1', 'Ayudantia', 'Perez', 'Sala_<=40', 40),
        Vertex('CalorOndas', '1', 'Catedra', 'Rojas', 'Sala_>60', 60),
        Vertex('Resistencia', '1', 'Catedra', 'Soto', 'Sala_>60', 60),   # mismo docente que Cálculo III
        Vertex('Algebra', '2', 'Catedra', 'Diaz', 'Sala_<=40', 35),
        Vertex('EstructurasDatos', '1', 'Catedra', 'Diaz', 'Lab_PC', 35), # mismo docente -> no tope
        Vertex('EstructurasDatos', '1', 'Laboratorio', 'Lagos', 'Lab_PC', 35),
    ]

    # [NUEVO] Restricciones de docentes (mapa de 'teacher_name' -> set de slot_idx prohibidos)
    teacher_constraints = {
        'Soto': {0, 1, 34}, # Soto no puede L1, L2, V7
        'Diaz': {10, 11, 12, 13}, # Diaz no puede M4, M5, M6, M7
        'Perez': set(),
        'Rojas': set(),
        'Lagos': set()
    }

    return slots, room_types, groups, sections, teacher_constraints


def load_from_json(path: str) -> Tuple[List[str], Dict[str, RoomType], Dict[str, List[str]], List[Vertex], Dict[str, Set[int]]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    slots = data['slots']
    room_types = {k: RoomType(**v) for k, v in data['room_types'].items()}
    groups = data.get('groups', {})

    sections = []
    for s in data['sections']:
        sections.append(Vertex(
            course=s['course'],
            section=str(s.get('section', '1')),
            activity=s.get('activity', 'Catedra'),
            teacher=s['teacher'],
            room_type_req=s['room_type'],
            seats=int(s['seats'])
        ))

    # [NUEVO] Cargar restricciones de docentes
    teacher_constraints_raw = data.get('teacher_constraints', {})
    teacher_constraints = {
        t: set(slots_idx) for t, slots_idx in teacher_constraints_raw.items()
    }

    return slots, room_types, groups, sections, teacher_constraints


def print_schedule(model: SchedulingModel, assignment: Dict[int, Tuple[int, str]]):
    slots = model.slots
    # Agrupar por (slot, room_type)
    bucket: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for vidx, color in assignment.items():
        bucket[color].append(vidx)

    print("\n=== HORARIO PROPUESTO ===")
    for slot_idx in range(len(slots)):
        for rt in model.room_types.keys():
            key = (slot_idx, rt)
            if key in bucket:
                print(f"\n[{slots[slot_idx]} | {rt}] (usados {len(bucket[key])}/{model.room_types[rt].count})")
                for vidx in bucket[key]:
                    v = model.vertices[vidx]
                    print(f"  - {v.key()} | Docente: {v.teacher} | Cupos: {v.seats}")


def main():
    parser = argparse.ArgumentParser(description="Demo de asignación de horarios UDP con Búsqueda Tabú")
    parser.add_argument('--demo', action='store_true', help='Ejecutar con datos de ejemplo')
    parser.add_argument('--input', type=str, default=None, help='Ruta a JSON de entrada')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--improve', type=int, default=200, help='Iteraciones de Búsqueda Tabú (0 para desactivar)')
    parser.add_argument('--tabu_size', type=int, default=10, help='Tamaño de la lista Tabú')
    parser.add_argument('--output', type=str, default=None, help='Ruta al archivo JSON de salida donde se guardará el horario')
    args = parser.parse_args()

    if args.demo:
        slots, room_types, groups, sections, teacher_constraints = demo_instance()
    elif args.input:
        slots, room_types, groups, sections, teacher_constraints = load_from_json(args.input)
    else:
        print("Debe usar --demo o --input <archivo.json>")
        return

    model = SchedulingModel(slots, room_types, groups, sections, teacher_constraints)

    print("Ejecutando DSATUR para encontrar solución inicial factible...")
    assignment = model.dsatur_schedule(seed=args.seed)
    if assignment is None:
        print("No se encontró asignación factible con DSATUR (probar otra semilla o más salas/menos restricciones).")
        return

    (base_total_pen, base_windows_pen) = model.calculate_total_penalty(assignment)
    
    if args.improve > 0:
        improved = model.tabu_search_improve(assignment, iters=args.improve, tabu_size=args.tabu_size)
        (imp_total_pen, imp_windows_pen) = model.calculate_total_penalty(improved)
        
        # Usamos la solución mejorada solo si es estrictamente mejor
        if imp_total_pen < base_total_pen:
            assignment = improved
            final_total_pen = imp_total_pen
            final_windows_pen = imp_windows_pen
        else:
            final_total_pen = base_total_pen
            final_windows_pen = base_windows_pen
            print("La Búsqueda Tabú no encontró una solución mejor que la inicial de DSATUR.")
    else:
        final_total_pen = base_total_pen
        final_windows_pen = base_windows_pen

    # --- Reporte Final ---
    hard_ok = True
    # Verificar restricciones duras: (1) vecinos en distinto slot, (2) capacidad de color, (3) disponibilidad docente
    used_count = Counter(assignment.values())
    for color, u in used_count.items():
        if u > model.room_types[color[1]].count:
            print(f"ERROR HARD: Capacidad de color excedida para {color}")
            hard_ok = False
            break
            
    for v in range(len(sections)):
        vtx = model.vertices[v]
        slot_idx, rt = assignment[v]
        
        # Chequeo docente
        if not model.teacher_slot_compatible(vtx, slot_idx):
             print(f"ERROR HARD: {vtx.key()} asignado a docente {vtx.teacher} en slot prohibido {slot_idx}")
             hard_ok = False
             break
        
        # Chequeo vecinos
        for u in model.adj[v]:
            if u < v: continue
            if assignment[v][0] == assignment[u][0]:  # mismo slot (tope)
                print(f"ERROR HARD: Tope entre {vtx.key()} y {model.vertices[u].key()}")
                hard_ok = False
                break
        if not hard_ok: break

    print(f"\n--- MÉTRICAS DE CALIDAD (Menor es mejor) ---")
    print(f"Restricciones duras OK: {hard_ok}")
    
    num_groups = len(model.groups)
    avg_pen = (final_windows_pen / num_groups) if num_groups > 0 else 0
    
    print(f"Penalización por ventanas (total): {final_windows_pen}")
    print(f"Penalización por ventanas (promedio por grupo): {avg_pen:.2f}")
    print(f"Penalización total (con días consecutivos, peso={5}): {final_total_pen}")

    print_schedule(model, assignment)

    if args.output and hard_ok:
        print(f"\nGuardando horario generado en -> {args.output}")
        # Creamos un formato de salida simple para el visualizador
        # Necesita 3 cosas: los slots, los vértices (para ver detalles) y la asignación
        try:
            output_data = {
                "slots": model.slots,
                "vertices": [v.__dict__ for v in model.vertices],
                "assignment": assignment # El dict[int, tuple(int, str)]
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print("Guardado exitosamente.")
        except Exception as e:
            print(f"ERROR al guardar el JSON: {e}")


if __name__ == '__main__':
    main()