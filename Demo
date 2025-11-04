#!/usr/bin/env python3
"""
Demo: Asignación de horarios FIC-UDP con grafos y DSATUR + mejora local
-----------------------------------------------------------------------------
• Modela vértices como (curso, sección, actividad) con docente, capacidad, requisitos
  de sala y pertenencia a grupos de conflicto curricular (para evitar topes).
• Colores = (franja_horaria, tipo_sala). Cada color tiene capacidad (número de salas
  disponibles de ese tipo) y cada tipo de sala tiene una capacidad mínima en cupos.
• Algoritmo: DSATUR factible con restricciones duras + post‑mejora local para reducir
  “ventanas” (soft constraint) sobre grupos de conflicto.

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
  - Penalización por ventanas (menor es mejor)

Nota: Este es un prototipo didáctico. Para instancias grandes, conviene persistir y
      ampliar las heurísticas/metaheurísticas (Tabu Search, etc.).
"""

from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict, Counter
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
                 vertices: List[Vertex]):
        self.slots = slots
        self.room_types = room_types
        self.groups = groups
        self.vertices = vertices

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
                return None  # fallo: no hay color factible (se podría backtrackear)

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
    # Métrica soft: penalización por ventanas en grupos
    # ------------------------------
    def windows_penalty(self, assignment: Dict[int, Tuple[int, str]]) -> int:
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

    # ------------------------------
    # Mejora local simple (1‑mover) para bajar ventanas
    # ------------------------------
    def improve_assignment(self,
                           assignment: Dict[int, Tuple[int, str]],
                           iters: int = 200) -> Dict[int, Tuple[int, str]]:
        current = dict(assignment)
        best_pen = self.windows_penalty(current)
        used_color_count: Counter = Counter(current.values())

        for _ in range(iters):
            vids = list(range(len(self.vertices)))
            random.shuffle(vids)
            improved = False
            for v in vids:
                old_color = current[v]
                # construir "vecinos->slots prohibidos" dinámicamente
                neighbor_slots = {current[u][0] for u in self.adj[v] if u in current}

                # probar colores alternativos
                candidates = []
                for color in self.colors:
                    slot_idx, rt = color
                    if slot_idx in neighbor_slots:
                        continue
                    if not self.room_type_compatible(self.vertices[v], rt):
                        continue
                    if color == old_color:
                        continue
                    if used_color_count[color] >= self.room_types[rt].count:
                        continue
                    candidates.append(color)

                random.shuffle(candidates)
                for new_color in candidates[:10]:  # limitar branching
                    # mover temporalmente
                    current[v] = new_color
                    used_color_count[old_color] -= 1
                    used_color_count[new_color] += 1

                    pen = self.windows_penalty(current)
                    if pen < best_pen:
                        best_pen = pen
                        improved = True
                        break
                    else:
                        # revertir
                        used_color_count[new_color] -= 1
                        used_color_count[old_color] += 1
                        current[v] = old_color

                if improved:
                    break
            if not improved:
                break
        return current


# ------------------------------
# Utilidades de E/S y demo
# ------------------------------

def demo_instance() -> Tuple[List[str], Dict[str, RoomType], Dict[str, List[str]], List[Vertex]]:
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

    return slots, room_types, groups, sections


def load_from_json(path: str) -> Tuple[List[str], Dict[str, RoomType], Dict[str, List[str]], List[Vertex]]:
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

    return slots, room_types, groups, sections


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
    parser = argparse.ArgumentParser(description="Demo de asignación de horarios UDP")
    parser.add_argument('--demo', action='store_true', help='Ejecutar con datos de ejemplo')
    parser.add_argument('--input', type=str, default=None, help='Ruta a JSON de entrada')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--improve', type=int, default=200, help='Iteraciones de mejora local (0 para desactivar)')
    args = parser.parse_args()

    if args.demo:
        slots, room_types, groups, sections = demo_instance()
    elif args.input:
        slots, room_types, groups, sections = load_from_json(args.input)
    else:
        print("Debe usar --demo o --input <archivo.json>")
        return

    model = SchedulingModel(slots, room_types, groups, sections)

    assignment = model.dsatur_schedule(seed=args.seed)
    if assignment is None:
        print("No se encontró asignación factible con DSATUR (probar otra semilla o más salas).")
        return

    base_pen = model.windows_penalty(assignment)
    if args.improve > 0:
        improved = model.improve_assignment(assignment, iters=args.improve)
        imp_pen = model.windows_penalty(improved)
        if imp_pen <= base_pen:
            assignment = improved
            base_pen = imp_pen

    # Reporte
    hard_ok = True
    # Verificar restricciones duras: (1) vecinos en distinto slot, (2) capacidad de color
    used_count = Counter(assignment.values())
    for color, u in used_count.items():
        slot_idx, rt = color
        if u > model.room_types[rt].count:
            hard_ok = False
            break
    for v in range(len(sections)):
        for u in model.adj[v]:
            if u < v:
                continue
            if assignment[v][0] == assignment[u][0]:  # mismo slot (tope)
                hard_ok = False
                break
    print(f"\nRestricciones duras OK: {hard_ok}")
    print(f"Penalización por ventanas (menor es mejor): {base_pen}")

    print_schedule(model, assignment)


if __name__ == '__main__':
    main()
