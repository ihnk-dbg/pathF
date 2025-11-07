"""
astar_game_world_
See inline comments for changes.
"""
import pygame
import heapq
import math
import json
from typing import Tuple, List, Optional, Dict, Set

# ---------- Config ---------- #
WINDOW_W, WINDOW_H = 1000, 720
GRID_ROWS, GRID_COLS = 40, 54  # rectangle grid (rows x cols)
TOOLBAR_W = 240
CELL_GAP = 1
FPS = 60
SAVE_FILE = "world.json"

# Terrain definitions: id: (name, cost, color)
TERRAINS = {
    0: ("Road", 1.0, (200, 200, 200)),
    1: ("Grass", 2.0, (120, 200, 120)),
    2: ("Mud", 5.0, (160, 120, 80)),
    9: ("Wall", math.inf, (40, 40, 40)),  # non-walkable
}
DEFAULT_TERRAIN = 0
# --------------------------- #

pygame.init()
# If the requested font isn't available, SysFont will fall back automatically.

FONT = pygame.font.SysFont("consolas", 16)
BIGFONT = pygame.font.SysFont("consolas", 20)

# Compute cell size to fit grid + toolbar
grid_area_w = WINDOW_W - TOOLBAR_W
CELL_W = max(6, (grid_area_w - (GRID_COLS + 1) * CELL_GAP) // GRID_COLS)
CELL_H = max(6, (WINDOW_H - (GRID_ROWS + 1) * CELL_GAP) // GRID_ROWS)
CELL_SIZE = min(CELL_W, CELL_H)

class Grid:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        # grid holds terrain id ints
        self.grid = [[DEFAULT_TERRAIN for _ in range(cols)] for _ in range(rows)]
        self.start: Optional[Tuple[int, int]] = None
        self.end: Optional[Tuple[int, int]] = None

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_walkable(self, r: int, c: int) -> bool:
        if not self.in_bounds(r, c):
            return False
        tid = self.grid[r][c]
        cost = TERRAINS.get(tid, TERRAINS[DEFAULT_TERRAIN])[1]
        return cost < math.inf

    def set_terrain(self, r: int, c: int, terrain_id: int):
        if not self.in_bounds(r, c):
            return
        # validate terrain id
        if terrain_id not in TERRAINS:
            return
        # don't turn start/end into wall by accident
        if (self.start and (r, c) == self.start) or (self.end and (r, c) == self.end):
            return
        self.grid[r][c] = terrain_id

    def to_dict(self) -> Dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "grid": self.grid,
            "start": list(self.start) if self.start else None,
            "end": list(self.end) if self.end else None,
        }

    def from_dict(self, data: Dict):
        # if sizes match, load directly (with fallback)
        if data.get("rows") == self.rows and data.get("cols") == self.cols:
            self.grid = data.get("grid", self.grid)
            s = data.get("start"); e = data.get("end")
            self.start = tuple(s) if s else None
            self.end = tuple(e) if e else None
        else:
            # adapt to size differences: clip or pad
            g = data.get("grid", [])
            for r in range(self.rows):
                for c in range(self.cols):
                    try:
                        self.grid[r][c] = g[r][c]
                    except Exception:
                        self.grid[r][c] = DEFAULT_TERRAIN

            s = data.get("start"); e = data.get("end")
            self.start = tuple(s) if s and self.in_bounds(*s) else None
            self.end = tuple(e) if e and self.in_bounds(*e) else None


# A* helpers
def heuristic(a: Tuple[int, int], b: Tuple[int, int], mode: str) -> float:

    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])

    if mode == "manhattan":
        return dx + dy
    elif mode == "euclidean":
        return math.hypot(dx, dy)
    elif mode == "chebyshev":
        return max(dx, dy)
    return dx + dy


def neighbors(grid: Grid, node: Tuple[int, int], diagonal: bool):
    r, c = node
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in steps:
        nr, nc = r + dr, c + dc
        if not grid.in_bounds(nr, nc):
            continue
        if not grid.is_walkable(nr, nc):
            continue
        # prevent cutting corners through walls (if diagonal)
        if diagonal and dr != 0 and dc != 0:
            if not (grid.is_walkable(r + dr, c) and grid.is_walkable(r, c + dc)):
                continue
        # cost: terrain cost times distance (sqrt(2) for diagonal)
        base_cost = TERRAINS.get(grid.grid[nr][nc], TERRAINS[DEFAULT_TERRAIN])[1]
        step_dist = math.hypot(dr, dc) if (dr != 0 and dc != 0) else 1.0
        yield (nr, nc), base_cost * step_dist


def astar_search(grid: Grid, diagonal: bool, heuristic_mode: str):
    """Return path (list of (r,c)) or [] if none"""
    start = grid.start
    end = grid.end
    if not start or not end:
        return []

    open_heap = []
    gscore: Dict[Tuple[int, int], float] = {start: 0.0}
    fscore: Dict[Tuple[int, int], float] = {start: heuristic(start, end, heuristic_mode)}
    heapq.heappush(open_heap, (fscore[start], start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed: Set[Tuple[int, int]] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == end:
            # reconstruct
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)
        for nbr, cost in neighbors(grid, current, diagonal):
            tentative_g = gscore[current] + cost
            if tentative_g < gscore.get(nbr, math.inf):
                came_from[nbr] = current
                gscore[nbr] = tentative_g
                fscore[nbr] = tentative_g + heuristic(nbr, end, heuristic_mode)
                heapq.heappush(open_heap, (fscore[nbr], nbr))

    return []

# Convert grid path to pixel center points
def path_to_points(path: List[Tuple[int, int]]):
    pts = []
    for r, c in path:
        x = CELL_GAP + c * (CELL_SIZE + CELL_GAP) + CELL_SIZE // 2
        y = CELL_GAP + r * (CELL_SIZE + CELL_GAP) + CELL_SIZE // 2
        pts.append((x, y))
    return pts

def compute_path_cost(grid: Grid, path: List[Tuple[int, int]]) -> float:
    if not path:
        return 0.0
    total = 0.0
    prev = path[0]
    for cell in path[1:]:
        # movement distance multiplier
        dr = cell[0] - prev[0]
        dc = cell[1] - prev[1]
        dist = math.hypot(dr, dc) if (dr != 0 and dc != 0) else 1.0
        total += TERRAINS.get(grid.grid[cell[0]][cell[1]], TERRAINS[DEFAULT_TERRAIN])[1] * dist
        prev = cell
    return total

class Agent:
    def __init__(self, pos_px: Tuple[float, float], speed_px_s: float = 120.0):
        self.x, self.y = pos_px
        self.speed = speed_px_s
        self.path_points: List[Tuple[float, float]] = []
        self.target_idx = 0
        self.radius = max(4, CELL_SIZE // 2 - 2)

    def set_path(self, pts: List[Tuple[float, float]]):
        self.path_points = pts
        self.target_idx = 0

    def update(self, dt: float):
        if not self.path_points or self.target_idx >= len(self.path_points):
            return
        tx, ty = self.path_points[self.target_idx]
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            self.target_idx += 1
            return
        move = self.speed * dt
        if move >= dist:
            self.x, self.y = tx, ty
            self.target_idx += 1
        else:
            self.x += dx / dist * move
            self.y += dy / dist * move

    def at_goal(self):
        return not self.path_points or self.target_idx >= len(self.path_points)

    def draw(self, screen):
        pygame.draw.circle(screen, (240, 200, 40), (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (180, 140, 20), (int(self.x), int(self.y)), max(2, self.radius // 2))

def draw_toolbar(screen, state, fps):
    # toolbar background
    pygame.draw.rect(screen, (25, 25, 25), (WINDOW_W - TOOLBAR_W, 0, TOOLBAR_W, WINDOW_H))
    x = WINDOW_W - TOOLBAR_W + 12
    y = 10
    lines = [
        f"A* Game World — Controls (FPS: {fps:.0f})",
        f"Terrain: {TERRAINS[state['paint_terrain']][0]}  (1/2/3/W to change)",
        f"Start: {state['grid'].start}  End: {state['grid'].end}",
        f"Diagonal: {state['diagonal']}  Heuristic: {state['heuristic']}",
        f"Agent speed: {state['agent'].speed:.0f}px/s",
        "",
        "LMB drag: paint terrain",
        "RMB click: cycle start->end->clear",
        "SPACE: find path",
        "R: reset path  C: clear",
        "S: save  L: load",
        "Mouse wheel: speed +/-",
    ]
    for i, txt in enumerate(lines):
        surf = FONT.render(txt, True, (230, 230, 230))
        screen.blit(surf, (x, y + i * 20))

def main():
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("A* Game World - Agent simulation")
    clock = pygame.time.Clock()

    grid = Grid(GRID_ROWS, GRID_COLS)
    # initialize agent at top-left cell center
    initial_x = CELL_GAP + CELL_SIZE // 2
    initial_y = CELL_GAP + CELL_SIZE // 2
    agent = Agent((initial_x, initial_y), speed_px_s=140.0)

    state = {
        "paint_terrain": 0,
        "painting": False,
        "brush": 1,
        "placing": None,
        "dragging": False,
        "diagonal": True,
        "heuristic": "manhattan",
        "grid": grid,
        "agent": agent,
        "path_cells": [],
        "path_points": [],
        "show_path": True,
    }

    running = True

    def pixel_to_cell(px: int, py: int):
        """Convert pixel coordinates to grid cell (r,c) or None.
        This correctly accounts for the leading CELL_GAP offset and cell sizes."""
        
        # reject clicks on toolbar area
        if px >= WINDOW_W - TOOLBAR_W or px < 0 or py < 0:
            return None
        
        # relative to grid origin (which starts at CELL_GAP, CELL_GAP)
        px_rel = px - CELL_GAP
        py_rel = py - CELL_GAP

        if px_rel < 0 or py_rel < 0:
            return None
        
        cell_w = CELL_SIZE + CELL_GAP
        c = px_rel // cell_w
        r = py_rel // cell_w

        if grid.in_bounds(int(r), int(c)):
            return int(r), int(c)
        
        return None

    def place_agent_on_cell(cell):
        if not cell:
            return
        r, c = cell
        x = CELL_GAP + c * (CELL_SIZE + CELL_GAP) + CELL_SIZE // 2
        y = CELL_GAP + r * (CELL_SIZE + CELL_GAP) + CELL_SIZE // 2
        agent.x, agent.y = x, y

    while running:
        dt = clock.tick(FPS) / 1000.0
        fps = clock.get_fps()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    # run A*
                    path = astar_search(grid, state["diagonal"], state["heuristic"])
                    state["path_cells"] = path
                    state["path_points"] = path_to_points(path)
                    if path:
                        place_agent_on_cell(path[0])
                        # agent moves to subsequent points (skip first because agent placed there)
                        agent.set_path(state["path_points"][1:])
                        cost = compute_path_cost(grid, path)
                        print(f"Path found with {len(path)} cells, approx cost {cost:.2f}")
                    else:
                        print("No path found.")
                        agent.set_path([])
                elif ev.key == pygame.K_r:
                    state["path_cells"] = []
                    state["path_points"] = []
                    agent.set_path([])
                elif ev.key == pygame.K_c:
                    grid = Grid(GRID_ROWS, GRID_COLS)
                    state["grid"] = grid
                    state["path_cells"] = []
                    state["path_points"] = []
                    agent.set_path([])
                elif ev.key == pygame.K_d:
                    state["diagonal"] = not state["diagonal"]
                elif ev.key == pygame.K_h:
                    modes = ["manhattan", "euclidean", "chebyshev"]
                    idx = (modes.index(state["heuristic"]) + 1) % len(modes)
                    state["heuristic"] = modes[idx]
                elif ev.key == pygame.K_s:
                    try:
                        with open(SAVE_FILE, "w") as f:
                            json.dump(grid.to_dict(), f)
                        print("Saved to", SAVE_FILE)
                    except Exception as e:
                        print("Save failed:", e)
                elif ev.key == pygame.K_l:
                    try:
                        with open(SAVE_FILE, "r") as f:
                            data = json.load(f)
                        grid.from_dict(data)
                        state["grid"] = grid
                        print("Loaded", SAVE_FILE)
                    except Exception as e:
                        print("Load failed:", e)
                elif ev.key == pygame.K_1:
                    state["paint_terrain"] = 0
                elif ev.key == pygame.K_2:
                    state["paint_terrain"] = 1
                elif ev.key == pygame.K_3:
                    state["paint_terrain"] = 2
                elif ev.key == pygame.K_w:
                    state["paint_terrain"] = 9
                elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                    agent.speed = min(500, agent.speed + 20)
                elif ev.key == pygame.K_MINUS:
                    agent.speed = max(20, agent.speed - 20)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                cell = pixel_to_cell(mx, my)
                if ev.button == 1:
                    # start painting (if inside grid)
                    if cell:
                        state["painting"] = True
                        r, c = cell
                        grid.set_terrain(r, c, state["paint_terrain"])
                elif ev.button == 3:
                    # right click: cycle start->end->clear
                    if cell:
                        if grid.start is None:
                            if grid.is_walkable(*cell):
                                grid.start = cell
                                place_agent_on_cell(cell)
                        elif grid.end is None:
                            if grid.is_walkable(*cell) and cell != grid.start:
                                grid.end = cell
                        else:
                            # clear start and/or end depending on where clicked; otherwise set start to clicked and clear end
                            if cell == grid.start:
                                grid.start = None
                            elif cell == grid.end:
                                grid.end = None
                            else:
                                if grid.is_walkable(*cell):
                                    grid.start = cell
                                    grid.end = None
                        # reset old path
                        state["path_cells"] = []
                        state["path_points"] = []
                        agent.set_path([])
                elif ev.button == 4:
                    agent.speed = min(500, agent.speed + 10)
                elif ev.button == 5:
                    agent.speed = max(20, agent.speed - 10)

            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    state["painting"] = False

            elif ev.type == pygame.MOUSEMOTION:
                if state["painting"]:
                    mx, my = ev.pos
                    cell = pixel_to_cell(mx, my)
                    if cell:
                        grid.set_terrain(cell[0], cell[1], state["paint_terrain"])

        # Update agent
        agent.update(dt)

        # Draw background
        screen.fill((18, 18, 22))

        # Draw grid
        for r in range(grid.rows):
            for c in range(grid.cols):
                tid = grid.grid[r][c]
                col = TERRAINS.get(tid, TERRAINS[DEFAULT_TERRAIN])[2]
                x = c * (CELL_SIZE + CELL_GAP) + CELL_GAP
                y = r * (CELL_SIZE + CELL_GAP) + CELL_GAP
                pygame.draw.rect(screen, col, (x, y, CELL_SIZE, CELL_SIZE))
                # lightly hatch mud to differentiate
                if tid == 2:
                    pygame.draw.line(screen, (140, 100, 60),
                                     (x, y + CELL_SIZE // 2), (x + CELL_SIZE, y + CELL_SIZE // 2), 1)

        # Draw start/end
        if grid.start:
            r, c = grid.start
            x = c * (CELL_SIZE + CELL_GAP) + CELL_GAP
            y = r * (CELL_SIZE + CELL_GAP) + CELL_GAP
            pygame.draw.rect(screen, (50, 200, 50), (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))
            surf = FONT.render("S", True, (0, 0, 0))
            screen.blit(surf, (x + 4, y + 2))
        if grid.end:
            r, c = grid.end
            x = c * (CELL_SIZE + CELL_GAP) + CELL_GAP
            y = r * (CELL_SIZE + CELL_GAP) + CELL_GAP
            pygame.draw.rect(screen, (200, 50, 50), (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))
            surf = FONT.render("E", True, (0, 0, 0))
            screen.blit(surf, (x + 4, y + 2))

        # Draw path cells highlight
        if state["path_cells"]:
            for (r, c) in state["path_cells"]:
                x = c * (CELL_SIZE + CELL_GAP) + CELL_GAP
                y = r * (CELL_SIZE + CELL_GAP) + CELL_GAP
                pygame.draw.rect(screen, (255, 230, 150),
                                 (x + CELL_SIZE // 4, y + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2))

        # Draw path lines (smooth)
        if state["path_points"] and len(state["path_points"]) >= 2:
            pygame.draw.lines(screen, (255, 200, 20), False, state["path_points"], max(2, CELL_SIZE // 6))

        # Draw agent last so it's on top
        agent.draw(screen)

        # Draw toolbar
        draw_toolbar(screen, state, fps)

        # Draw small legend of terrain boxes
        tx = WINDOW_W - TOOLBAR_W + 10
        ty = WINDOW_H - 140
        for i, tid in enumerate([0, 1, 2, 9]):
            name, cost, col = TERRAINS[tid]
            pygame.draw.rect(screen, col, (tx, ty + i * 28, 22, 22))
            keyhint = ("1" if tid == 0 else "2" if tid == 1 else "3" if tid == 2 else "W")
            cost_text = f"{cost}" if cost < math.inf else "X"
            label = FONT.render(f"{name} (cost {cost_text}) - press {keyhint}", True, (220, 220, 220))
            screen.blit(label, (tx + 30, ty + i * 28))

        # Draw header + path stats
        header = BIGFONT.render("A* Game World - Agent Walk Simulation", True, (230, 230, 230))
        screen.blit(header, (10, WINDOW_H - 34))
        if state["path_cells"]:
            cost = compute_path_cost(grid, state["path_cells"])
            stats = FONT.render(f"Path length: {len(state['path_cells'])}  Cost: {cost:.2f}", True, (200, 200, 200))
            screen.blit(stats, (10, WINDOW_H - 56))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
