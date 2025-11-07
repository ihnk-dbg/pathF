from src.pathF import Grid, astar_search

def test_simple_path_diagonal():
    grid = Grid(3,3)
    grid.start = (0,0)
    grid.end = (2,2)
    path = astar_search(grid, diagonal=True, heuristic_mode="manhattan")
    assert path[0] == grid.start
    assert path[-1] == grid.end
    assert len(path) >= 2

def test_no_path_blocked():
    grid = Grid(3,3)
    grid.start = (0,0)
    grid.end = (2,2)
    # Surround end with walls
    for r in range(1,3):
        for c in range(1,3):
            grid.set_terrain(r,c,9)
    path = astar_search(grid, diagonal=True, heuristic_mode="manhattan")
    assert path == []

def test_astar_without_start_end():
    grid = Grid(3,3)
    path = astar_search(grid, diagonal=True, heuristic_mode="manhattan")
    assert path == []
