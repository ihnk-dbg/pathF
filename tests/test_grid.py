import pytest
from src.pathF import Grid, TERRAINS, DEFAULT_TERRAIN

def test_grid_bounds():
    grid = Grid(5,5)
    assert grid.in_bounds(0,0)
    assert grid.in_bounds(4,4)
    assert not grid.in_bounds(-1,0)
    assert not grid.in_bounds(5,5)

def test_set_terrain_valid():
    grid = Grid(3,3)
    grid.set_terrain(1,1,1)  # Grass
    assert grid.grid[1][1] == 1

def test_set_terrain_invalid_id():
    grid = Grid(3,3)
    grid.set_terrain(1,1,999)  # invalid terrain
    assert grid.grid[1][1] == DEFAULT_TERRAIN

def test_set_terrain_on_start_end():
    grid = Grid(3,3)
    grid.start = (1,1)
    grid.end = (2,2)
    grid.set_terrain(1,1,9)  # wall on start
    grid.set_terrain(2,2,9)  # wall on end
    assert grid.grid[1][1] == DEFAULT_TERRAIN
    assert grid.grid[2][2] == DEFAULT_TERRAIN
