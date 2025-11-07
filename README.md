# A* Game World - Python Simulator

A smooth and interactive **A* pathfinding simulator** in Python using **Pygame**.  
Place terrains, walls, start/end points, and watch an agent navigate optimally across a grid.

---

## **Features**

- Grid-based environment with customizable **terrain types**:
  - Road (low cost)
  - Grass (medium cost)
  - Mud (high cost)
  - Wall (impassable)
- Click to set **start and end points**.
- Click-and-drag to paint terrain or obstacles.
- Real-time **agent movement** along the computed path.
- Supports **diagonal movement** and multiple heuristics:
  - Manhattan
  - Euclidean
  - Chebyshev
- **Save/load** grid configurations.
- Adjustable **agent speed**.
- Smooth visual path highlighting.

## Future Enhancements

Multiple agents navigating simultaneously.

Weighted terrain with random dynamic obstacles.

Customizable grid size and colors.

Export path data to CSV or JSON for analysis.

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/ihnk-dbg/pathF.git
cd pathF
