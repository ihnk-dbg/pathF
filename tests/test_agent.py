from src.pathF import Agent

def test_agent_movement_simple():
    agent = Agent((0,0), speed_px_s=100)
    agent.set_path([(100,0)])
    agent.update(0.5)  # half a second
    assert agent.x > 0
    assert agent.y == 0

def test_agent_reaches_goal():
    agent = Agent((0,0), speed_px_s=200)
    agent.set_path([(50,0)])
    dt = 0.3
    agent.update(dt)
    agent.update(dt)
    assert agent.at_goal()
