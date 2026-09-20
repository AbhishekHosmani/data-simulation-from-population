from pathlib import Path
import pandas as pd
from src.config import load_archetypes
from src.simulator import EVSimulator, SimulationConfig

ROOT = Path(__file__).resolve().parents[1]

def test_reproducible():
    a = load_archetypes(ROOT/'config'/'archetypes.csv')
    cfg = SimulationConfig(n_agents=30, n_days=5, seed=7)
    e1,_ = EVSimulator(a,cfg).run(); e2,_ = EVSimulator(a,cfg).run()
    pd.testing.assert_frame_equal(e1,e2)

def test_physical_bounds():
    a = load_archetypes(ROOT/'config'/'archetypes.csv')
    e,_ = EVSimulator(a,SimulationConfig(n_agents=50,n_days=7,seed=1)).run()
    assert e.plug_in_soc.between(0,1).all()
    assert e.end_soc.between(0,1).all()
    assert (e.end_soc >= e.plug_in_soc).all()
    assert (e.charging_duration_hours <= e.plug_duration_hours + 1e-9).all()
    assert (e.flexible_hours >= 0).all()
