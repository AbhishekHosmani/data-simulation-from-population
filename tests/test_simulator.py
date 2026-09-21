from pathlib import Path
import pandas as pd
from src.config import load_archetypes
from src.simulator import EVSimulator, SimulationConfig

ROOT = Path(__file__).resolve().parents[1]

def test_reproducible():
    a = load_archetypes(ROOT / "config" / "archetypes.csv")
    cfg = SimulationConfig(n_agents=30, n_days=5,seed=7)

    events1, driving1, timeline1 = EVSimulator(a, cfg).run()
    events2, driving2, timeline2 = EVSimulator(a, cfg).run()

    pd.testing.assert_frame_equal(events1, events2)
    pd.testing.assert_frame_equal(driving1, driving2)
    pd.testing.assert_frame_equal(timeline1, timeline2)

def test_physical_bounds():
    a = load_archetypes(ROOT / "config" / "archetypes.csv")
    events, driving, timeline = EVSimulator(a,
        SimulationConfig(n_agents=50, n_days=7, seed=1)).run()

    # SoC must always be between 0 and 1
    assert driving["soc"].between(0, 1).all()

    assert events["plug_in_soc"].between(0, 1).all()
    assert events["end_soc"].between(0, 1).all()

    # Physical quantities cannot be negative
    assert (events["energy_delivered_kwh"] >= 0).all()
    assert (events["charging_duration_hours"] >= 0).all()
    assert (events["plug_duration_hours"] >= 0).all()

    # Cannot charge longer than the vehicle is plugged in
    assert (
        events["charging_duration_hours"]
        <= events["plug_duration_hours"]
    ).all()

    # Plug-out must occur after plug-in
    assert (events["plug_out"] > events["plug_in"]).all()
