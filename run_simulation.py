from pathlib import Path
from src.config import load_archetypes
from src.simulator import EVSimulator, SimulationConfig

ROOT = Path(__file__).parent
archetypes = load_archetypes(ROOT / 'config' / 'archetypes.csv')
events, timeline = EVSimulator(archetypes, SimulationConfig()).run()
events.to_csv(ROOT / 'simulated_events.csv', index=False)
timeline.to_csv(ROOT / 'simulated_timeline.csv', index=False)
print(f'Generated {len(events):,} plug events for {events.agent_id.nunique():,} agents.')
