from __future__ import annotations
from pathlib import Path
import pandas as pd


def _pct(x):
    """
    Helper to convert percentage string to float
    """
    if isinstance(x, str) and x.endswith('%'):
        return float(x[:-1]) / 100.0
    return float(x)


def _hour(x: str) -> float:
    """
    Helper to convert time format: 'HH:MM AM/PM' -> float hour
    params: x (eg. '12:30 PM')
    returns: 12.5
    """
    t = pd.to_datetime(x, format='%I:%M %p')
    return t.hour + t.minute / 60.0


def load_archetypes(path: str | Path) -> pd.DataFrame:
    """converts archetype data to appropriate column names and types."""
    df = pd.read_csv(path)
    df = df.rename(columns={
        '#': 'archetype_id', 'Name': 'name', '% of population': 'population_pct',
        'Miles/yr': 'miles_per_year', 'Battery (kWh)': 'battery_kwh',
        'Efficiency (mi/kWh)': 'efficiency_mi_per_kwh',
        'Plug-in frequency (per day)': 'plug_frequency_per_day',
        'Charger kW': 'charger_kw', 'Plug-in time': 'plug_in_time',
        'Plug-out time': 'plug_out_time', 'Target SoC': 'target_soc',
        'kWh/year': 'kwh_per_year', 'kWh/plug-in': 'kwh_per_plugin',
        'Plug-in SoC': 'plug_in_soc', 'SoC requirement': 'soc_requirement',
        'Charging duration (hrs)': 'charging_duration_hours'
    })
    for c in ['target_soc', 'plug_in_soc', 'soc_requirement']:
        df[c] = df[c].map(_pct)
    df['plug_in_hour'] = df['plug_in_time'].map(_hour)
    df['plug_out_hour'] = df['plug_out_time'].map(_hour)
    df['population_weight'] = df['population_pct'] / df['population_pct'].sum()
    print(f"Loaded {len(df)} archetypes from {path}", df)
    return df
