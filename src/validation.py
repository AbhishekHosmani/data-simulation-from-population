from __future__ import annotations
import pandas as pd


def validation_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=['metric','simulation','reference','note'])
    return pd.DataFrame([
        ['Median plug-in SoC', f"{events.plug_in_soc.median():.0%}", '52%', 'CNZ population reference'],
        ['Median charging duration', f"{events.charging_duration_hours.median():.2f} h", '2.5 h', 'CNZ population reference'],
        ['Median plug duration', f"{events.plug_duration_hours.median():.2f} h", '10–15 h overnight', 'CNZ qualitative range'],
        ['Median flexible time', f"{events.flexible_hours.median():.2f} h", 'Substantial idle time', 'Derived simulator output'],
    ], columns=['metric','simulation','reference','note'])
