from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

## NOTES ON DEVELOPMENT:
# simulate_daily_soc()  ->  Using arch type behaviour, you predict daily driver behaviour and simulate hourly SoC use, as well as plug-in and plug-out times.
#                           The challenge here is 
# offesets              ->  Thoughout the code and simulations we add offsets to introduce persistant variation between people.
#                           eg. base_plugin_hour + plugin_hour_offset: This is a persistent offset for each agent, so they add variation between people.
# CURRENT PRIORITIZATION OF WORK:
# 1. Driving behaviour and SoC simulation : creating framework to simulate both agent level and agent-driving level simulations as they are fundamental to the goal.
# 2. 
# FUTURE WORK:
# * Need different behaviour between weekday and weekend.
# * Currently plug-in decision is happending based on random draw from Bernoulli Distibution. 
#   Need a more sophesticated model to predict plug-in behaviour based on archetype and driving behaviour. 

@dataclass(frozen=True)
class SimulationConfig:
    n_agents: int = 500
    n_days: int = 30
    start_date: str = "2026-01-01"
    seed: int = 42
    timestep_minutes: int = 30
    time_noise_hours: float = 0.65
    energy_cv: float = 0.25
    soc_noise: float = 0.04


def _sample_wrapped_hour(random_number_generator, mean, sd) -> float:
    """
    Sample a decimal hour from a normal distribution and
    wrap the result onto a 24-hour clock.
    """
    return float(
        random_number_generator.normal(mean, sd) % 24
    )


def _timestamp_for_hour(day: pd.Timestamp, hour: float) -> pd.Timestamp:
    """
    Convert a decimal clock hour into a timestamp for a given day.

    Example:
        hour=18.5 -> 18:30
    """
    h = int(hour) % 24
    m = int(round((hour - int(hour)) * 60))

    if m == 60:
        h = (h + 1) % 24
        m = 0

    return (day.normalize()+pd.Timedelta(hours=h, minutes=m))


def _positive_lognormal(random_number_generator, mean: float, coeff_of_variation: float) -> float:
    """
    Draw a positive random value from a log-normal distribution
    with approximately the requested mean and coefficient of variation.
    """
    if mean <= 0:
        return 0.0

    sigma2 = np.log(1 + coeff_of_variation * coeff_of_variation)
    mu = np.log(mean) - sigma2 / 2

    return float(random_number_generator.lognormal(mu, np.sqrt(sigma2)))


class EVSimulator:
    """
    Agent-based EV simulator.

    Each agent receives a persistent archetype plus small persistent
    behavioural offsets.

    Driving consumes battery SoC. Charging replenishes SoC.

    Importantly, SoC persists between simulation days.
    """

    def __init__(self, archetypes: pd.DataFrame, config: SimulationConfig):
        self.a = (archetypes.reset_index(drop=True).copy())
        self.cfg = config
        self.random_number_generator = (np.random.default_rng(config.seed))
        self.agents = self._make_agents()

    def _make_agents(self) -> pd.DataFrame:
        """
        Sample individual EV drivers from the archetype population.
        """
            # Validate archetype population weights

        if not np.isclose(self.a.population_weight.sum(), 1.0):
            raise ValueError(
                "Archetype population weights must sum to 1.0. "
                f"Current sum: {self.a.population_weight.sum():.3f}")
        
        idx = self.random_number_generator.choice(
            len(self.a),
            self.cfg.n_agents,
            p=self.a.population_weight,
        )
        base = (self.a.iloc[idx].reset_index(drop=True))

        agents = pd.DataFrame({
            "agent_id": [f"EV{i:05d}"for i in range(self.cfg.n_agents)],
            "archetype_id": base.archetype_id.astype(int),
            "archetype": base.name,
            "battery_kwh": base.battery_kwh.astype(float),
            "miles_per_year": base.miles_per_year.astype(float),
            "efficiency_mi_per_kwh": base.efficiency_mi_per_kwh.astype(float),
            "charger_kw": base.charger_kw.astype(float),
            "target_soc": base.target_soc.astype(float),
            "base_plugin_soc": base.plug_in_soc.astype(float),
            "base_kwh_per_plugin": base.kwh_per_plugin.astype(float),
            "plug_frequency_per_day": base.plug_frequency_per_day.astype(float),
            "base_plugin_hour": base.plug_in_hour.astype(float),
            "base_plugout_hour": base.plug_out_hour.astype(float),
        })

        # Persistent driver-level heterogeneity
        agents["plugin_hour_offset"] = (
            self.random_number_generator.normal(0,1.45, len(agents)))

        agents["plugout_hour_offset"] = (
            self.random_number_generator.normal(0, 1.45, len(agents)))

        agents["soc_offset"] = (
            self.random_number_generator.normal(0, 0.25, len(agents)))
        return agents

    # Daily driving simulation and SoC expenditure simulations
    def simulate_daily_soc(
        self,
        agent,
        date: str | pd.Timestamp,
        starting_soc: float = 0.80,
        daily_miles: float | None = None,
        n_trips: int | None = None,
        trip_hours: list[float] | None = None,
        timestep_minutes: int | None = None,
    ) -> pd.DataFrame:
        """
        Simulate an EV driver's battery consumption for one day.

        Returns a timestep-level DataFrame showing:

            timestamp
            agent_id
            archetype
            miles_driven
            energy_used_kwh
            soc_consumed
            soc

        SoC decreases when trips occur.
        """

        battery_kwh = float(agent.battery_kwh)
        efficiency = float(agent.efficiency_mi_per_kwh)
        annual_miles = float(agent.miles_per_year)

        # 1. Check for validity of value assignment
        if battery_kwh <= 0: 
            raise ValueError("battery_kwh must be positive")

        if efficiency <= 0:
            raise ValueError("efficiency_mi_per_kwh must be positive")

        if not 0 <= starting_soc <= 1:
            raise ValueError("starting_soc must be between 0 and 1")

        # 2. Estimate daily mileage
        expected_daily_miles = (annual_miles/365.0)

        if daily_miles is None:
            daily_miles = _positive_lognormal(
                self.random_number_generator,
                expected_daily_miles,
                coeff_of_variation=0.35
            )

        daily_miles = max(0.0,float(daily_miles))

        # 3. Number of trips
        if n_trips is None:
            n_trips = int(
                self.random_number_generator.choice([1, 2, 3, 4],
                    # Note: I am manually setting priority here based on my intuition of what a typical EV driver might do.
                    p=[0.10, 0.60, 0.20, 0.10]))

        if n_trips < 1:
            raise ValueError("n_trips must be >= 1")

        # 4. Generate trip times
        if trip_hours is None:
            if n_trips == 1:
                trip_hours = [_sample_wrapped_hour(self.random_number_generator,mean=12,sd=2)]

            elif n_trips == 2:
                # Commute-like behaviour:
                # using morning outbound + evening return behaviour.
                trip_hours = [_sample_wrapped_hour(self.random_number_generator, mean=8, sd=0.75),
                    _sample_wrapped_hour(self.random_number_generator, mean=17.5, sd=0.75)]

            else:
                # Multiple trips spread across daytime.
                trip_hours = sorted(self.random_number_generator.uniform(7, 21, size=n_trips))

        if len(trip_hours) != n_trips:
            raise ValueError("Length of trip_hours must equal n_trips")
        
        last_trip_hour = max(trip_hours)

        # 5. Divide mileage between trips using a Dirichlet distribution to ensure all trips sum to the daily mileage.
        trip_weights = (self.random_number_generator.dirichlet(np.ones(n_trips)))
        trip_miles = (daily_miles * trip_weights)

        # 6. Build daily timeline
        if timestep_minutes is None:
            timestep_minutes = (self.cfg.timestep_minutes)

        day = pd.Timestamp(date)
        timeline = pd.date_range(
            start=day,
            end=day + pd.Timedelta(days=1),
            freq=f"{timestep_minutes}min",
            inclusive="left",
        )

        # 7. Put trips onto nearest timestep
        steps_per_day = int(24 * 60 / timestep_minutes)

        miles_by_timestep = np.zeros(steps_per_day, dtype=float)

        for trip_hour, miles in zip(trip_hours,trip_miles):
            trip_index = int(round(trip_hour * 60 / timestep_minutes)
            ) % steps_per_day

            miles_by_timestep[trip_index] += miles

        # 8. Miles -> energy
        energy_used_kwh = (miles_by_timestep / efficiency)

        # 9. Energy -> SoC reduction
        soc_consumed = (energy_used_kwh / battery_kwh)

        # 10. Persistent intraday SoC
        cumulative_soc_consumption = (soc_consumed.cumsum())
        soc = np.clip(starting_soc - cumulative_soc_consumption, 0.0, 1.0)

        # generate timeline
        timeline = pd.date_range(start=day,
            periods=steps_per_day,
            freq=f"{timestep_minutes}min")

        # Create one DataFrame
        result = pd.DataFrame({
        "timestamp": timeline,
        "agent_id": agent.agent_id,
        "archetype": agent.archetype,
        "miles_driven": miles_by_timestep,
        "energy_used_kwh": energy_used_kwh,
        "soc_consumed": soc_consumed,
        "soc": soc,
        "battery_kwh": battery_kwh,
        "efficiency_mi_per_kwh": efficiency,
        })
        return result, last_trip_hour

    # MAIN SIMULATION
    def run(self):
        """
        Run the complete stateful EV simulation.

        Driving decreases SoC.
        When an agent plugs in, charging increases SoC toward the driver's target.
        SoC carries over between days.

        Returns
        -------
        charging_df
            One row per plug-in event.

        driving_df
            Half-hourly driving/SoC observations.

        timeline_df
            Half-hourly plugged-in/charging observations.
        """

        charging_events = []
        driving_events = []

        start_date = pd.Timestamp(self.cfg.start_date)

        # Persistent battery state  
        current_soc = { row.agent_id: float(row.target_soc)
            for row in self.agents.itertuples(index=False)
        }

        # Day loop
        for day_idx in range(self.cfg.n_days):
            date = (start_date + pd.Timedelta(days=day_idx))

            # Some days people are collectively more/less likely to plug in
            daily_plugin_multiplier = np.clip(
            self.random_number_generator.normal(loc=1.0,scale=0.20),0.50, 1.50)

            # Inducing more variability through daily time shift
            daily_time_shift = self.random_number_generator.normal(loc=0.0,scale=1.0)

            # Loop through each agent and simulate their driving and charging behaviour for the day.
            for agent in self.agents.itertuples(index=False):

                starting_soc = (current_soc[agent.agent_id])

                # 1. SIMULATE DRIVING
                driving, last_trip_hour = self.simulate_daily_soc(
                    agent=agent,
                    date=date,
                    starting_soc=starting_soc,
                    timestep_minutes=(self.cfg.timestep_minutes))

                driving_events.append(driving)

                # Aggregate today's driving
                soc_after_driving = float(driving["soc"].iloc[-1])
                daily_miles = float(driving["miles_driven"].sum())
                driving_energy_kwh = float(driving["energy_used_kwh"].sum())
                soc_consumed = float(driving["soc_consumed"].sum())

                # 2. PLUG-IN DECISION
                plug_probability = float(np.clip(agent.plug_frequency_per_day*daily_plugin_multiplier, 0.0, 1.0))
                plugs_in = (self.random_number_generator.random() < plug_probability)

                # No charging
                if not plugs_in:
                    current_soc[agent.agent_id] = soc_after_driving
                    continue

                # 3. Generate plug-in and plug-out times with noise
                plugin_hour = (_sample_wrapped_hour(self.random_number_generator,
                        agent.base_plugin_hour
                        + agent.plugin_hour_offset + daily_time_shift,
                        self.cfg.time_noise_hours))

                # Vehicle cannot plug in before its final trip of the day.
    
                # Add one timestep after the final trip to represent the
                # driver returning home before connecting the vehicle.
                minimum_plugin_hour = (last_trip_hour + self.cfg.timestep_minutes / 60.0)
                plugin_hour = max(plugin_hour, minimum_plugin_hour)

                # Keep plug-in within the current calendar day
                plugin_hour = min(plugin_hour,23.99)

                plugout_hour = (
                    _sample_wrapped_hour(
                        self.random_number_generator,
                        agent.base_plugout_hour
                        + agent.plugout_hour_offset,
                        self.cfg.time_noise_hours,
                    )
                )

                plug_in = _timestamp_for_hour(date,plugin_hour)
                plug_out = _timestamp_for_hour(date,plugout_hour)

                # Overnight connection
                if plug_out <= plug_in:
                    plug_out += pd.Timedelta(days=1)

                plug_duration_hours = (
                    (plug_out - plug_in).total_seconds() / 3600.0
)

                # 4. PLUG-IN SOC
                plug_in_soc = (soc_after_driving)
                target_soc = float(agent.target_soc)

                # 5. Energy required to reach target SoC
                energy_required_kwh = max(0.0,(target_soc - plug_in_soc) * float(agent.battery_kwh))

                # 6. CHARGING TIME
                if agent.charger_kw <= 0:
                    theoretical_charge_hours = (0.0)
                else:
                    theoretical_charge_hours = (energy_required_kwh / float(agent.charger_kw))

                charging_duration_hours = min(theoretical_charge_hours, plug_duration_hours)

                # ----------------------------------------------
                # 7. ENERGY DELIVERED
                # ----------------------------------------------

                energy_delivered_kwh = min(energy_required_kwh,
                                           charging_duration_hours* float(agent.charger_kw))

                # ----------------------------------------------
                # 8. END SOC
                # ----------------------------------------------

                soc_added = (
                    energy_delivered_kwh
                    / float(
                        agent.battery_kwh
                    )
                )

                end_soc = min(
                    target_soc,
                    plug_in_soc
                    + soc_added,
                )

                end_soc = float(
                    np.clip(
                        end_soc,
                        0.0,
                        1.0,
                    )
                )
                # ----------------------------------------------
                # 9. SAVE STATE FOR TOMORROW
                # ----------------------------------------------
                current_soc[agent.agent_id] = end_soc

                # ----------------------------------------------
                # 10. FLEXIBILITY
                # ----------------------------------------------

                flexible_hours = max(0.0, plug_duration_hours - charging_duration_hours)

                # ----------------------------------------------
                # 11. SAVE CHARGING EVENT
                # ----------------------------------------------

                charging_events.append({
                    "date": date,
                    "agent_id": agent.agent_id,
                    "archetype":agent.archetype,
                    "daily_miles": daily_miles,
                    "driving_energy_kwh": driving_energy_kwh,
                    "soc_consumed": soc_consumed,
                    "plug_in": plug_in,
                    "plug_out": plug_out,
                    "plug_in_soc": plug_in_soc,
                    "target_soc": target_soc,
                    "end_soc": end_soc,
                    "battery_kwh": float(agent.battery_kwh),
                    "charger_kw": float(agent.charger_kw),
                    "energy_required_kwh": energy_required_kwh,
                    "energy_delivered_kwh": energy_delivered_kwh,
                    "charging_duration_hours": charging_duration_hours,
                    "plug_duration_hours": plug_duration_hours,
                    "flexible_hours": flexible_hours,
                })

        # ======================================================
        # CREATE OUTPUT DATAFRAMES
        # ======================================================

        charging_df = pd.DataFrame(charging_events)

        if not charging_df.empty:
            # Decimal hour, e.g. 18:30 -> 18.5
            charging_df["plug_in_hour"] = (
            charging_df["plug_in"].dt.hour + charging_df["plug_in"].dt.minute / 60.0
        )

        charging_df["plug_out_hour"] = (
        charging_df["plug_out"].dt.hour + charging_df["plug_out"].dt.minute / 60.0
        )

        if driving_events:
            driving_df = pd.concat(driving_events, ignore_index=True)
        else:
            driving_df = pd.DataFrame()

        timeline_df = self._make_timeline(charging_df)

        return (
            charging_df,
            driving_df,
            timeline_df,
        )

    # ==========================================================
    # CHARGING TIMELINE
    # ==========================================================

    def _make_timeline(
        self,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Expand charging events into timestep-level
        plugged-in observations.
        """

        columns = [
            "timestamp",
            "agent_id",
            "archetype",
            "plugged_in",
            "charging",
            "charging_power_kw",
            "soc",
        ]

        if events.empty:
            return pd.DataFrame(
                columns=columns
            )

        rows = []

        freq = (
            f"{self.cfg.timestep_minutes}min"
        )

        for event in events.itertuples(
            index=False
        ):

            times = pd.date_range(
                event.plug_in.floor(freq),
                event.plug_out.ceil(freq),
                freq=freq,
            )

            charge_end = (
                event.plug_in
                + pd.Timedelta(
                    hours=(
                        event.charging_duration_hours
                    )
                )
            )

            for timestamp in times:

                if (
                    timestamp < event.plug_in
                    or timestamp > event.plug_out
                ):
                    continue

                elapsed_charge_hours = max(
                    0.0,
                    min(
                        (
                            timestamp
                            - event.plug_in
                        ).total_seconds()
                        / 3600.0,
                        event.charging_duration_hours,
                    ),
                )

                soc = min(
                    event.end_soc,
                    event.plug_in_soc
                    + (
                        elapsed_charge_hours
                        * event.charger_kw
                        / event.battery_kwh
                    ),
                )

                charging = int(
                    event.plug_in
                    <= timestamp
                    < charge_end
                    and event.energy_delivered_kwh
                    > 0
                )

                rows.append({
                    "timestamp": timestamp,
                    "agent_id": event.agent_id,
                    "archetype": event.archetype,
                    "plugged_in": 1,
                    "charging": charging,
                    "charging_power_kw":
                        (
                            event.charger_kw
                            if charging
                            else 0.0
                        ),

                    "soc":
                        float(
                            np.clip(
                                soc,
                                0.0,
                                1.0,
                            )
                        ),
                })

        return pd.DataFrame(
            rows,
            columns=columns,
        )