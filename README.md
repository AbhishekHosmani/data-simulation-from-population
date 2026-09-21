# Axle EV Behaviour Simulator

A small agent-based EV driver simulator for the Axle Data Science take-home. This project aimes to simulate raw events data from user-defined archetypes as the population configuration. The goal here is to generates stochastic individual plug-in events that can model customer behaviours.

## Design

Each agent is assigned one archetype according to `% of population`. The archetype supplies battery, efficiency, charger, plug-in frequency, typical plug-in/out times, target SoC, plug-in SoC and energy-per-plug-in anchors. Agents receive persistent personal offsets, then day-level randomness is sampled around those characteristics. This creates variation both **between drivers** and **within a driver's days**.

For each simulated plug event the simulation engine:
1. decides whether the agent plugs in from archetype frequency
2. samples plug-in/out times around the agent's persistent schedule
3. samples energy need around the archetype kWh/plug-in
4. combines the energy-implied SoC with the archetype SoC anchor and stochastic variation
5. computes energy needed to target SoC, charging duration, end SoC and flexible idle time
6. expands the event into a half-hourly timeline (user tuneable parameter) for visualisation

The engine is independent of Streamlit so it can later be exposed through FastAPI or used by optimisation/forecasting jobs without changing simulation logic.


## Run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run dashboard/app.py 
```

User can also pass the custom archetype.csv file as argument

```bash
streamlit run dashboard/app.py -- --archetypes path/to/archetypes.csv

example: run dashboard/app.py -- --archetypes '/Users/abhishekhosmani/DS_workspace/projects/Axel Take Home/data/Axle custom_archetypes.csv'

```

Or generate CSV outputs without the UI:

```bash
python run_simulation.py
```

Tests:

```bash
pytest -q
```

## Dashboard

- Population distributions: plug-in time, SoC, duration by archetype
- Individual-agent SoC/plugged-in timeline
- Flexibility: available agents, charging load, flexible hours
- Validation: simulation vs selected CNZ summary statistics
- CSV download of generated plug events


## Notes:
1. Our first objective is to use population characteristics to:
    * First simulate X agents, and for each agent simulate their daily driving behaviour 
    * Using their driving behaviour predict their energy consumption. Their driving decides their plug-in and plug-out time as well as SoC
    * Aggregate usage to population level and compare it to provided referenece document (Intelligent Octopus CNZ Report)


## Limitations / Future Work:
1. Add weekday/weekend schedules and correlated behaviours. (can be currently done by introducing specific archetypes)
2. Currently plug-in decision is happending based on random draw from Bernoulli Distibution. Need a more sophesticated model to predict plug-in behaviour based on archetype and driving behaviour. 
3. An agent can only plug-in once per day.
4. Need more rigirious testing.
5. Calibrate distribution widths from raw telemetry when available.
6. Model explicit trips and derive SoC from trip distance rather than partially anchoring it to kWh/plug-in.
7. Add smart-charge scheduling in half-hour blocks against price/carbon signals.
8. Add bump-charge/override behaviour.
9. Add more extensive testing
10. Expose the engine through FastAPI only when multiple clients need it.

## Design Decisions, Assumptions and Trade-offs

### Decisions made during development

I designed the simulator around a population of individual EV drivers sampled from the provided archetypes. Each simulated driver inherits characteristics such as annual mileage, battery capacity, charging frequency, typical plug-in/plug-out times and target SoC from their archetype.

The simulation is stateful at the individual-driver level. An agent's SoC is carried forward from one day to the next:

`Previous SoC → Driving → Energy consumed → Plug-in decision → Charging → Updated SoC`

Driving is simulated at 30-minute intervals. Daily mileage is sampled around the archetype's annual mileage assumption and distributed across a small number of trips. Driving consumes energy according to vehicle efficiency and battery capacity, which allows plug-in SoC to emerge from the simulated behaviour rather than being independently sampled each day.

Charging sessions are then generated using the archetype's charging frequency and typical plug-in behaviour. Charging is constrained by the vehicle's battery capacity, charger power, target SoC and the amount of time the vehicle remains plugged in.

I deliberately kept archetypes configuration-driven. `_make_agents()` samples from the rows in the archetype configuration rather than hard-coding individual archetype names. Therefore, a new archetype can be introduced by adding a new configuration row and assigning an appropriate population weight, without changing the agent-generation logic.

### Assumptions

The supplied archetypes describe population-level behaviour rather than complete individual travel histories, so several assumptions are required to turn them into an agent-based simulation.

- Annual mileage is converted into an expected daily mileage and random variation is added between days.
- Drivers make a small number of trips per day, with two trips being the most common case. This is a modelling assumption rather than a parameter directly provided by the archetype data.
- Trip times are approximated using plausible daily travel patterns rather than attempting to build a detailed transport-demand model.
- Plug-in and plug-out times vary around the archetype averages so that all agents within an archetype do not behave identically.
- `plug_frequency_per_day` is currently interpreted as the probability that an agent plugs in on a given day, meaning the model generates at most one charging session per agent per day.
- Charging power is assumed to remain constant during a charging session until the target SoC is reached.
- SoC is constrained to the physical range of 0–100%.
- Behavioural distributions and noise parameters are intentionally exposed/configurable so they can be recalibrated if richer observed data becomes available.

These assumptions prioritise interpretable population behaviour over modelling every aspect of an individual's journey.

### Where I spent time

I prioritised the parts of the simulation that directly determine the required outputs: **when a driver plugs in and their battery SoC when they plug in**.

Specifically, I focused on:

1. Generating a heterogeneous population from the supplied archetypes.
2. Maintaining SoC as state across multiple days.
3. Connecting driving behaviour to energy consumption and therefore plug-in SoC.
    `NOTE`: time spent making sure the SoC charge/discharge value is dependent on trips and plug-in/out times.
4. Modelling realistic variation in plug-in and plug-out times.
5. Ensuring charging is physically constrained by battery capacity, charger power and available plug-in time.
6. Producing both event-level and time-series outputs that can be inspected and validated.
7. Building an interactive dashboard to make individual-agent and population-level behaviour easy to explore.

I spent less time modelling detailed journey behaviour such as simulating varied weekend behaviour, batterey usage forcasting/deterioration and to some extent scalability of simulating high number of agents. These would add complexity but are secondary to the main objective of modelling EV availability and SoC at plug-in.

### Known Issues
1. Dashboard lagging with change in simulation parameters. Potential Scaling issues with generating complex pobability samplings as well as loaded UI. Need efficient data generation as well as premium UI hosting service.


### Design for the end use

I treated the simulator as a tool that could eventually support analysis of EV charging flexibility rather than as a one-off synthetic-data script.

The system therefore separates:

`Configuration → Agent generation → Driving simulation → SoC state → Charging simulation → Outputs / Dashboard`

This separation makes the model easier to extend and calibrate. For example, additional archetypes can be added through configuration, simulation parameters can be adjusted without rewriting the core logic, and individual components could later be replaced with models learned from observed data.

The simulator produces detailed agent-level records in addition to aggregate statistics. This is important because downstream applications may need to answer questions such as:

- How many EVs are plugged in at a particular time?
- What is the distribution of SoC when vehicles connect?
- How much energy does each vehicle require?
- How long is a vehicle connected but not actively charging?
- How much flexible charging capacity exists across the population?

The Streamlit dashboard provides a lightweight way to inspect these behaviours at both the individual-driver and population level. In a production setting, the same simulation outputs could instead feed forecasting, optimisation or smart-charging systems.

Overall, I prioritised a model that is **interpretable, configurable and extensible**, while keeping assumptions explicit so that they can be replaced or calibrated as higher-quality behavioural data becomes available.