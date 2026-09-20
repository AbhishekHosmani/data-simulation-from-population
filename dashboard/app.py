from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_archetypes
from src.simulator import EVSimulator, SimulationConfig
from src.validation import validation_table

st.set_page_config(page_title='Axle EV Behaviour Simulator', layout='wide')
st.title('⚡ Axle EV Behaviour Simulator')
st.caption('Agent-based simulation of EV driver plug-in behaviour using Axle archetypes.')

archetypes = load_archetypes(ROOT / 'config' / 'archetypes.csv')

with st.sidebar:
    st.header('Simulation controls')
    n_agents = st.slider('Agents', 100, 10000, 1500, 100)
    n_days = st.slider('Days', 7, 90, 30)
    seed = st.number_input('Random seed', min_value=0, value=42)
    st.subheader('Behavioural variation')
    time_noise = st.slider('Daily time variation (hours)', 0.0, 2.0, 0.65, 0.05)
    energy_cv = st.slider('Energy-use variability (CV)', 0.0, 0.75, 0.25, 0.05)
    run = st.button('Run simulation', type='primary', use_container_width=True)
    

st.subheader('Archetype configuration')
display_cols = ['name','population_pct','miles_per_year','battery_kwh','plug_frequency_per_day','plug_in_time','plug_out_time','target_soc','plug_in_soc','charging_duration_hours']
st.dataframe(archetypes[display_cols], use_container_width=True, hide_index=True)

@st.cache_data(show_spinner=False)
def simulate(n_agents, n_days, seed, time_noise, energy_cv):
    cfg = SimulationConfig(n_agents=n_agents, n_days=n_days, seed=int(seed), time_noise_hours=time_noise, energy_cv=energy_cv)
    return EVSimulator(archetypes, cfg).run()

if run or 'events' not in st.session_state:
    with st.spinner('Simulating EV drivers...'):
        (
    st.session_state.events, st.session_state.driving, st.session_state.timeline,
) = simulate(n_agents, n_days, seed, time_noise, energy_cv)

events = st.session_state.events
timeline = st.session_state.timeline

if events.empty:
    st.warning('No plug-in events were generated. Increase agents/days or plug-in frequency.')
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric('Plug-in events', f'{len(events):,}')
c2.metric('Median plug-in SoC', f'{events.plug_in_soc.median():.0%}')
c3.metric('Median plug duration', f'{events.plug_duration_hours.median():.1f} h')
c4.metric('Median flexible time', f'{events.flexible_hours.median():.1f} h')

raw_data = pd.concat([
    events.assign(record_type='event'),
    timeline.assign(record_type='timeline'),
], ignore_index=True, sort=False)
st.download_button(
    'Download all raw simulation data (CSV)',
    raw_data.to_csv(index=False),
    'axle_simulation_raw_data.csv',
    'text/csv',
)

indivisual_tab, population_tab, flexibility_tab, validation_tab = st.tabs(['Individual agent','Population', 'Flexibility', 'Validation'])
with population_tab:
    # ---------------------------------------------------------
    # % of population plugged in by hour of day
    # ---------------------------------------------------------

    plugged = timeline.copy()

    plugged["date"] = plugged["timestamp"].dt.date
    plugged["hour"] = plugged["timestamp"].dt.hour

    # Count unique agents plugged in for each date + hour.
    # Using nunique prevents counting the same agent multiple times
    # within an hour because timeline has 30-minute observations.
    hourly_daily = (plugged.groupby(["date", "hour"], as_index=False).agg(
            plugged_agents=("agent_id", "nunique")))

    # Convert count into % of the full simulated population
    hourly_daily["pct_agents_plugged"] = (hourly_daily["plugged_agents"] / n_agents* 100)

    # Average the percentage across simulation days
    hourly_population = (
        hourly_daily
        .groupby("hour", as_index=False)
        .agg(
            pct_agents_plugged=("pct_agents_plugged", "mean")
        )
    )

    # Make sure hours with no plugged-in agents are still shown
    all_hours = pd.DataFrame({"hour": range(24)})

    hourly_population = all_hours.merge(hourly_population, on="hour",how="left")

    hourly_population["pct_agents_plugged"] = (
        hourly_population["pct_agents_plugged"]
        .fillna(0)
    )

    # ---------------------------------------------------------
    # Bar chart
    # ---------------------------------------------------------

    fig = px.bar(
        hourly_population,
        x="hour",
        y="pct_agents_plugged",
        title="Percentage of agents plugged in by hour",
        labels={
            "hour": "Hour of day",
            "pct_agents_plugged": "Agents plugged in (%)",
        },
    )

    fig.update_xaxes(range=[-0.5, 23.5], dtick=1)

    fig.update_yaxes(range=[0, 100], ticksuffix="%")

    fig.update_traces(
        hovertemplate=(
            "Hour: %{x}:00"
            "<br>Agents plugged in: %{y:.1f}%"
            "<extra></extra>"
        )
    )
    st.plotly_chart(fig, width="stretch")
    # left,right = st.columns(2)
    # with left:
    #     fig = px.histogram(events, x='plug_in_hour', nbins=48, histnorm='probability density', title='Plug-in time distribution')
    #     fig.update_xaxes(range=[0,24], dtick=2, title='Hour of day')
    #     st.plotly_chart(fig, use_container_width=True)
    # with right:
    #     fig = px.histogram(events, x='plug_in_soc', nbins=25, histnorm='probability density', title='Plug-in SoC distribution')
    #     fig.update_xaxes(tickformat='.0%', range=[0,1])
    #     st.plotly_chart(fig, use_container_width=True)
    # fig = px.box(events, x='archetype', y='plug_duration_hours', points=False, title='Plug duration by archetype')
    # st.plotly_chart(fig, use_container_width=True)

with indivisual_tab:
    # 1. Select archetype
    archetypes = sorted(events["archetype"].dropna().unique())

    selected_archetype = st.selectbox(
        "Choose an archetype",
        archetypes,
        key="individual_archetype",
    )

    # 2. Select agent belonging to the selected archetype
    filtered_agents = (
        events.loc[
            events["archetype"] == selected_archetype,
            "agent_id",
        ]
        .dropna()
        .unique()
    )

    selected = st.selectbox(
        "Choose an agent",
        filtered_agents,
        key="individual_agent",
    )

    # All charging events for selected agent
    ae = events[
        (events["archetype"] == selected_archetype)
        & (events["agent_id"] == selected)
    ].sort_values("plug_in")

    st.write(f"**Archetype:** {selected_archetype}")

    # ---------------------------------------------------------
    # 3. Select date
    # ---------------------------------------------------------

    # Use driving data so dates where the agent did NOT plug in
    # are also available in the date selector.
    agent_driving = st.session_state.driving[
        st.session_state.driving["agent_id"] == selected
    ].copy()

    available_dates = sorted(
        agent_driving["timestamp"].dt.date.unique()
    )

    selected_date = st.selectbox(
        "Choose a date",
        available_dates,
        key="individual_date",
    )

    # ---------------------------------------------------------
    # 4. Driving / SoC data for selected date
    # ---------------------------------------------------------

    at = agent_driving[
        agent_driving["timestamp"].dt.date == selected_date
    ].copy()

    at = at.sort_values("timestamp")


    # ---------------------------------------------------------
    # Charging timeline for selected agent/date
    # ---------------------------------------------------------

    agent_charging = st.session_state.timeline[
        st.session_state.timeline["agent_id"] == selected
    ].copy()

    charging_day = agent_charging[
        agent_charging["timestamp"].dt.date == selected_date
    ].copy()


    # ---------------------------------------------------------
    # Charging events overlapping selected date
    # ---------------------------------------------------------

    day_start = pd.Timestamp(selected_date)
    day_end = day_start + pd.Timedelta(days=1)

    ae_date = ae[
        (ae["plug_in"] < day_end)
        & (ae["plug_out"] > day_start)
    ].copy()


    if not at.empty:

        # Rename original driving SoC
        at = at.rename(
            columns={
                "soc": "driving_soc"
            }
        )

        # -----------------------------------------------------
        # Merge charging SoC onto driving timeline
        # -----------------------------------------------------

        if not charging_day.empty:

            charging_day = charging_day[
                [
                    "timestamp",
                    "plugged_in",
                    "charging",
                    "soc",
                ]
            ].rename(
                columns={
                    "soc": "charging_soc"
                }
            )

            at = at.merge(
                charging_day,
                on="timestamp",
                how="left",
            )

            # Use charging SoC whenever charging timeline exists.
            # Otherwise use the normal driving SoC.
            at["soc"] = (
                at["charging_soc"]
                .combine_first(at["driving_soc"])
            )

            at["plugged_in"] = (
                at["plugged_in"]
                .fillna(False)
                .astype(bool)
            )

            at["charging"] = (
                at["charging"]
                .fillna(False)
                .astype(bool)
            )

        else:

            # No charging occurred on this date
            at["soc"] = at["driving_soc"]
            at["plugged_in"] = False
            at["charging"] = False


        # Convert timestamp into decimal hour
        at["hour_of_day"] = (
            at["timestamp"].dt.hour
            + at["timestamp"].dt.minute / 60.0
        )

        # Convert SoC from 0-1 to percentage
        at["soc_pct"] = at["soc"] * 100

        # ---------------------------------------------------------
        # 5. Plot daily SoC profile
        # ---------------------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=at["hour_of_day"],
                y=at["soc_pct"],
                mode="lines+markers",
                name="SoC (%)",
            )
        )

        # ---------------------------------------------------------
        # 6. Shade actual plug-in period
        # ---------------------------------------------------------

        for event in ae_date.itertuples(index=False):
            # Clip charging session to the selected calendar day
            visible_start = max(
                event.plug_in,
                day_start,
            )

            visible_end = min(
                event.plug_out,
                day_end,
            )

            x0 = (
                visible_start - day_start
            ).total_seconds() / 3600.0

            x1 = (
                visible_end - day_start
            ).total_seconds() / 3600.0

            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="dodgerblue",
                opacity=0.45,
                line_width=0,
                layer="below",
            )
            # plug_in_hour = (
            #     event.plug_in.hour
            #     + event.plug_in.minute / 60.0
            # )

            # plug_out_hour = (
            #     event.plug_out.hour
            #     + event.plug_out.minute / 60.0
            # )

            # # If plug-out is on the following day,
            # # split the shading around midnight.
            # if event.plug_out.date() > event.plug_in.date():

            #     # Midnight -> plug-out
            #     fig.add_vrect(
            #         x0=0,
            #         x1=plug_out_hour,
            #         fillcolor="dodgerblue",
            #         opacity=0.45,
            #         line_width=0,
            #         layer="below",
            #     )

            #     # Plug-in -> midnight
            #     fig.add_vrect(
            #         x0=plug_in_hour,
            #         x1=24,
            #         fillcolor="dodgerblue",
            #         opacity=0.45,
            #         line_width=0,
            #         layer="below",
            #     )

            # else:
            #     # Plug-in and plug-out occur on same day
            #     fig.add_vrect(
            #         x0=plug_in_hour,
            #         x1=plug_out_hour,
            #         fillcolor="dodgerblue",
            #         opacity=0.45,
            #         line_width=0,
            #         layer="below",
            #     )

        # ---------------------------------------------------------
        # 7. Chart formatting
        # ---------------------------------------------------------

        fig.update_layout(
            title=f"Daily profile — {selected} — {selected_date}",
            xaxis_title="Time of day",
            yaxis_title="SoC (%)",
            hovermode="x unified",
        )

        fig.update_xaxes(
            range=[0, 24],
            dtick=1,
        )

        fig.update_yaxes(
            range=[0, 100],
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    # ---------------------------------------------------------
    # 8. Show charging-event details
    # ---------------------------------------------------------

    if not ae_date.empty:
        st.dataframe(
            ae_date[
                [
                    "plug_in",
                    "plug_out",
                    "plug_in_soc",
                    "end_soc",
                    "energy_delivered_kwh",
                    "flexible_hours",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "This agent did not plug in on the selected date."
        )

with flexibility_tab:
    hourly = timeline.assign(hour=timeline.timestamp.dt.hour).groupby('hour', as_index=False).agg(
        plugged_agents=('plugged_in','sum'), charging_kw=('charging_power_kw','sum'))
    fig = px.line(hourly, x='hour', y=['plugged_agents','charging_kw'], title='Availability and charging load by hour')
    fig.update_xaxes(dtick=2)
    st.plotly_chart(fig, use_container_width=True)
    flex = events.groupby('archetype', as_index=False).agg(events=('agent_id','size'), median_flexible_hours=('flexible_hours','median'), total_energy_kwh=('energy_delivered_kwh','sum'))
    st.dataframe(flex, use_container_width=True, hide_index=True)

with validation_tab:
    st.markdown('The simulator is calibrated to Axle archetypes; these CNZ values are useful external population-level checks, not exact targets for every archetype.')
    st.dataframe(validation_table(events), use_container_width=True, hide_index=True)
    st.markdown('**Interpretation:** deviations are expected because the six-archetype mixture includes behaviours beyond the Intelligent Octopus average. Calibration parameters are intentionally explicit and configurable.')


