import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.stats import gaussian_kde
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
st.title('Axle User Behaviour Simulator')
st.caption('Agent-based simulation of EV driver plug-in behaviour using Axle archetypes.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archetypes",
        type=Path,
        required=False,
        help="Path to archetypes CSV",
        default=ROOT / "config" / "archetypes.csv",
    )
    args, _ = parser.parse_known_args()
    return args

args = parse_args()
archetypes = load_archetypes(args.archetypes)


with st.sidebar:
    st.header('Simulation controls')
    n_agents = st.slider('Agents', 100, 1000, 500, 100)
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
    st.subheader("Population behaviour")
    st.caption(
        "Explore variation in charging behaviour across the simulated EV population."
    )

    # ============================================================
    # ARCHETYPE FILTER
    # ============================================================

    all_archetypes = sorted(
        st.session_state.driving["archetype"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_archetypes = st.multiselect(
        "Filter by driver archetype",
        options=all_archetypes,
        default=all_archetypes,
        key="population_archetype_filter",
    )

    if not selected_archetypes:
        st.warning("Select at least one archetype to view population behaviour.")
        st.stop()

    # ============================================================
    # FILTER DATA
    # ============================================================

    events = st.session_state.events[
        st.session_state.events["archetype"].isin(selected_archetypes)].copy()

    timeline = st.session_state.timeline[
        st.session_state.timeline["archetype"].isin(selected_archetypes)].copy()

    driving = st.session_state.driving[
        st.session_state.driving["archetype"].isin(selected_archetypes)].copy()
    
    st.subheader("Population behaviour")
    st.caption(
        "Explore variation in charging behaviour across the simulated EV population.")

    events = st.session_state.events[
        st.session_state.events["archetype"].isin(selected_archetypes)].copy()
    driving = st.session_state.driving[
        st.session_state.driving["archetype"].isin(selected_archetypes)].copy()
    timeline = st.session_state.timeline[
        st.session_state.timeline["archetype"].isin(selected_archetypes)].copy()

    # ============================================================
    # Summary metrics
    # ============================================================

    n_agents = driving["agent_id"].nunique()
    n_archetypes = driving["archetype"].nunique()
    n_days = driving["timestamp"].dt.date.nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric("EV drivers", f"{n_agents:,}")
    col2.metric("Archetypes", n_archetypes)
    col3.metric(
        "Simulation days",
        driving["timestamp"].dt.date.nunique(),
    )

    st.divider()

    # ============================================================
    # 1. POPULATION PLUG-IN BEHAVIOUR BY HOUR
    # ============================================================

    st.subheader("Population plug-in behaviour")

    st.caption(
        "Average percentage of EV drivers plugged in at each hour. "
        "The dotted lines show the 5th–95th percentile range across simulated days."
    )

    timeline["date"] = timeline["timestamp"].dt.date
    timeline["hour"] = timeline["timestamp"].dt.hour

    # Determine whether each agent was plugged in during each hour.
    #
    # Because the simulation is at 30-minute resolution, max() means
    # the agent counts as plugged in if they were connected during
    # either timestep within the hour.
    agent_hour = (timeline.groupby(["date", "hour", "agent_id"],as_index=False)["plugged_in"].max()
    )

    # Number of plugged-in agents for each day/hour
    daily_hourly = (agent_hour
        .groupby(["date", "hour"],as_index=False)["plugged_in"].sum())

    daily_hourly["pct_plugged"] = ((daily_hourly["plugged_in"] / n_agents) * 100)

    # Calculate mean and population variation across simulated days
    population_hourly = (
        daily_hourly
        .groupby("hour")["pct_plugged"]
        .agg(
            mean="mean",
            p05=lambda x: x.quantile(0.05),
            p95=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )

    fig = go.Figure()

    # Mean behaviour
    fig.add_trace(
        go.Bar(
            x=population_hourly["hour"],
            y=population_hourly["mean"],
            name="Average",
        )
    )

    # P95
    fig.add_trace(
        go.Scatter(
            x=population_hourly["hour"],
            y=population_hourly["p95"],
            mode="lines",
            name="95th percentile",
            line=dict(
                dash="dot",
                width=2,
            ),
        )
    )

    # P05
    fig.add_trace(
        go.Scatter(
            x=population_hourly["hour"],
            y=population_hourly["p05"],
            mode="lines",
            name="5th percentile",
            line=dict(
                dash="dot",
                width=2,
            ),
        )
    )

    fig.update_layout(
        title="% of EV drivers plugged in by hour",
        xaxis_title="Hour of day",
        yaxis_title="% of drivers plugged in",
        yaxis=dict(range=[0, 100]),
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=1,
        ),
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(fig, width="stretch")

    st.divider()

    # ============================================================
    # 2. VARIATION ACROSS INDIVIDUAL DRIVERS
    # ============================================================
# ============================================================
# Plug-in SoC probability distribution by archetype
# ============================================================

    driver_stats = (
        events
        .groupby(
            ["agent_id", "archetype"],
            as_index=False,
        )
        .agg(
            avg_plugin_soc=("plug_in_soc", "mean"),
        )
    )

    driver_stats["avg_plugin_soc_pct"] = (
        driver_stats["avg_plugin_soc"] * 100
    )

    fig_soc = go.Figure()

    # Common X-axis for every archetype
    x_grid = np.linspace(0, 100, 500)

    for archetype in selected_archetypes:

        values = driver_stats.loc[
            driver_stats["archetype"] == archetype,
            "avg_plugin_soc_pct",
        ].dropna().values

        # KDE requires multiple observations with some variation
        if len(values) < 2 or np.std(values) == 0:
            continue

        kde = gaussian_kde(values)

        density = kde(x_grid)

        fig_soc.add_trace(
            go.Scatter(
                x=x_grid,
                y=density,
                mode="lines",
                name=archetype,
                line=dict(
                    width=3,
                ),
            )
        )

    fig_soc.update_layout(
        title="Plug-in SoC distribution by archetype",
        xaxis_title="Average plug-in SoC (%)",
        yaxis_title="Probability density",
        template="plotly_dark",
        hovermode="x unified",
    )

    fig_soc.update_xaxes(
        range=[0, 100],
        dtick=10,
    )

    fig_soc.update_yaxes(
        rangemode="tozero",
    )

    st.plotly_chart(
        fig_soc,
        width="stretch",
    )
    # ============================================================
    # Plug-in time probability distribution by archetype
    # ============================================================

    fig_time = go.Figure()

    # Evaluate density from 0:00 to 24:00
    x_grid = np.linspace(0, 24, 500)

    for archetype in selected_archetypes:

        values = (
            events.loc[
                events["archetype"] == archetype,
                "plug_in_hour",
            ]
            .dropna()
            .values
        )

        if len(values) < 2 or np.std(values) == 0:
            continue

        # --------------------------------------------------------
        # Circular KDE
        #
        # Copy observations +/- 24 hours so that observations
        # around midnight are treated as being close together.
        # --------------------------------------------------------
        wrapped_values = np.concatenate([
            values - 24,
            values,
            values + 24,
        ])

        kde = gaussian_kde(
            wrapped_values,
            bw_method=0.15,
        )

        density = kde(x_grid)

        # Renormalize density over the visible 0-24 hour interval
        density = density / np.trapezoid(density, x_grid)

        fig_time.add_trace(
            go.Scatter(
                x=x_grid,
                y=density,
                mode="lines",
                name=archetype,
                line=dict(
                    width=3,
                ),
            )
        )


    fig_time.update_layout(
        title="Distribution of plug-in times by archetype",
        xaxis_title="Time of day",
        yaxis_title="Probability density",
        template="plotly_dark",
        hovermode="x unified",
    )


    fig_time.update_xaxes(
        range=[0, 24],
        tickmode="array",
        tickvals=[
            0, 2, 4, 6, 8, 10, 12,
            14, 16, 18, 20, 22, 24,
        ],
        ticktext=[
            "00:00", "02:00", "04:00", "06:00",
            "08:00", "10:00", "12:00", "14:00",
            "16:00", "18:00", "20:00", "22:00",
            "24:00",
        ],
    )

    fig_time.update_yaxes(
        rangemode="tozero",
    )

    st.plotly_chart(
        fig_time,
        width="stretch",
    )
    
    # ============================================================
    # 3. VARIATION BY ARCHETYPE
    # ============================================================

    st.subheader("Behaviour by archetype")

    st.caption(
        "Compare the distribution of plug-in battery state across "
        "the different EV driver archetypes."
    )

    archetype_events = events.copy()

    archetype_events["plug_in_soc_pct"] = (
        archetype_events["plug_in_soc"] * 100
    )

    fig_arch = px.box(
        archetype_events,
        x="archetype",
        y="plug_in_soc_pct",
        points=False,
        title="Plug-in SoC variation by archetype",
        labels={
            "archetype": "Driver archetype",
            "plug_in_soc_pct": "Plug-in SoC (%)",
        },
    )

    fig_arch.update_layout(
        template="plotly_dark",
        xaxis_title="Driver archetype",
        yaxis_title="Plug-in SoC (%)",
    )

    fig_arch.update_yaxes(
        range=[0, 100],
    )

    st.plotly_chart(
        fig_arch,
        width="stretch",
    )

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
        st.info("This agent did not plug in on the selected date.")

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


