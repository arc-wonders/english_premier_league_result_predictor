import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
import requests
import time
import threading

def ping_server():
    """Background task to ping the server every 5 minutes to prevent sleep in free tier"""
    while True:
        try:
            # Use the specific Streamlit app URL
            url = "https://arc-wonders-english-premier-league-result-predictor-app-d8jlqp.streamlit.app/"
            requests.get(url)
            time.sleep(300)  # Wait for 5 minutes
        except Exception:
            pass  # Ignore any errors and continue pinging

# Start the background health check thread
if not st.session_state.get('health_check_started'):
    health_check_thread = threading.Thread(target=ping_server, daemon=True)
    health_check_thread.start()
    st.session_state['health_check_started'] = True

# Set page config
st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide"
)

# Historical winners data
HISTORICAL_WINNERS = {
    "2020-21": "Man City",
    "2021-22": "Man City",
    "2022-23": "Man City",
    "2023-24": "Man City",
    "2024-25": "Liverpool",
}

def get_season_status(season_tag):
    """Determine if a season is historical, current, or future"""
    season_start = int(season_tag.split('-')[0])
    current_year = 2025  # Hardcoded as per context
    
    if season_start < current_year:
        return "historical"
    elif season_start == current_year:
        return "current"
    else:
        return "future"

# Load manifest to get available seasons
ARTIFACT_DIR = Path("artifacts")
with open(ARTIFACT_DIR / "manifest.json") as f:
    manifest = json.load(f)

# Helper function to load season artifacts
def load_season_artifacts(season_tag):
    season_dir = ARTIFACT_DIR / season_tag
    model = joblib.load(season_dir / "model.joblib")
    scaler = joblib.load(season_dir / "scaler.joblib")
    with open(season_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)
    test_df = pd.read_csv(season_dir / "test_df.csv")
    baseline_champs = pd.read_csv(season_dir / "baseline_champs.csv")
    return model, scaler, feature_cols, test_df, baseline_champs

def apply_team_deltas(test_df, feature_cols, team, elo_d=0.0, r5_d=0.0, r10_d=0.0):
    Xmod = test_df.copy()
    
    def add_if_exists(cols, mask, delta):
        for c in cols:
            if c in Xmod.columns:
                Xmod.loc[mask, c] = Xmod.loc[mask, c] + delta

    hmask = Xmod['HomeTeam_name'].eq(team)
    amask = Xmod['AwayTeam_name'].eq(team)

    # Elo adjustments
    add_if_exists(['elo_home'], hmask, elo_d)
    add_if_exists(['elo_away'], amask, elo_d)

    # Rolling PPG adjustments
    add_if_exists(['roll5_ppg_home'], hmask, r5_d)
    add_if_exists(['roll10_ppg_home'], hmask, r10_d)
    add_if_exists(['roll5_ppg_away'], amask, r5_d)
    add_if_exists(['roll10_ppg_away'], amask, r10_d)

    X_features = Xmod[feature_cols].copy()
    meta_cols = ['HomeTeam_name', 'AwayTeam_name']
    meta = Xmod[meta_cols].copy()
    return X_features, meta

def simulate_championship(proba, meta_df, n_sims=5000, seed=42):
    rng = np.random.default_rng(seed)
    teams = pd.unique(pd.concat([meta_df['HomeTeam_name'], meta_df['AwayTeam_name']]))
    team_idx = {t:i for i,t in enumerate(teams)}
    champ_counts = np.zeros(len(teams), dtype=int)

    for _ in range(n_sims):
        pts = np.zeros(len(teams), dtype=int)
        outcomes = [rng.choice([0,1,2], p=p) for p in proba]
        for i, row in meta_df.reset_index(drop=True).iterrows():
            h = team_idx[row['HomeTeam_name']]
            a = team_idx[row['AwayTeam_name']]
            o = outcomes[i]
            if o == 0:      # Home win
                pts[h] += 3
            elif o == 1:    # Draw
                pts[h] += 1; pts[a] += 1
            else:           # Away win
                pts[a] += 3
        champ_counts[np.argmax(pts)] += 1

    out = pd.DataFrame({'Team': teams, 'ChampProb': champ_counts/n_sims})
    return out.sort_values('ChampProb', ascending=False).reset_index(drop=True)

# Streamlit UI
st.title("⚽ Premier League Predictor")
st.markdown("""
This app lets you simulate how changes in team performance metrics might affect their chances 
of winning the Premier League. Select a season and adjust team strength parameters to see the impact.
""")

# Sidebar controls
st.sidebar.header("Simulation Controls")

# Season selection
season_tags = [s["season_tag"] for s in manifest["seasons"]]
selected_season = st.sidebar.selectbox(
    "Select Season",
    season_tags,
    format_func=lambda x: f"Season {x}"
)

# Load artifacts for selected season
model, scaler, feature_cols, test_df, baseline_champs = load_season_artifacts(selected_season)

# Team selection and parameter adjustment
teams = pd.unique(pd.concat([test_df['HomeTeam_name'], test_df['AwayTeam_name']]))
selected_team = st.sidebar.selectbox("Select Team to Analyze", sorted(teams))

st.sidebar.markdown("### Adjust Team Performance")
st.sidebar.markdown("How would the team perform if they were playing...")

# Overall strength adjustment (maps to Elo)
strength_options = {
    "No change": 0.0,
    "Slightly better than usual": 15.0,
    "Much better than usual": 30.0,
    "Slightly worse than usual": -15.0,
    "Much worse than usual": -30.0
}
strength_selection = st.sidebar.selectbox(
    "Overall Team Strength",
    options=list(strength_options.keys()),
    index=0,
    help="How much better or worse is the team performing overall?"
)
elo_delta = strength_options[strength_selection]

# Recent form adjustment (maps to 5-game PPG)
form_options = {
    "No change in form": 0.0,
    "In great form": 0.3,
    "In good form": 0.15,
    "In poor form": -0.15,
    "In terrible form": -0.3
}
form_selection = st.sidebar.selectbox(
    "Recent Form (Last 5 Games)",
    options=list(form_options.keys()),
    index=0,
    help="How well has the team been playing in their most recent matches?"
)
roll5_delta = form_options[form_selection]

# Consistent performance adjustment (maps to 10-game PPG)
consistency_options = {
    "No change in consistency": 0.0,
    "Very consistent good results": 0.2,
    "Somewhat consistent good results": 0.1,
    "Somewhat inconsistent results": -0.1,
    "Very inconsistent results": -0.2
}
consistency_selection = st.sidebar.selectbox(
    "Consistent Performance (Last 10 Games)",
    options=list(consistency_options.keys()),
    index=0,
    help="How consistently has the team been performing over a longer period?"
)
roll10_delta = consistency_options[consistency_selection]

st.sidebar.markdown("### Simulation Settings")
simulation_options = {
    "Quick (less accurate)": 1000,
    "Balanced": 5000,
    "Thorough (more accurate)": 10000
}
sim_selection = st.sidebar.selectbox(
    "Simulation Accuracy",
    options=list(simulation_options.keys()),
    index=1,
    help="How thoroughly should we analyze the scenarios? More thorough = more accurate but slower"
)
n_sims = simulation_options[sim_selection]

# Hide the random seed in an expander since it's technical
with st.sidebar.expander("Advanced Settings"):
    sim_seed = st.number_input("Simulation Seed", min_value=1, max_value=999999, value=42, 
                              help="This controls the randomness in the simulation. Change it to get slightly different results.")

# Main content
st.subheader("Season Overview")

# Get season status and display appropriate context
season_status = get_season_status(selected_season)
if season_status == "historical":
    actual_winner = HISTORICAL_WINNERS.get(selected_season)
    st.info(f"Historical Season - Actual Winner: **{actual_winner}**", icon="📅")
elif season_status == "current":
    st.warning("Current Season - In Progress", icon="🏃")
else:
    st.info("Future Season - Prediction Only", icon="🔮")

# Show base model predictions first
st.subheader("Model's Base Predictions")
baseline_df = baseline_champs.head(10)
st.bar_chart(baseline_df.set_index('Team')['ChampProb'])

# Display table with probabilities and historical context
st.markdown("### Detailed Prediction Analysis")
baseline_table = baseline_df.copy()
baseline_table['Probability'] = baseline_table['ChampProb'].map('{:.1%}'.format)
baseline_table = baseline_table.drop('ChampProb', axis=1)

if season_status == "historical":
    baseline_table['Actual Outcome'] = baseline_table['Team'].map(
        lambda x: "🏆 Champion" if x == HISTORICAL_WINNERS.get(selected_season) else ""
    )
st.dataframe(baseline_table, hide_index=True)

# What-if Simulation section
st.markdown("---")
st.subheader("What-If Simulator")
st.markdown("Adjust team parameters in the sidebar to simulate alternative scenarios.")

if st.sidebar.button("Run What-If Simulation"):
    with st.spinner("Running simulation..."):
        # Baseline display
        st.subheader("Baseline Championship Probabilities")
        baseline_df = baseline_champs.head(10)
        st.bar_chart(baseline_df.set_index('Team')['ChampProb'])
        
        # What-if simulation
        X_mod, meta = apply_team_deltas(
            test_df, feature_cols, selected_team, 
            elo_delta, roll5_delta, roll10_delta
        )
        X_mod_s = scaler.transform(X_mod.values)
        proba_mod = model.predict_proba(X_mod_s)
        
        whatif_champs = simulate_championship(proba_mod, meta, n_sims=n_sims, seed=sim_seed)
        
        # Compare results
        st.subheader(f"Modified Probabilities (with {selected_team} adjustments)")
        whatif_df = whatif_champs.head(10)
        st.bar_chart(whatif_df.set_index('Team')['ChampProb'])
        
        # Detailed comparison table
        st.subheader("Detailed Comparison (Top 10)")
        comparison = baseline_df.merge(
            whatif_df, 
            on='Team', 
            suffixes=('_baseline', '_modified')
        )
        comparison['Δ Probability'] = comparison['ChampProb_modified'] - comparison['ChampProb_baseline']
        comparison_formatted = comparison.copy()
        
        # Add actual winner information for historical seasons
        if season_status == "historical":
            comparison_formatted['Actual Outcome'] = comparison_formatted['Team'].map(
                lambda x: "🏆 Champion" if x == HISTORICAL_WINNERS.get(selected_season) else ""
            )
        
        for col in ['ChampProb_baseline', 'ChampProb_modified', 'Δ Probability']:
            comparison_formatted[col] = comparison_formatted[col].map('{:.1%}'.format)
        
        formatted_cols = {
            'Team': 'Team',
            'ChampProb_baseline': 'Base Prediction',
            'ChampProb_modified': 'Modified Prediction',
            'Δ Probability': 'Change'
        }
        if season_status == "historical":
            formatted_cols['Actual Outcome'] = 'Actual Outcome'
            
        st.dataframe(
            comparison_formatted.rename(columns=formatted_cols),
            hide_index=True
        )

        # Show modification summary
        st.sidebar.markdown("### Applied Changes")
        st.sidebar.markdown(f"""
        **Team Performance Changes:**
        - Overall Strength: {strength_selection}
        - Recent Form: {form_selection}
        - Consistent Performance: {consistency_selection}
        
        **Analysis Depth:**
        - {sim_selection}
        """)
else:
    st.info("👈 Adjust the parameters in the sidebar and click 'Run Simulation' to see results")

# Footer
st.markdown("---")
st.markdown("""
*Note: This simulator uses machine learning models trained on historical Premier League data. 
The modifications affect the team's underlying metrics (Elo rating and rolling performance stats) 
to estimate how changes in form might impact their title chances.*
""")