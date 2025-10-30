# EPL Champion Predictor (2020 – 2025)
## Overview

This project predicts and simulates English Premier League (EPL) outcomes using historical match data from football-data.co.uk (2020 – 2025).
It builds season-specific machine-learning models to estimate the probability of each match outcome (Home Win / Draw / Away Win) and then runs Monte Carlo simulations to estimate each team’s probability of becoming champion.

The project consists of:

A data-preparation and modeling notebook that downloads raw season data, computes features, trains models, and saves artifacts.

A Streamlit web app (app.py) that loads those artifacts, visualizes champion probabilities, and lets you run what-if experiments and ongoing (as-of) simulations.

## Dataset
### Source

All data comes from football-data.co.uk
, which provides public CSVs for each Premier League season.

Files used

The notebook automatically downloads:

Season	URL	Example File
2020-21	https://www.football-data.co.uk/mmz4281/2021/E0.csv	E0.csv
2021-22	https://www.football-data.co.uk/mmz4281/2122/E0.csv	E0.csv
2022-23	https://www.football-data.co.uk/mmz4281/2223/E0.csv	E0.csv
2023-24	https://www.football-data.co.uk/mmz4281/2324/E0.csv	E0.csv
2024-25	https://www.football-data.co.uk/mmz4281/2425/E0.csv	E0.csv

Each CSV includes columns such as:

Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, HS, AS, HST, AST, HC, AC, ...


Where FTR = Full-Time Result (H, D, A).

## Feature Engineering

For each match, the script constructs features summarizing team strength and form:

Feature	Description
elo_home, elo_away	Rolling Elo ratings reflecting team strength and past results
roll5_ppg_home, roll5_ppg_away	Average points per game over the last 5 matches
roll10_ppg_home, roll10_ppg_away	Average points per game over the last 10 matches

These capture both long-term quality (Elo) and short-term form (rolling PPG).
The features are standardized using a StandardScaler.

## Model Training

For each season S:

Training data = all previous seasons (< S).

Test data = fixtures within season S.

Target (label) =

0 → H (Home win)

1 → D (Draw)

2 → A (Away win)

### Algorithm

Model: XGBoostClassifier

Loss: multi-class log-loss

Metrics saved:

Log Loss (lower = better)

Brier Score (lower = better)

Each season’s model is trained, evaluated, and stored under artifacts/{season-tag}/.

## Monte Carlo Simulation
### Step 1: Predict match probabilities

The trained model outputs P(Home Win), P(Draw), P(Away Win) for each fixture.

### Step 2: Simulate the league

For each of N = 5 000 iterations:

Randomly draw an outcome for every fixture according to its predicted probabilities.

Assign points (3 / 1 / 0).

Record which team finishes with the most points.

Champion probability = (times team won) / N.

This gives a probability distribution over champions, not just a single predicted winner.

## Artifacts Generated

Each folder under artifacts/ corresponds to a season (e.g. 2023-24) and contains:

File	Purpose
model.joblib	Trained XGBoost model
scaler.joblib	Fitted StandardScaler
feature_cols.json	Feature names used for inference
test_df.csv	Fixtures with features & team names
baseline_champs.csv	Monte Carlo champion probabilities
(root) manifest.json	Summary of all seasons (fixtures, logloss, brier)

Example manifest snippet:

{
  "seasons": [
    {
      "season_start": 2023,
      "season_tag": "2023-24",
      "fixtures": 380,
      "logloss": 0.732,
      "brier": 0.145
    }
  ]
}

## Streamlit Web App

Run locally:

streamlit run app.py

Features

Baseline probabilities — View pre-saved champion odds for each season.

Ongoing (as-of) — Choose a date; the app computes current league points (from actual results) and simulates only remaining fixtures.

What-If analysis — Boost a team’s Elo and/or form, recompute probabilities, and compare What-If vs Baseline.

Downloads — Export all tables to CSV.

## How It All Works (Step-by-Step)

Data Load
Each season CSV is parsed and cleaned.

Feature Engineering

Elo ratings are updated match-by-match using prior results.

Rolling PPG features are computed via moving averages.

Model Training

Features and labels are standardized and fit into XGBoost.

Evaluation metrics (logloss / brier) are recorded.

Simulation

Probabilities → Monte Carlo championship likelihoods.

Artifact Persistence
Everything (model, scaler, features, test data, baseline probs) is saved for later reuse.

Streamlit Interaction

User selects season or date.

App loads artifacts, performs any requested re-simulation, and visualizes the outcomes.

## Example Output
Team	Champ Prob
Man City	0.48
Arsenal	0.36
Liverpool	0.13
Aston Villa	0.03

(Based on 2024-25 data at the time of training.)

## Notes & Limitations

No tie-breakers (goal difference / head-to-head).

Predictions reflect probabilities, not deterministic picks.

Quality depends on how well Elo + form capture real team dynamics.

No external features (xG, injuries, transfers, weather).

Training uses historical seasons → forecasts can lag recent shocks (e.g. manager changes).

## Future Extensions

Integrate richer features (xG, possession, injuries).

Calibrate probabilities (Platt or Isotonic).

Visualize per-match prediction distributions.

Add multi-team “scenario planning” in the web app.

## Tech Stack
Layer	Tools
Data Source	football-data.co.uk CSV feeds
Feature Engineering	pandas, numpy
Modeling	XGBoost, scikit-learn
Visualization	Streamlit, matplotlib
Persistence	joblib, JSON, CSV
## Summary

This pipeline turns raw EPL results into interpretable, probabilistic champion forecasts by combining:

Historical data → Feature engineering → Model training → Simulation

Reusable artifacts → Interactive web app for ongoing and what-if analysis