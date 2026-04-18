import pandas as pd
import numpy as np
import os
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
DATA_DIR = BASE_DIR / "data"

os.makedirs(DATA_DIR, exist_ok=True)

# LOAD FILES
boxscores = pd.read_csv(RAW_DATA_DIR / "team_boxscores.csv")
games = pd.read_csv(RAW_DATA_DIR / "full_season_games.csv")
elo = pd.read_csv(RAW_DATA_DIR / "elo_ratings.csv")

# CLEAN
boxscores["team"] = boxscores["team"].str.strip()
boxscores["game_id"] = boxscores["game_id"].astype(str)

# BASIC FEATURES
boxscores["off_eff"] = boxscores["points"] / boxscores["possessions"] * 100

# BUILD MATCHUPS
opp = boxscores.copy()
opp = opp.rename(columns={
    "team": "opponent",
    "points": "opp_points",
    "possessions": "opp_possessions"
})

game_matchups = boxscores.merge(opp, on="game_id")
game_matchups = game_matchups[game_matchups["team"] != game_matchups["opponent"]]

game_matchups["def_eff"] = game_matchups["opp_points"] / game_matchups["opp_possessions"] * 100

# TEAM STATS
team_stats = game_matchups.groupby("team").agg({
    "off_eff": "mean",
    "def_eff": "mean",
    "possessions": "mean"
}).reset_index()

team_stats.columns = ["Team", "off_eff", "def_eff", "possessions"]

# MERGE ELO
team_stats = team_stats.merge(elo, left_on="Team", right_on="team", how="left")
team_stats["Elo"] = team_stats["rating"].fillna(1500)

# SAVE
team_stats.to_csv(DATA_DIR / "team_stats_current.csv", index=False)

team_rankings = team_stats.sort_values("Elo", ascending=False).reset_index(drop=True)
team_rankings.insert(0, "Rank", range(1, len(team_rankings) + 1))
team_rankings.to_csv(DATA_DIR / "team_rankings.csv", index=False)

metadata = {
    "teams": len(team_stats)
}

with open(DATA_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f)

print("Pipeline complete")