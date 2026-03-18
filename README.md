# League of Legends T1 Early-Game Performance Statistical Analysis

## Project Overview
This project explores the relationship between 10-minute gameplay metrics (Gold, XP, Kills) and match outcomes for the professional League of Legends team **T1**. 

Using a dataset of over 1,000 professional matches, I performed exploratory data analysis, hypothesis testing, and built a tuned Random Forest classifier to predict match results based on early-game leads.

---
Author: Jookyoung Lee

## Introduction

**League of Legends (LoL)** is a complex 5v5 multiplayer online battle arena (MOBA) where the ultimate objective is to destroy the enemy team's base. The first 10 to 15 minutes of the match—the "early game"—are primarily spent accumulating resources. Teams gather **Gold** to purchase powerful items and **Experience (XP)** to level up their champion's abilities. 

This project focuses specifically on **T1**, arguably the most storied and successful professional Esports organization in League of Legends history. Professional teams constantly balance their strategies between drafting for early-game dominance (to snowball a quick lead) versus late-game scaling (sacrificing the early game for unstoppable late-game power).

### The Central Question
This brings us to the core question of this analysis:
**How reliably do 10-minute early-game performance metrics (specifically gold, experience, creep score, and kill differentials) predict the overall match outcome (Win or Loss) for T1?**

### Why This Matters
For esports analysts, coaches, and passionate fans, understanding a team's true win conditions is crucial. If early-game leads are highly predictive of T1's success, it suggests their playstyle heavily relies on gaining early momentum—meaning they should prioritize drafting early-game dominant champions. Conversely, if 10-minute metrics have little correlation with their win rate, it highlights a high level of late-game resilience and superior team-fighting execution, allowing them to confidently draft scaling compositions even if they fall behind early. Bridging the gap between raw gameplay data and statistical prediction helps us understand the anatomy of a world-class team.

### The Dataset
The original dataset was sourced from **Oracle's Elixir**, which provides comprehensive, professional League of Legends match data. For the scope of this project, the data was strictly filtered to matches involving T1.

There are originally 120636 rows in this dataset, and the cleaned dataset contains **190 rows** (where each row represents a single professional match played by T1). The columns most relevant to answering our central question are described below:

| Column Name | Description |
| :--- | :--- |
| `result` | The final outcome of the match for T1 (`True` for a Win, `False` for a Loss). This is our target variable. |
| `side` | The side of the map T1 was assigned to play on (`Blue` or `Red`). |
| `league` | The professional tournament the match took place in (e.g., LCK, Worlds). |
| `golddiffat10` | The total gold difference between T1 and their opponent at the 10-minute mark. Positive values mean T1 was ahead. |
| `xpdiffat10` | The total experience (XP) difference between T1 and their opponent at the 10-minute mark. |
| `csdiffat10` | The Creep Score (minions killed) difference at 10 minutes. |
| `killsat10` | The total number of kills secured by T1 by the 10-minute mark. |
