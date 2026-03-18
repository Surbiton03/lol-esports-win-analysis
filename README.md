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

There are originally 120636 rows in this dataset, and the cleaned dataset contains **187 rows** (where each row represents a single professional match played by T1). The columns most relevant to answering our central question are described below:

| Column Name | Description |
| :--- | :--- |
| `result` | The final outcome of the match for T1 (`True` for a Win, `False` for a Loss). This is our target variable. |
| `side` | The side of the map T1 was assigned to play on (`Blue` or `Red`). |
| `league` | The professional tournament the match took place in (e.g., LCK, Worlds). |
| `golddiffat10` | The total gold difference between T1 and their opponent at the 10-minute mark. Positive values mean T1 was ahead. |
| `xpdiffat10` | The total experience (XP) difference between T1 and their opponent at the 10-minute mark. |
| `csdiffat10` | The Creep Score (minions killed) difference at 10 minutes. |
| `killsat10` | The total number of kills secured by T1 by the 10-minute mark. |

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning and the Data Generating Process
To prepare the raw Oracle's Elixir dataset for accurate predictive modeling, I performed several targeted data cleaning steps. Each step was designed to address how the League of Legends esports data is structurally generated and recorded by the underlying API:

**1. Filtering for Team-Level Aggregation (`position == 'team'`)**
* **The Data Generating Process:** The Oracle's Elixir dataset logs 12 rows for every single professional match: one for each of the 10 individual players, plus two summary rows that aggregate the overall Blue and Red teams. 
* **Impact on Analysis:** I filtered the dataset to strictly include rows where `teamname == 'T1'` and `position == 'team'`. If I had not done this, my model would have treated the 5 individual T1 players as 5 independent observations for the exact same match. This would have artificially inflated the sample size by 500% and caused severe data leakage, as the match `result` would be duplicated repeatedly.

**2. Feature Selection to Prevent Data Leakage**
* **The Data Generating Process:** The dataset contains hundreds of columns tracking the entire lifespan of a game, including end-of-game statistics like total game duration, final gold, and nexus kills. 
* **Impact on Analysis:** Because my research question strictly focuses on the *predictive power of the early game*, I dropped all columns except for the 10-minute telemetry data (`golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`), the target variable (`result`), and necessary metadata. This isolates the timeline, ensuring the model cannot "cheat" by looking at late-game metrics to predict the winner.

**3. Handling Structural Missingness**
* **The Data Generating Process:** Granular 10-minute telemetry relies on advanced Riot Games API tracking. While major tier-1 leagues (like the LCK or World Championship) consistently capture this telemetry, minor or amateur leagues often lack this infrastructure, resulting in `NaN` values for early-game stats.
* **Impact on Analysis:** As proven in my missingness permutation test, the absence of this data is Missing at Random (MAR) dependent on the `league`. I chose to drop rows with missing 10-minute features (`t1_clean.dropna()`). This appropriately focuses my analysis entirely on fully-tracked, high-tier professional play, removing noisy or incomplete amateur data.

**4. Data Type Formatting**
* **Impact on Analysis:** To ensure compatibility with `scikit-learn` classification pipelines, I cast the `date` column to proper datetime objects and formally converted the binary `result` column into boolean values (`True` for Wins, `False` for Losses).

After executing these steps, the final cleaned dataset consists of **187 rows** and **9 columns**. 

### Cleaned DataFrame Head
Below are the first five rows of the cleaned dataset used for modeling:

| league | side | result | golddiffat10 | xpdiffat10 | csdiffat10 | killsat10 |
|:-------|:-----|:-------|-------------:|-----------:|-----------:|----------:|
| LCK    | Red  | False  |        -1026 |        166 |         -8 |         1 |
| LCK    | Blue | True   |          657 |         45 |         25 |         0 |
| LCK    | Red  | False  |          691 |       -242 |        -11 |         3 |
| LCK    | Red  | True   |          944 |       1010 |          6 |         3 |
| LCK    | Red  | True   |         3707 |       2419 |         59 |         3 |

### Univariate Analysis

This interactive histogram displays the distribution of T1's gold difference at the 10-minute mark across all analyzed matches. The distribution is centered slightly to the right of zero, indicating a trend where T1 more frequently secures a positive gold lead in the early game rather than falling behind.

<iframe
  src="assets/univariate_golddiffat10.html"
  width="800"
  height="600"
  frameborder="0"></iframe>

### Bivariate Analysis

This scatter plot illustrates the relationship between T1's Gold Difference and Experience (XP) Difference at the 10-minute mark, with the data points colored by the final match result. There is a strong positive correlation between early gold and XP leads, and the distinct clustering of yellow "Win" points in the top-right quadrant demonstrates that when T1 secures an advantage in both resources early on, they are highly likely to win the match.

<iframe
  src="assets/bivariate_scatter.html"
  width="800"
  height="600"
  frameborder="0"></iframe>

### Interesting Aggregates

To further explore how map side interacts with T1's early-game performance and match outcomes, I created a pivot table calculating the average 10-minute gold difference, grouped by map side (Blue vs. Red) and the final match result (Win vs. Loss).

**Pivot Table: Average Gold Difference at 10 Minutes by Side & Result**

| Side | Loss (False) | Win (True) |
|:-----|-------------:|-----------:|
| Blue |      -124.14 |     502.95 |
| Red  |      -364.78 |     394.00 |

**Significance:** This pivot table reveals that T1's early-game gold difference is heavily tied to both the map side and the match outcome. Notably, T1 performs significantly better in the early game when playing on the Blue side. Even in their eventual losses, their average 10-minute gold deficit on the Blue side (-124 gold) is far less severe than their deficit on the Red side (-365 gold). This suggests the Blue side offers T1 a much more stable early game.

## Assessment of Missingness

### MNAR Analysis
In this dataset, I believe the **league** column is likely **MNAR (Missing Not At Random)**.

**Reasoning:**
The missingness of the league name often depends on the value of the league itself. In competitive League of Legends data collection (such as Oracle's Elixir), matches from major, premier leagues (like the LCK, LPL, or LCS) are recorded with 100% accuracy because they are the highest priority for data analysts. However, matches from amateur, independent, or "Tier 3" tournaments frequently lack a league name because those leagues are small, unofficial, or not recognized by primary data-tracking APIs. Therefore, the fact that a league is a "minor" or "amateur" league is the direct reason why its name is missing from the record.

**Additional Data to make it MAR:**
To move this column from **MNAR** to **MAR (Missing At Random)**, I would want to obtain a column for **tournament_organizer_type**. If we had a column identifying whether the organizer was "Riot Games Official" versus an "Independent Community Organizer," we might find that the missingness of the league name is purely dependent on the organizer type. By accounting for this third variable, the missingness would no longer depend on the league name itself, effectively making the data MAR.
