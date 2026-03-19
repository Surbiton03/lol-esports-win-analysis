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

### Missingness Dependency

To determine if the missingness of `golddiffat10` is dependent on other variables, I conducted two permutation tests using **Total Variation Distance (TVD)** as the test statistic.

#### Test 1: Dependency on League
* **Null Hypothesis (H0):** The missingness of `golddiffat10` does not depend on the league.
* **Alternative Hypothesis (H1):** The missingness of `golddiffat10` does depend on the league.

The permutation test yielded an **observed TVD of 0.9909** with a **p-value of 0.0**. As shown in the distribution below, our observed statistic is a massive outlier compared to the null distribution.

<iframe
  src="assets/missingness_tvd.html"
  width="800"
  height="600"
  frameborder="0"></iframe>

#### Test 2: Independence from Side
* **Null Hypothesis (H₀):** The missingness of `golddiffat10` does not depend on the map side (Blue vs. Red).
* **Alternative Hypothesis (H₁):** The missingness of `golddiffat10` does depend on the map side.

This test yielded an **observed TVD of 0.0000** and a **p-value of 1.0**, indicating that missingness is completely independent of which side T1 plays on.

#### Conclusion
Based on these results, we classify the missingness of `golddiffat10` as **Missing at Random (MAR)**. The missingness is clearly tied to the `league` column; major regions like the LCK have high-fidelity data tracking, while smaller leagues often lack the infrastructure to record 10-minute gold differentials. Because the missingness can be explained by the observed `league` variable, it is MAR.

## Hypothesis Testing

To determine if T1 has an inherent early-game advantage based on map placement, I performed a permutation test evaluating their Gold Difference at 10 minutes (`golddiffat10`) across the Blue and Red sides.

### 1. Hypotheses
* **Null Hypothesis (H₀):** T1's mean gold difference at 10 minutes is the same whether they play on the Blue side or the Red side. Any observed difference in our dataset is purely due to random chance.
* **Alternative Hypothesis (H₁):** T1's mean gold difference at 10 minutes is strictly **greater** when playing on the Blue side compared to the Red side.

### 2. Test Choice and Justification
* **Test Statistic:** Difference in Means (Mean Blue Gold Diff - Mean Red Gold Diff).
* **Significance Level (α):** 0.05
* **Method:** Permutation Test with 10,000 simulations.

**Justification:** The difference in means is an ideal statistic for this question because we are comparing the central tendencies of two continuous distributions (gold differentials). A permutation test is preferred over a standard t-test because it is non-parametric; it does not assume our data follows a normal distribution, which is important for competitive gaming data that often contains performance outliers.

### 3. Results and Visualization
The permutation test yielded an **observed difference of 196.53** and a **p-value of 0.0924**.

<iframe
  src="assets/hypothesis_test.html"
  width="800"
  height="600"
  frameborder="0"></iframe>

### 4. Conclusion
With a p-value of **0.0924**, which is greater than our significance level of **0.05**, we **fail to reject the null hypothesis**. 

While the data shows that T1 averaged roughly 196 more gold on the Blue side than the Red side in this specific timeframe, the statistical evidence is not strong enough to conclude that this is a systematic advantage. This result suggests that the observed difference is reasonably likely to have occurred due to random variation in match performance rather than an inherent map-side advantage. We do not have sufficient evidence to support the claim that T1 performs significantly better on the Blue side early in the game.

## Framing a Prediction Problem

### The Prediction Problem
The goal of this project is to predict whether a team (specifically **T1**) will win or lose a match based strictly on their performance and game state at the **10-minute mark**. 

### Type of Problem
This is a **Binary Classification** problem. The model is tasked with assigning each match to one of two discrete classes: **Win** or **Loss**.

### Response Variable (Target)
The response variable is **`result`**, where a value of `True` indicates a win and `False` indicates a loss. 
* **Justification:** This variable is the ultimate "bottom line" of any competitive match. By predicting `result`, we can evaluate how much early-game momentum (gold, experience, and map pressure) actually translates into a final victory, which is the core of our research question.

### Time of Prediction & Data Leakage
To ensure the model is a valid predictive tool and not simply a retrospective summary, the "time of prediction" is set exactly at **10:00 minutes** into the match. 
* **Justification:** At this timestamp, we only have access to early-game metrics. To prevent **data leakage**, we have strictly excluded any features that would only be known after the 10-minute mark—such as total towers destroyed, total dragons slain, or end-game gold totals. The model only "sees" what a spectator or coach would see 10 minutes into a live broadcast (e.g., `golddiffat10`, `xpdiffat10`, `killsat10`).

### Evaluation Metric
The primary evaluation metric for this model is **Accuracy**, supported by the **F1-Score**.
* **Accuracy:** This is a suitable baseline because the dataset is relatively balanced; top-tier professional teams like T1 generally have win rates that don't suffer from extreme class imbalance (e.g., they aren't winning 99% or 1% of the time). 
* **Why F1-Score over just Accuracy?** While Accuracy measures overall correctness, it can be misleading if a team has a significantly high win rate; a "naive" model could achieve high accuracy by simply predicting a "Win" for every game regardless of the features. I chose to include the **F1-Score** (the harmonic mean of Precision and Recall) as a more robust metric. Unlike simple Accuracy, the F1-Score ensures that the model is penalized for both False Positives (predicting a win when they actually lose) and False Negatives (predicting a loss when they actually win), ensuring the model is truly learning the nuances of the early game rather than just the team's historical win rate.

## Baseline Model

### Model Description
For the baseline model, I built a **Logistic Regression** classifier within a `scikit-learn` Pipeline to predict our binary response variable, `result` (Win/Loss). 

### Features and Encoding
The baseline model utilizes two initial features from the 10-minute mark. To prevent data leakage, all transformations were applied inside a column transformer pipeline before fitting the model.

* **`side` (Nominal):** This categorical feature represents the map side T1 played on (Blue or Red). Because it is nominal (having no inherent mathematical order), I applied a `OneHotEncoder(drop='first')` to convert it into a binary numeric format (0 or 1) while avoiding the dummy variable trap.
* **`golddiffat10` (Quantitative):** This continuous quantitative feature represents the gold difference at 10 minutes. As per the baseline requirements, I left this feature as-is using the `passthrough` command in the transformer.

### Performance and Generalization
To ensure the model can generalize to unseen data and isn't simply overfitting to the data it learned from, I performed a `train_test_split` and evaluated the model on both the training set and the held-out testing set.

* **Training Accuracy:** 0.6510 (65.10%)
* **Testing Accuracy:** 0.6316 (63.16%)
* **Testing F1-Score:** 0.7407

### Is this model "good"?
At present, I consider this model **"adequate as a baseline, but not objectively good."** Because the Training Accuracy and Testing Accuracy are very close to each other, it proves that the model is successfully generalizing to unseen data without severe overfitting. An accuracy of ~63% means the model performs noticeably better than a random coin flip (50%), indicating that early gold leads and map side *do* hold predictive power. The F1-score of ~0.74 shows it is reasonably capable of balancing precision and recall when identifying wins. 

However, 63% accuracy is too low to be considered a highly reliable predictive tool in professional esports. Attempting to predict a highly complex match using only a single gold metric and map side ignores other massive early-game factors like objective control, experience leads, and champion scaling. To create a "good" model, we must introduce additional, engineered features.

## Final Model

### Method of Algorithm Selection
My method for selecting the **Random Forest Classifier** algorithm was based on evaluating the limitations of the baseline Logistic Regression. The baseline assumed a perfectly linear relationship between early-game stats and final outcomes, which is insufficient for the highly complex, interactive nature of League of Legends. I selected an ensemble decision tree model because it is natively equipped to capture non-linear interactions and complex thresholds without severely overfitting to the noisy esports data.

### Engineered Features and Data-Generating Rationale
To improve the model, I introduced new features and applied specific transformations based on the actual mechanics of how League of Legends is played (the data-generating process):

1. **Robust Transformation on `golddiffat10` (QuantileTransformer):** * **The Addition:** I applied a `QuantileTransformer` to map the 10-minute gold difference to a normal, Gaussian-like distribution.
   * **The Rationale:** In standard gameplay, early gold swings are relatively small and predictable. However, occasionally a chaotic "Level 1 Invade" or early team fight goes horribly wrong, resulting in a massive, abnormal gold spike for one team. These extreme outliers can heavily skew a model's understanding of a "normal" game. This transformation mitigates the impact of those rare chaotic matches, forcing the data into a distribution where the model isn't overly punished by extreme, uncharacteristic games.
2. **Standardizing `killsat10` (StandardScaler):** * **The Addition:** I incorporated the total kills at 10 minutes (`killsat10`) into the model and applied a `StandardScaler`.
   * **The Rationale:** The raw count of kills in the early game is on a drastically different, much smaller scale (e.g., 0 to 5) compared to economic metrics like gold, XP, or CS differentials (which are in the hundreds or thousands). Because machine learning algorithms often weigh larger numbers more heavily, leaving kills unscaled might cause the model to ignore early map aggression entirely. Standardizing this feature ensures it has a mean of 0 and a variance of 1, allowing the model to properly weigh early kills alongside massive economic leads.

### Method of Hyperparameter Selection
To select the optimal hyperparameters for this new algorithm, my method was utilizing **`GridSearchCV`**. This allowed me to exhaustively test different combinations of parameters using 5-fold cross-validation (`cv=5`) to ensure the model wouldn't overfit to a specific subset of the training data. 

The Grid Search identified the following as the best performing hyperparameters:
* **`max_depth`: 5** (Restricts the trees from growing too deep. A depth of 5 is the "sweet spot" that allows the model to learn complex patterns without perfectly memorizing the training data, effectively preventing overfitting.)
* **`n_estimators`: 50** (Dictates that the "forest" is made of 50 individual decision trees. This provides enough models to reduce variance and create a stable consensus without adding unnecessary computational bloat.)

### Final Model Performance and Conclusion
After fitting the optimized Random Forest Pipeline with our newly engineered features, the model was evaluated on the **unseen testing set**.

* **Baseline Testing Accuracy:** 0.6316 (63.16%)
* **Final Testing Accuracy:** 0.6842 (68.42%)

**Improvement:** The final model achieved an accuracy of **68.42%**, representing a solid **~5.2% absolute improvement** over the baseline model. Furthermore, looking at the classification report, the model is highly effective at identifying T1 victories (Recall: 0.85, F1-Score: 0.79 for the `True` class). 

By accounting for outliers with the Quantile Transformer, scaling the `killsat10` metric so early aggression is properly weighted, and utilizing a non-linear Random Forest algorithm, the final model is significantly better equipped to analyze the complex early-game state of a professional League of Legends match.

## Fairness Analysis

To ensure our final model does not possess a bias based on map placement, I conducted a fairness analysis to see if the model's predictive accuracy differs significantly depending on which side T1 plays on.

### 1. Groups and Evaluation Metric
* **Group X:** Matches where T1 played on the **Blue Side**.
* **Group Y:** Matches where T1 played on the **Red Side**.
* **Evaluation Metric:** Accuracy (Testing for Overall Parity).

### 2. Hypotheses
* **Null Hypothesis (H₀):** The model is fair. Its accuracy for Blue side games and Red side games is roughly the same, and any observed differences are purely due to random chance.
* **Alternative Hypothesis (H₁):** The model is unfair. Its accuracy for Red side games is significantly different from its accuracy for Blue side games.

### 3. Test Details
* **Test Statistic:** The Absolute Difference in Accuracy between Blue side and Red side predictions (|Accuracy<sub>Blue</sub> - Accuracy<sub>Red</sub>|).
* **Significance Level (α):** 0.05
* **Method:** Permutation Test with 1,000 simulations shuffling the `side` labels.

### 4. Results and Visualization
The baseline metrics on our test set showed a **Blue Side Accuracy of 0.6316** and a **Red Side Accuracy of 0.7368**, resulting in an **Observed Absolute Difference of 0.1053**.

<iframe
  src="assets/fairness_test.html"
  width="800"
  height="600"
  frameborder="0"></iframe>

### 5. Conclusion
The permutation test yielded a **p-value of 0.7440**. 

Because the p-value (0.7440) is significantly greater than our significance level (0.05), we **fail to reject the null hypothesis**. Visually, we can see in the distribution above that an accuracy difference of ~10.5% is extremely common just by random chance when shuffling the labels. Therefore, we can conclude that **our model is fair**. It achieves overall parity, meaning it does not predict significantly worse for T1 whether they are drafted on the Blue side or the Red side.
