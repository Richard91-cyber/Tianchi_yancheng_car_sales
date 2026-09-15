# Tianchi Yancheng Passenger Car Sales Prediction

> Predicting monthly sales volume for each car model (`class_id`) in Yancheng, China for November 2017, using historical sales records from 2012 onwards. Built for the **Tianchi Yancheng Passenger Car Sales Volume Prediction (天池盐城乘用车销售量预测)** competition.

***

## 📌 Overview

| Item               | Detail                                                                       |
| ------------------ | ---------------------------------------------------------------------------- |
| **Competition**    | Tianchi Yancheng Passenger Car Sales Volume Prediction (天池盐城乘用车销售量预测)        |
| **Host**           | Alibaba / Tianchi                                                            |
| **Task Type**      | Regression — predict next-month sales volume per car model                   |
| **Target**         | November 2017 sales volume per `class_id`                                    |
| **Metric**         | RMSE (Root Mean Squared Error)                                               |
| **Score**          | RMSE ≈ 180 (peaked at rank \~91 / 2,500 on the B-leaderboard)                |
| **Final Rank**     | 139 / 2,500 (dropped from peak due to inactive last two days of the contest) |
| **Core Technique** | Two-model split (high-volume vs. low-volume) + Stacking ensemble             |
| **Language**       | Python (pandas / numpy / xgboost / scikit-learn / mlxtend)                   |

***

## 🧠 Problem Statement

The city of Yancheng publishes monthly sales volumes for every car model (`class_id`) sold in the region. Given historical monthly sales from 2012 through October 2017, the task is to **predict the sales volume of each** **`class_id`** **for November 2017**.

The challenge is threefold:

1. **Heavy-tailed targets** — most `class_id`s sell fewer than 1,000 units, but a handful sell far more. Because the metric is RMSE, errors on high-volume models dominate the score.
2. **Sparse per-model history** — many `class_id`s have only a few months of records.
3. **Seasonal/trend noise** — earlier years (2012–2014) exhibit different sales dynamics from 2015–2017.

***

## 📂 Dataset

Two CSV files (sitting in the parent directory, referenced via `r'..\xxx.csv'`):

| File                                | Role          | Key Columns                                                                                                      |
| ----------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------- |
| `[new] yancheng_train_20171226.csv` | Training data | `class_id`, `sale_date`, `sale_quantity`, plus \~30 car-attribute columns (dimensions, engine, drivetrain, etc.) |
| `yancheng_testA_20171225.csv`       | Test set      | `class_id`, `predict_date`, `predict_quantity` (to be filled)                                                    |

Car-attribute columns include physical dimensions (`car_length`, `car_width`, `car_height`), powertrain specs (`displacement`, `power`, `engine_torque`, `cylinder_number`, `gearbox_type`), chassis (`wheelbase`, `front_track`, `rear_track`, `total_quality`), and categorical tags (`brand_id`, `price_level`, `if_MPV_id`, `if_luxurious_id`, `compartment`, `if_charging`, `type_id`, `department_id`).

***

## 🏗️ Solution Architecture

```
        ┌──────────────────────────────────────────────────────┐
        │     yancheng_train_20171226.csv (raw monthly data)  │
        └────────────────────────────┬─────────────────────────┘
                                     │
                ┌────────────────────▼─────────────────────┐
                │  Split by single-transaction sale_quantity│
                │     • >100 → high-volume pipeline         │
                │     • ≤100 → low-volume pipeline          │
                └────────────────────┬─────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
   ┌──────────▼──────────┐                      ┌────────────▼───────────┐
   │  ReadTrain_100plus  │                      │  ReadTrain_100minus   │
   │  ReadPre_100plus    │                      │  ReadPre_100minus     │
   │  (feat: 2016-07~10) │                      │  (feat: 2016-07~10)   │
   │  (label: 2016-11)   │                      │  (label: 2016-11)     │
   └──────────┬──────────┘                      └────────────┬───────────┘
              │                                              │
              └──────────────────────┬──────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   FEATURE.py / Feat  │
                          │  per-class_id &      │
                          │  per-(class,attr)    │
                          │  aggregations        │
                          └──────────┬───────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │                                     │
          ┌───────▼────────┐                  ┌────────▼────────┐
          │  XgbPre (XGB)  │                  │ Ensemble        │
          │  reg:linear    │                  │ (Stacking:      │
          │  eta=0.01      │                  │  GBDT+SVR+LR+   │
          │  n=1000        │                  │  RF+XGB → XGB)  │
          └───────┬────────┘                  └────────┬────────┘
                  │                                     │
                  └─────────────────┬───────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Submission file    │
                          │  (predict_quantity) │
                          └────────────────────┘
```

***

## 🔧 Feature Engineering (`FEATURE.py`)

A single `Feat(dataset_x, dataset_y)` function aggregates \~150 features from the raw monthly records. All features are computed **per** **`class_id`** — sometimes grouped by an additional categorical attribute.

### 1. Per-`class_id` Continuous-Attribute Statistics

For each numeric attribute, the mean / median / max / min / var (where applicable) is computed per `class_id`:

| Attribute                               | Description                                                        |
| --------------------------------------- | ------------------------------------------------------------------ |
| `car_length`, `car_width`, `car_height` | Physical dimensions                                                |
| `displacement`                          | Engine displacement                                                |
| `power`                                 | Engine power (filtered for anomalous `"81/70"` entries)            |
| `engine_torque`                         | Engine torque (NaN-as-`"-"` cleaned)                               |
| `total_quality`                         | Vehicle total mass                                                 |
| `wheelbase`                             | Wheelbase                                                          |
| `front_track`, `rear_track`             | Front / rear track widths                                          |
| `price`                                 | List price — `"-"` entries filled with 5 (a domain-driven default) |

### 2. Per-`(class_id, categorical)` Statistics

For each categorical attribute, the script computes sum / median / max / min / mean / var of `sale_quantity`, plus the **share of each category within the class**:

| Categorical       | Categories                  |
| ----------------- | --------------------------- |
| `brand_id`        | brand of the car            |
| `price_level`     | 9 price tiers               |
| `if_MPV_id`       | MPV flag (1/2)              |
| `if_luxurious_id` | luxury flag (1/2)           |
| `compartment`     | 3 compartment tiers (1/2/3) |
| `if_charging`     | Turbo flag (L/T)            |
| `type_id`         | 4 model categories          |
| `department_id`   | 7 model-series categories   |

For `price_level`, `if_luxurious_id`, `compartment`, `if_charging`, the script computes **two ratio variants**: share-within-category (e.g. `L_ratio1 = L / sum(L)`) and share-within-class (e.g. `L_ratio2 = L / (L + T)`) — capturing both categorical popularity and within-class composition.

### Design Principles

- **Per-`class_id`** **granularity** — the prediction unit is `class_id`, so all aggregations roll up to that level.
- **Multi-statistic coverage** — every numeric attribute is summarized with 4–5 statistics (mean/median/max/min/var) to capture both central tendency and dispersion.
- **Two ratio variants** — for categoricals, share-within-category captures "how popular is this brand across all classes," while share-within-class captures "how much of this class's sales come from this brand."
- **Domain-driven cleaning** — `price` filled with 5 (not dropped), `power` outliers filtered, `engine_torque` `"-"` cleaned before parsing.

***

## 🤖 Modeling (`main.py`)

### Two-Model Split (Key Strategy)

Because the target is **heavy-tailed** and the metric is **RMSE**, the project splits the dataset into two independent pipelines:

| Pipeline                                  | Filter (per-record `sale_quantity`) | Rationale                                                                              |
| ----------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------- |
| `ReadTrain_100plus` / `ReadPre_100plus`   | `> 100`                             | Captures the high-volume `class_id`s whose RMSE contribution dominates the leaderboard |
| `ReadTrain_100minus` / `ReadPre_100minus` | `≤ 100`                             | Captures the long tail of low-volume `class_id`s                                       |

Each pipeline trains independently and predicts its own slice; the two prediction sets are concatenated for the final submission.

### Models

| Model                 | Hyperparameters                                                                               | Role                          |
| --------------------- | --------------------------------------------------------------------------------------------- | ----------------------------- |
| **XGBoost**           | `objective=reg:linear`, `eta=0.01`, `max_depth=5`, `num_boost_round=1000`, `eval_metric=rmse` | Primary regressor (`XgbPre`)  |
| **Stacking ensemble** | 5 base regressors → 1 meta-regressor                                                          | Experimental — see note below |

### Stacking Configuration (`Ensemble`)

```python
stregr = StackingRegressor(
    regressors=[gbdt, svr_rbf, lr, rf, xgbr],
    meta_regressor=xgbr
)
```

| Layer | Models            | Hyperparameters                                                       |
| ----- | ----------------- | --------------------------------------------------------------------- |
| Base  | GBDT              | `lr=0.01`, `max_depth=5`, `n_estimators=3500`                         |
| Base  | XGBoost           | `max_depth=5`, `lr=0.01`, `n_estimators=1000`, `objective=reg:linear` |
| Base  | Random Forest     | `max_depth=5`, `n_estimators=100`, `criterion=mse`                    |
| Base  | SVR (RBF)         | default                                                               |
| Base  | Linear Regression | default                                                               |
| Meta  | XGBoost           | same as base XGBoost                                                  |

> **Note (from the original README):** Stacking underperformed — likely because the base regressors were not precise enough individually. The author notes that a more effective stacking setup would use XGB / LGB / GBDT / RF as bases and a simple Linear Regression as the meta-learner.

***

## ⏱️ Validation Strategy

A **time-based sliding window**, identical in structure to the test split:

| Split                  | Feature Window    | Label Window |
| ---------------------- | ----------------- | ------------ |
| **Train**              | 2016-07 → 2016-10 | 2016-11      |
| **Validation**         | 2015-07 → 2015-10 | 2015-11      |
| **Test (leaderboard)** | 2017-07 → 2017-10 | 2017-11      |

### Why Only Jul–Oct, Not All History?

Initially all months before November were used as features. Visualization revealed that the **Jul–Oct sales patterns of 2015, 2016, and 2017 are highly correlated**, while earlier years (2012–2014) introduced noise. Restricting the feature window to Jul–Oct alone dropped RMSE from **200+ to \~180** on the A-leaderboard — a substantial improvement.

### Class-id Alignment

The training code drops any `class_id` that appears in the label window but not in the feature window — ensuring the model only evaluates on classes for which features could be constructed.

***

## ⚙️ Engineering Highlights

- **Two-model split based on metric awareness** — the heavy-tailed target + RMSE metric means high-volume classes dominate the score. Splitting them into a separate model lets each pipeline specialize.
- **Domain-driven data cleaning** — `price` missing values filled with 5 (domain default), `power` anomalous `"81/70"` records filtered, `engine_torque` `"-"` characters stripped before float parsing.
- **Feature-dimension alignment in Stacking** — the `Ensemble` function pads the validation set with zero-columns when train and validation have different feature counts (a pragmatic fix for the column-mismatch issue that arises when some `class_id`s are absent in one split).
- **Per-`class_id`** **aggregation as the unit of analysis** — the original records are at the (class\_id × month × attribute) grain; the `Feat` function rolls them up to (class\_id × feature) so the model sees one row per `class_id`.

***

## 📁 Project Structure

```
github/
├── main.py        # Two-model split (100plus / 100minus) + XGBoost + Stacking
├── FEATURE.py     # ~150 features aggregated per class_id & per (class_id, categorical)
└── README.md      # This file
```

External data files (not committed) sit in the parent directory:

```
../
├── [new] yancheng_train_20171226.csv   # Training data (monthly sales + car attributes)
└── yancheng_testA_20171225.csv         # Test set (predict_quantity to be filled)
```

***

## 📝 Key Takeaways

- **Metric-aware model design** — splitting the dataset by single-transaction `sale_quantity` (100 as the threshold) is the single most impactful decision in this project. RMSE heavily penalizes errors on high-volume classes; isolating them in a dedicated model lets the regressor specialize.
- **Feature-window narrowing matters** — using all available history is not always better. Visualization-driven restriction of the feature window to Jul–Oct dropped RMSE from 200+ to 180 — a 10% improvement from a one-line change.
- **Stacking is not free** — the Stacking ensemble underperformed here because the base regressors were not precise enough. A safer default is XGB / LGB / GBDT / RF as bases and a simple Linear Regression as the meta-learner (which the author noted in the original README).
- **Domain knowledge in data cleaning** — filling `price` missing values with 5 (instead of dropping the row) preserved valuable `class_id`s that would otherwise have been excluded.
- **Per-`class_id`** **aggregation as the analysis unit** — rolling up monthly records to one row per `class_id` matches the prediction grain and avoids leakage from the label month.

***

## 📜 License

Personal project for educational and portfolio purposes. Dataset © Tianchi / competition organizers.
