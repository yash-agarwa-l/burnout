#  MotoGP Lap Time Prediction using LightGBM

This project aims to predict the **Lap Time (in seconds)** for MotoGP riders using historical race data and engineered features. The solution leverages LightGBM for efficient regression modeling with proper handling of categorical data and domain-specific features.

---

## 📂 Dataset Description

The dataset includes historical MotoGP race data with details like:
- Rider, team, bike
- Track, weather, temperature
- Tire compounds and degradation
- Championship performance stats

### Files:
- `train.csv`: Contains training data with actual `Lap_Time_Seconds`
- `test.csv`: Contains test data without `Lap_Time_Seconds`
- `sample_submission.csv`: Format for the final predictions

##Feature Engineering

Key derived features:
- **LapTime_Estimate**: Estimated time using `Distance / Speed`
- **Points_per_Year**: Championship points per year active
- **Finish_Rate, Podium_Rate, Win_Rate**: Performance metrics
- **Avg_Temp**: Average of ambient and track temperatures
- **Speed_Degradation**: Speed loss due to tire degradation
- **Temp_Condition**: Effect of temperature on track condition
- **Corners_per_Km**: Corner density
## Preprocessing Steps

- Missing values filled with `-1`
- Categorical columns label encoded
- Combined train + test data for consistent encoding
- Feature scaling not needed due to LightGBM’s robustness
## Model Used

- **Model**: LightGBM Regressor (`LGBMRegressor`)
- **Hyperparameters**:
  - `n_estimators`: 2000
  - `learning_rate`: 0.025
  - `max_depth`: 12
  - `num_leaves`: 100
  - `subsample`: 0.85
  - `colsample_bytree`: 0.8
  - `reg_alpha`: 1.5, `reg_lambda`: 2.0
## Evaluation

- **Metric**: RMSE (Root Mean Squared Error)
- Validation is done using an 80-20 train/validation split
## How to Run

1. **Install dependencies** (if running locally):
   ```bash
   pip install pandas numpy lightgbm scikit-learn
