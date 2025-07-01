# Tennis Match Outcome Prediction for Wimbledon 2025

This project analyzes historical tennis data to predict match outcomes for the Wimbledon 2025 tournament. It leverages modern machine learning methods and detailed point-by-point data.

## Project Overview

- **Data Sources:** Historical Grand Slam match data from Jeff Sackmann
- **Feature Engineering:** Extraction and calculation of features from point-by-point data
- **Modeling:** Use of classification models such as XGBoost to predict match results
- **Application:** Forecasting match outcomes for Wimbledon 2025

## Datasets

- `tennis_slam_pointbypoint`: [Jeff Sackmann GitHub](https://github.com/JeffSackmann/tennis_slam_pointbypoint)
- `tennis_atp`: [Match Stats & Rankings](https://github.com/JeffSackmann/tennis_atp)

## Steps

1. **Data Acquisition:** Download relevant datasets from the sources above.
2. **Data Preparation:** Clean and merge the data, perform feature engineering using point-by-point data.
3. **Model Training:** Train classification models (e.g., XGBoost) on historical data.
4. **Prediction:** Apply the model to current Wimbledon 2025 data for outcome forecasting.

## Learnings

After completing the project and running it, these points were discovered:


## Run the project

### Requirements

- Pandas, NumPy
- scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit

```bash
pip install -r requirements.txt
```

### Run
```
python src/train_model.py
streamlit run app/dashboard.py
```


## License

The datasets used are subject to the respective licenses of the original sources.