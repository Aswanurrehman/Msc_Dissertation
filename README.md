# Msc_Dissertation
This repository contains an MSc research project on two solar power plants in India over a 34-day period, combining machine learning models such as Random Forest, XGBoost and deep learning model Long Short-Term Memory (LSTM) to evaluate their performance in real-world solar forecasting scenarios

# Project Overview
The primary goal is to forecast the 'GHI' (a measure of solar radiation) using a sequence of past weather conditions. This is a time-series regression task. The project leverages an LSTM network to capture temporal dependencies in the data.

The workflow consists of:

Data Loading and Preprocessing: Ingesting and cleaning the solar_weather.csv dataset.
Feature Engineering: Creating new features from existing data to improve model performance (e.g., cyclical time features).
Data Transformation: Applying a Yeo-Johnson power transform to the target variable to stabilize variance and handling non-normality. Features are scaled to a [0, 1] range.
Hyperparameter Tuning: Using Optuna to systematically find the optimal model architecture and training parameters.
Model Training: Training the final LSTM model on a dedicated GPU (mps or cuda) with a cosine annealing learning rate scheduler and early stopping.
Model Evaluation: Assessing the model's performance on a held-out test set using metrics like R², RMSE, and MAE.
Uncertainty Estimation: Using Monte Carlo (MC) Dropout during inference to quantify the model's prediction uncertainty.

# Methodology
## Data Preparation
Data preparation is handled by the data_prep.py module, which performs the following steps:

Renaming & Sorting: Columns are renamed for clarity, and the data is sorted by timestamp.
Feature Engineering: SolarElevation is engineered as a proxy for the sun's position.
Scaling:
Features are scaled using MinMaxScaler.
The transformed target is scaled using StandardScaler.
Sequencing: The data is converted into sequences with a window size of 24 time steps.
Data Splitting: The dataset is split into test_size=0.2

#  Hyperparameter Optimization

1. RandomForestRegressor
'n_estimators': [100, 200],
'max_depth': [10, 20, None],
'min_samples_split': [2, 5]

param_grid,
cv=3,
scoring='neg_mean_squared_error',
n_jobs=-1

2. XGBRegressor
'n_estimators': [100, 200],
'max_depth': [3, 6],
'learning_rate': [0.01, 0.1]

cv=3,
scoring='neg_mean_squared_error',
n_jobs=-1

3. LSTM Hyperparameter Tuning
LSTM(128, return_sequences=True, input_shape=(seq_length, X.shape[1])),
Dropout(0.4),
LSTM(64),
Dense(1)

optimizer='adam', loss='mse')

epochs=20,
batch_size=32,
validation_split=0.2)

# Model Performance Comparison

Model	MAE	        RMSE	  R² Score
RF    	0.045   	0.078	  0.91
XGBoost	0.032   	0.061	  0.94
LSTM	0.089   	0.124	  0.78

The results clearly indicate that XGBoost is the best-performing model. It achieves the lowest MAE (0.032) and RMSE (0.061), along with the highest R² score (0.94), indicating a strong fit to the data. Random Forest also performs well, achieving an R² score of 0.91, but with slightly higher error values. In contrast, the LSTM model performs significantly worse, with an MAE of 0.089 and RMSE of 0.124, and a lower R² score of 0.78, indicating weaker predictive capability