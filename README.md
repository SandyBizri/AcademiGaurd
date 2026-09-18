# AcademiGuard — Early Student Risk Detection System

AcademiGuard is an early-warning system that identifies students at risk of dropping out based on grades, study habits, absences, and other factors.  
It classifies students into **Low Risk** or **At Risk** and provides a probability score.

## Features

- Load and preprocess student performance data
- Calculate risk scores
- Train a Random Forest model to predict student risk
- Predict risk for new students with probability
- Visualize risk distributions and correlations

## Project Structure

- `data/raw/` → Original CSV files (student performance)
- `data/processed/` → Preprocessed CSV with risk scores
- `models/` → Trained model (`risk_model.pkl`)
- `src/` → Python scripts (`data_loader.py`, `preprocessing.py`, `risk_calculator.py`, `visualization.py`, `train_model.py`, `predict.py`)
- `notebooks/` → Week 1 & Week 2 notebooks
- `main.py` → Run training and prediction
- `requirements.txt` → Python dependencies

## Installation

```bash
pip install -r requirements.txt

## Run the main program
python main.py