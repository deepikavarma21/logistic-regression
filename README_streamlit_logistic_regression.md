# Streamlit Logistic Regression App

## Files

- `app_streamlit_logistic_regression.py` — Streamlit app
- `train_u6lujuX_CVtuZ9i.csv` — training dataset (includes `Loan_Status`)
- `test_Y3wMUE5_7gLdaTN.csv` — test dataset

## Run

From the `datasets` folder:

```bash
pip install streamlit pandas numpy scikit-learn
streamlit run app_streamlit_logistic_regression.py
```

## What it does

- Trains a Logistic Regression model (with preprocessing for numeric + categorical features)
- Splits training data into train/validation and shows metrics
- Predicts on the test CSV and displays the first 50 predictions
