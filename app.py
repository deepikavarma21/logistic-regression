import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


st.set_page_config(page_title="Logistic Regression - Loan Status", layout="wide")

st.title("Logistic Regression (Train data → Predict test data)")

DATA_DIR = "."

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def identify_columns(df: pd.DataFrame):
    # Expected from the provided datasets (Loan Prediction)
    target_candidates = ["Loan_Status", "loan_status", "target", "Target", "y", "Y"]
    target = None
    for c in target_candidates:
        if c in df.columns:
            target = c
            break
    if target is None:
        # Fallback: try last column
        target = df.columns[-1]

    drop_cols = []
    if "Loan_ID" in df.columns:
        drop_cols.append("Loan_ID")

    feature_cols = [c for c in df.columns if c not in ([target] + drop_cols)]
    return target, drop_cols, feature_cols


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def train_model(df_train: pd.DataFrame, test_size: float, random_state: int, reg_strength: float, max_iter: int):
    target, drop_cols, feature_cols = identify_columns(df_train)

    X = df_train[feature_cols].copy()
    y = df_train[target].copy()

    # Normalize target to binary 0/1 if possible
    if y.dtype == object:
        y_unique = sorted(y.dropna().unique().tolist())
        # Common mapping: Y/N
        if set(y_unique).issubset({"Y", "N"}):
            y = y.map({"N": 0, "Y": 1}).astype(int)
        else:
            # Label encode
            y = pd.Categorical(y).codes

    preprocessor = build_preprocessor(X)

    clf = LogisticRegression(
        C=reg_strength,
        penalty="l2",
        solver="lbfgs",
        max_iter=max_iter,
        class_weight="balanced",
    )

    model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "roc_auc": float(roc_auc_score(y_val, y_proba)) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_val, y_pred),
        "classification_report": classification_report(y_val, y_pred, digits=4),
    }

    return model, metrics, target, drop_cols, feature_cols


def predict_and_show(model, df_test: pd.DataFrame, feature_cols, target_name_for_display: str):
    X_test = df_test[feature_cols].copy()

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Try inverse mapping to Y/N if it looks like that
    # (We can't guarantee, but gives nicer display.)
    display_pred = pred
    if set(np.unique(pred)).issubset({0, 1}):
        display_pred = pd.Series(pred).map({0: "N", 1: "Y"}).values

    out = df_test.copy()
    out["prediction"] = display_pred
    out["prediction_proba"] = proba

    st.subheader("Predictions on test data")
    st.dataframe(out.head(50), use_container_width=True)

    st.subheader("Prediction distribution")
    vals, counts = np.unique(display_pred, return_counts=True)
    st.bar_chart(pd.DataFrame({"count": counts}, index=vals))

    return out


with st.sidebar:
    st.header("Inputs")

    default_train = "train_u6lujuX_CVtuZ9i.csv"
    default_test = "test_Y3wMUE5_7gLdaTN.csv"

    train_path = st.text_input("Train CSV path", value=default_train)
    test_path = st.text_input("Test CSV path", value=default_test)

    st.markdown("---")
    test_size = st.slider("Validation split (test_size)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    st.markdown("---")
    reg_strength = st.number_input("Logistic Regression: C (inverse reg)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    max_iter = st.number_input("Max iterations", min_value=50, max_value=2000, value=500, step=50)

    st.markdown("---")
    run = st.button("Train & Predict", type="primary")


if run:
    try:
        df_train = load_csv(train_path)
        df_test = load_csv(test_path)

        st.write("Training shape:", df_train.shape)
        st.write("Test shape:", df_test.shape)

        with st.spinner("Training logistic regression model..."):
            model, metrics, target, drop_cols, feature_cols = train_model(
                df_train,
                test_size=float(test_size),
                random_state=int(random_state),
                reg_strength=float(reg_strength),
                max_iter=int(max_iter),
            )

        st.success("Training completed")

        st.subheader("Validation metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", metrics["accuracy"])
        c2.metric("ROC AUC", metrics["roc_auc"] if metrics["roc_auc"] is not None else "N/A")
        c3.metric("Confusion Matrix (size)", f"{metrics['confusion_matrix'].shape}")

        st.write("Confusion Matrix")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"],
                                  index=["Actual 0", "Actual 1"],
                                  columns=["Pred 0", "Pred 1"]),
                     use_container_width=True)

        st.write("Classification Report")
        st.code(metrics["classification_report"])

        predict_and_show(model, df_test, feature_cols, target)

        st.subheader("Model details")
        st.write("Target column:", target)
        st.write("Dropped columns:", drop_cols)
        st.write("Number of feature columns used:", len(feature_cols))

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Configure inputs in the sidebar and click **Train & Predict**.")

