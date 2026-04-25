from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


DROP_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET_COLUMN = "Machine failure"
TYPE_COLUMN = "Type"
NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_and_preprocess(file):
    file_bytes = file.getvalue() if hasattr(file, "getvalue") else file
    data = pd.read_csv(BytesIO(file_bytes)).copy()
    data = data.drop(columns=[column for column in DROP_COLUMNS if column in data.columns])

    encoder = LabelEncoder()
    data[TYPE_COLUMN] = encoder.fit_transform(data[TYPE_COLUMN])

    return data, encoder


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[NUMERIC_COLUMNS] = scaler.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])

    return X_train, X_test, scaler


def train_models(X_train, y_train):
    X_train_values = X_train.to_numpy()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        ),
        "SVM": SVC(kernel="linear", probability=True, random_state=42),
    }

    for model in models.values():
        model.fit(X_train_values, y_train)

    return models


def evaluate_model(model, X_test, y_test):
    X_test_values = X_test.to_numpy()
    y_pred = model.predict(X_test_values)
    y_pred_proba = model.predict_proba(X_test_values)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "fpr": fpr,
        "tpr": tpr,
    }


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")

    if uploaded_file is None:
        st.info("Загрузите файл predictive_maintenance.csv из папки data.")
        return

    data, encoder = load_and_preprocess(uploaded_file)
    missing_values = data.isnull().sum()

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train, X_test, scaler = scale_data(X_train, X_test)
    models = train_models(X_train, y_train)
    results = {
        model_name: evaluate_model(model, X_test, y_test)
        for model_name, model in models.items()
    }

    st.header("Результаты обучения модели")

    missing_df = pd.DataFrame(
        {
            "Признак": missing_values.index,
            "Пропущенные значения": missing_values.values,
        }
    )
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    comparison_df = pd.DataFrame(
        [
            {
                "Модель": model_name,
                "Accuracy": round(metrics["accuracy"], 4),
                "ROC-AUC": round(metrics["roc_auc"], 4),
            }
            for model_name, metrics in results.items()
        ]
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.subheader("ROC-кривые")
    roc_figure, roc_axis = plt.subplots()
    for model_name, metrics in results.items():
        roc_axis.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"{model_name} (AUC = {metrics['roc_auc']:.2f})",
        )
    roc_axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    roc_axis.set_xlabel("False Positive Rate")
    roc_axis.set_ylabel("True Positive Rate")
    roc_axis.legend()
    st.pyplot(roc_figure)
    plt.close(roc_figure)

    selected_model_name = st.selectbox("Выберите модель", list(results.keys()))
    selected_result = results[selected_model_name]

    st.write(f"Accuracy: {selected_result['accuracy']:.2f}")
    st.write(f"ROC-AUC: {selected_result['roc_auc']:.2f}")

    st.subheader("Confusion Matrix")
    confusion_figure, confusion_axis = plt.subplots()
    sns.heatmap(
        selected_result["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=confusion_axis,
    )
    st.pyplot(confusion_figure)
    plt.close(confusion_figure)

    st.subheader("Classification Report")
    st.text(selected_result["classification_report"])

    st.header("Предсказание по новым данным")
    with st.form("prediction_form"):
        st.write("Введите значения признаков для предсказания:")
        product_type = st.selectbox("Type", encoder.classes_.tolist())
        air_temp = st.number_input("Air temperature [K]", value=300.0)
        process_temp = st.number_input("Process temperature [K]", value=310.0)
        rotational_speed = st.number_input("Rotational speed [rpm]", value=1500.0)
        torque = st.number_input("Torque [Nm]", value=40.0)
        tool_wear = st.number_input("Tool wear [min]", value=0.0)
        submit_button = st.form_submit_button("Предсказать")

    if submit_button:
        input_data = pd.DataFrame(
            {
                TYPE_COLUMN: encoder.transform([product_type]),
                "Air temperature [K]": [air_temp],
                "Process temperature [K]": [process_temp],
                "Rotational speed [rpm]": [rotational_speed],
                "Torque [Nm]": [torque],
                "Tool wear [min]": [tool_wear],
            }
        )

        input_data[NUMERIC_COLUMNS] = scaler.transform(input_data[NUMERIC_COLUMNS])
        input_data = input_data[X.columns]

        prediction = models[selected_model_name].predict(input_data.to_numpy())
        prediction_proba = models[selected_model_name].predict_proba(input_data.to_numpy())[:, 1]

        st.write(f"Предсказание: {int(prediction[0])}")
        st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")


analysis_and_model_page()
