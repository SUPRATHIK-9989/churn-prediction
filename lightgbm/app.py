import streamlit as st
import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load Model, Scaler, Threshold
# -----------------------------
model = joblib.load(r"lgbm_model.pkl")
scaler = joblib.load(r"scaler.pkl")
best_threshold = joblib.load(r"best_threshold.pkl")

# -----------------------------
# Feature Names & Display Mapping
# -----------------------------
feature_names = [
    "feature_0", "feature_1", "feature_2", "feature_3",
    "feature_4", "feature_5", "feature_6", "feature_7",
    "feature_8", "feature_9", "feature_10", "feature_11",
    "feature_12", "feature_13", "feature_14", "feature_15"
]

feature_display = {
    "feature_0": "Policy Term (years)",
    "feature_1": "Customer Age",
    "feature_2": "Annual Premium",
    "feature_3": "Past Complaints",
    "feature_4": "Payment Method",
    "feature_5": "Tenure with Company (years)",
    "feature_6": "No. of Dependents",
    "feature_7": "Region Code",
    "feature_8": "Vehicle Age (years)",
    "feature_9": "Vehicle Damage (0=No,1=Yes)",
    "feature_10": "Gender (0=Male,1=Female)",
    "feature_11": "Marital Status (0=Single,1=Married)",
    "feature_12": "Occupation",
    "feature_13": "Channel",
    "feature_14": "Claims Made",
    "feature_15": "Policy Type"
}

# -----------------------------
# Helper Functions
# -----------------------------
def predict_single(input_dict, threshold):
    df = pd.DataFrame([input_dict])[feature_names]
    X_scaled = scaler.transform(df)
    proba = model.predict_proba(X_scaled)[:, 1][0]
    pred = int(proba >= threshold)
    return pred, proba

def explain_prediction(input_dict):
    df = pd.DataFrame([input_dict])[feature_names]
    X_scaled = scaler.transform(df)

    # SHAP explainer for tree-based model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    return explainer, shap_values, df

def predict_batch(data, threshold):
    df = data[feature_names]
    X_scaled = scaler.transform(df)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)
    data["Churn_Probability"] = probs
    data["Prediction"] = preds
    return data

def load_train_data():
    return pd.read_csv(r"E:\lightgbm\data\Train.csv")

# -----------------------------
# Streamlit Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Single Customer Prediction",
    "Batch Prediction",
    "Model Performance",
    "Exploratory Data Analysis",
    "About"
])

# -----------------------------
# Single Prediction Page
# -----------------------------
if page == "Single Customer Prediction":
    st.title("Single Customer Churn Prediction")

    input_data = {}
    defaults = {
        "feature_0": 10, "feature_1": 40, "feature_2": 40000, "feature_3": 0,
        "feature_4": 1, "feature_5": 5, "feature_6": 2, "feature_7": 1,
        "feature_8": 3, "feature_9": 0, "feature_10": 0, "feature_11": 0,
        "feature_12": 1, "feature_13": 1, "feature_14": 0, "feature_15": 1
    }

    for feat in feature_names:
        val = st.number_input(feature_display[feat], value=float(defaults[feat]))
        input_data[feat] = val

    # Threshold selector
    threshold_mode = st.radio(
        "Select Threshold Mode:",
        ("Best Threshold (optimized)", "Default Threshold (0.5)")
    )
    threshold = best_threshold if "Best" in threshold_mode else 0.5

    if st.button("Predict"):
        pred, proba = predict_single(input_data, threshold)
        st.success(f"Prediction: {'Churn' if pred==1 else 'No Churn'}")
        st.info(f"Churn Probability: {proba:.2f} (Threshold = {threshold:.2f})")

        # Insights
        st.markdown("###Key Feature Insights")
        st.write("""
        - **Policy Term**: Longer terms → lower churn risk.  
        - **Customer Age**: Older customers churn less.  
        - **Tenure with Company**: Higher tenure reduces churn.  
        - **Annual Premium**: Extremely high premiums increase churn risk.  
        - **Past Complaints**: More complaints = higher churn.  
        """)

if st.button("Explain Prediction"):
    explainer, shap_values, df = explain_prediction(input_data)
    st.subheader("🔍 Feature Contributions to Prediction")

    # Handle binary classification SHAP output
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]  # use class 1 (churn)
        base_val = explainer.expected_value[1]
    else:
        shap_vals = shap_values[0]
        base_val = explainer.expected_value

    # Waterfall plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals,
            base_values=base_val,
            data=df.iloc[0].values,
            feature_names=[feature_display[f] for f in feature_names]
        )
    )
    st.pyplot(fig)


# -----------------------------
# Batch Prediction Page
# -----------------------------
elif page == "Batch Prediction":
    st.title("Batch Churn Prediction")
    uploaded = st.file_uploader("Upload CSV file (must have feature_0 … feature_15)", type=["csv"])

    threshold_mode = st.radio(
        "Select Threshold Mode:",
        ("Best Threshold (optimized)", "Default Threshold (0.5)")
    )
    threshold = best_threshold if "Best" in threshold_mode else 0.5

    if uploaded:
        data = pd.read_csv(uploaded)
        st.write("File Uploaded. Preview:")
        st.dataframe(data.head())

        if st.button("Run Batch Prediction"):
            results = predict_batch(data, threshold)
            st.dataframe(results.head())
            st.download_button("Download Predictions", results.to_csv(index=False),
                               "predictions.csv", "text/csv")

# -----------------------------
# Model Performance Page
# -----------------------------
elif page == "Model Performance":
    st.title("Model Performance")
    st.write("""
            Accuracy: 0.89   
            Precision: 0.76  
            Recall: 0.76  
            F1-Score: 0.62   
            AUC: 0.92
    """)

    plots = {
        r"E:\lightgbm\confusion_matrix.png": "Confusion Matrix",
        r"E:\lightgbm\feature_importance.png": "Feature Importance",
    }

    for fname, caption in plots.items():
        fpath = os.path.join("data", fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=caption)
        else:
            st.warning(f"{caption} not found ({fname})")

# -----------------------------
# EDA Page
# -----------------------------
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    try:
        train = load_train_data()
        churn_count = train["labels"].sum()
        churn_rate = churn_count / len(train) * 100

        st.metric("Total Customers", len(train))
        st.metric("Churned Customers", churn_count)
        st.metric("Churn Rate (%)", f"{churn_rate:.2f}")
    except Exception as e:
        st.error(f"Error loading Train.csv: {e}")

    eda_plots = [
        (r"E:\lightgbm\1_class_distribution.png", "Class Distribution"),
        (r"E:\lightgbm\2_feature_distributions.png", "Feature Distributions"),
        (r"E:\lightgbm\4_boxplots_by_label.png", "Boxplots by Label"),
    ]

    for fname, caption in eda_plots:
        fpath = os.path.join("data", fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=caption)
        else:
            st.warning(f"{caption} not found ({fname})")

# -----------------------------
# About Page
# -----------------------------
elif page == "About":
    st.title("About This Project")
    st.write("""
    This project builds a **customer churn prediction system** using **LightGBM**.  
    - Imbalance handled with **SMOTE**  
    - Features scaled before training  
    - Hyperparameter tuning & threshold optimization applied  
    - Exploratory Data Analysis performed  
    - Streamlit app supports single & batch predictions  

    Key insights:
    - **Short policy terms, younger customers, and frequent complaints drive churn.**  
    - **Longer tenure and older age reduce churn probability.**  
    - **High premiums may increase churn unless offset by loyalty factors.**  
    """)
