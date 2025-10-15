import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Bankruptcy Prediction Showcase",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL AND IMAGE LOADING ---
# Use caching to load models and images only once
@st.cache_resource
def load_resources():
    """Loads all models and images into memory."""
    try:
        scaler = joblib.load("scaler_all.pkl")
        xgb_model = joblib.load("xgb_all.joblib")
        ann_model = load_model("ann_all.keras")
        meta_model = joblib.load("meta_logreg_all.joblib")
    except FileNotFoundError as e:
        st.error(f"A model file was not found. Please make sure all model files are in the same directory. Missing file: {e.filename}")
        st.stop()


    images = {
        "confusion_matrix": Image.open("Figure_1.png"),
        "roc_curves": Image.open("Figure_2.png"),
        "feature_importance": Image.open("Figure_3.png"),
        "class_distribution": Image.open("Figure_4.png")
    }
    return scaler, xgb_model, ann_model, meta_model, images

scaler, xgb_model, ann_model, meta_model, images = load_resources()

# --- SIDEBAR ---
st.sidebar.title("📋 Project Overview")
st.sidebar.info(
    """
    This application showcases a machine learning project for **predicting corporate bankruptcy.**

    It walks through the challenges of an imbalanced dataset, the hybrid modeling approach used, and the final performance of the predictive models.

    You can also use the **Interactive Predictor** to get a live prediction for a new company based on its financial attributes.
    """
)
st.sidebar.title("👨‍💻 About the Model")
st.sidebar.success(
    """
    The final model is a **stacked ensemble**, which combines the predictions from two powerful base models:
    - **XGBoost:** A gradient boosting algorithm known for its performance.
    - **Artificial Neural Network (ANN):** A deep learning model.

    A Logistic Regression **meta-model** learns from their outputs to make a more robust final prediction.
    """
)

# --- MAIN PAGE ---
st.title("⚖️ Corporate Bankruptcy Prediction")
st.write("An interactive showcase of a stacked ensemble model for predicting financial distress.")
st.markdown("---")


# --- THE CHALLENGE ---
st.header("1. The Challenge: A Highly Imbalanced Dataset")
st.write(
    "The first major hurdle in this dataset is the severe class imbalance. As the chart below shows, the number of non-bankrupt companies (Class 0) vastly outweighs the bankrupt ones (Class 1)."
)

col1, col2 = st.columns([1, 1.5])

with col1:
    st.image(images["class_distribution"], caption="Vast majority of companies are not bankrupt (Class 0).", use_column_width=True)

with col2:
    st.warning(
        """
        **Why is this a problem?**

        A naive model could achieve over 95% accuracy by simply guessing "Not Bankrupt" every single time. This creates a misleading sense of performance.

        Our goal is not just high accuracy, but to correctly identify the companies that **are** at risk, which is the minority class.
        """
    )
    st.info(
        """
        **Our Solution:**

        To combat this, we employed specific techniques during training:
        1.  **`scale_pos_weight` (for XGBoost):** This technique increases the importance of the minority class, forcing the model to pay more attention to it.
        2.  **`class_weight='balanced'` (for ANN):** Similarly, this method adjusts the loss function to penalize misclassifications of the minority class more heavily.
        """
    )

st.markdown("---")

# --- MODEL PERFORMANCE ---
st.header("2. Evaluating Our Hybrid Model's Performance")
st.write("We evaluated the performance of the base models and the final stacked meta-model. The meta-model effectively combines the strengths of both, leading to robust results.")

# Display metrics in a styled way
st.subheader("📊 Key Performance Metrics")
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'XGBoost': [0.9863, 0.7759, 0.5233, 0.6250, 0.9657],
    'ANN': [0.8637, 0.0993, 0.6512, 0.1723, 0.8666],
    'Stacked Meta-Model': [0.9871, 0.8571, 0.4884, 0.6222, 0.9397]
}
metrics_df = pd.DataFrame(metrics_data).set_index('Metric')
st.dataframe(metrics_df.style.highlight_max(axis=1, color='lightgreen').format("{:.4f}"))
st.caption("The Stacked Meta-Model achieves the best balance, particularly in Precision.")

# Display visual results
col1, col2 = st.columns(2)
with col1:
    st.subheader("Confusion Matrix (Meta-Model)")
    st.image(images["confusion_matrix"], caption="Results on the test set.", use_column_width=True)
    st.markdown(
        """
        - **3855 True Negatives:** Correctly predicted 'Not Bankrupt'.
        - **42 True Positives:** Correctly predicted 'Bankrupt'.
        - **44 False Negatives (Type II Error):** Incorrectly predicted 'Not Bankrupt' when it was bankrupt. **This is the most critical error to minimize.**
        - **7 False Positives (Type I Error):** Incorrectly predicted 'Bankrupt'.
        """
    )

with col2:
    st.subheader("ROC Curves")
    st.image(images["roc_curves"], caption="Model's ability to distinguish classes.", use_column_width=True)
    st.markdown(
        """
        The **Receiver Operating Characteristic (ROC)** curve shows the trade-off between the true positive rate and false positive rate.

        - A curve closer to the top-left corner indicates better performance.
        - All models perform significantly better than random chance (the dashed line).
        - The XGBoost model shows a slightly superior ROC-AUC score, indicating excellent class separation ability.
        """
    )

st.markdown("---")

# --- FEATURE IMPORTANCE ---
st.header("3. What Financial Attributes Drive the Predictions?")
st.write("The XGBoost model allows us to peek inside the 'black box' and see which features were most influential in making predictions.")
st.image(images["feature_importance"], caption="Top 20 most important features according to the XGBoost model.", use_column_width=True)
st.success(
    """
    **Key Insight:** The model's decisions are heavily influenced by attributes like **`Attr26`** (Retained Earnings / Total Assets) and **`Attr34`** (Operating Expenses / Total Liabilities). This aligns with financial intuition that profitability and debt management are critical indicators of a company's health.
    """
)

st.markdown("---")

# --- INTERACTIVE PREDICTOR ---
st.header("🔮 Interactive Predictor")
st.write("Enter the 64 financial attributes for a company below to get a real-time bankruptcy prediction from the stacked model.")

with st.form("prediction_form"):
    input_text = st.text_area(
        "Paste 64 comma-separated numerical values:",
        height=150,
        placeholder="Example: 0.2, -0.5, 0.1, ..., 1.5"
    )
    submitted = st.form_submit_button("Get Prediction")

if submitted:
    if not input_text:
        st.warning("Please enter the 64 attribute values.")
    else:
        try:
            # 1. Parse and validate the input
            values = [float(v.strip()) for v in input_text.split(',')]
            if len(values) != 64:
                st.error(f"Input Error: Expected 64 values, but received {len(values)}. Please check your input.")
            else:
                with st.spinner('Analyzing financial data...'):
                    # 2. Scale the features
                    input_array = np.array(values).reshape(1, -1)
                    scaled_features = scaler.transform(input_array)

                    # 3. Get predictions from base models
                    xgb_pred_prob = xgb_model.predict_proba(scaled_features)[:, 1]
                    ann_pred_prob = ann_model.predict(scaled_features, verbose=0).flatten()

                    # 4. Create input for the meta-model
                    meta_input = np.column_stack([xgb_pred_prob, ann_pred_prob])

                    # 5. Get final prediction from the meta-model
                    final_prob = meta_model.predict_proba(meta_input)[:, 1][0]
                    final_prediction = (final_prob > 0.5).astype(int)

                # 6. Display the result
                st.subheader("Prediction Result")
                if final_prediction == 1:
                    st.error(f"**Prediction: Bankrupt** (Confidence: {final_prob:.2%})")
                    st.write("The model indicates a high risk of financial distress for this company based on the provided attributes.")
                else:
                    st.success(f"**Prediction: Not Bankrupt** (Confidence: {1-final_prob:.2%})")
                    st.write("The model indicates a low risk of financial distress for this company.")

                # Show breakdown of model probabilities
                with st.expander("See individual model probabilities"):
                    st.write(f"- **XGBoost Probability of Bankruptcy:** `{xgb_pred_prob[0]:.4f}`")
                    st.write(f"- **ANN Probability of Bankruptcy:** `{ann_pred_prob[0]:.4f}`")
                    st.write(f"- **Final Meta-Model Probability:** `{final_prob:.4f}`")

        except ValueError:
            st.error("Input Error: Please make sure all values are numeric and separated by commas.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
