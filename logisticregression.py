import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    roc_curve,
)

# Load the CSV dataset
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# One-hot encode categorical features
def prepare_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df[features], drop_first=True)

# Train logistic regression
def logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def main() -> None:
    st.set_page_config(page_title="Credit Risk Logistic Regression", layout="wide")
    st.title("Credit Risk Dataset Visualization and Logistic Regression")

    df = load_data("cleaned_credit_risk_dataset.csv")

    # Dataset preview
    with st.expander("Dataset Preview"):
        st.dataframe(df.head(100))
        st.write("Shape:", df.shape)

    # Dataset summary
    with st.expander("Dataset Summary"):
        st.write(df.describe(include="all"))
        st.subheader("Target value counts")
        if "loan_status" in df.columns:
            st.bar_chart(df["loan_status"].value_counts())

    # Sidebar config
    st.sidebar.header("Model configuration")
    target_column = st.sidebar.selectbox(
        "Choose target variable",
        ["loan_status", "cb_person_default_on_file"],
        index=0,
    )

    if target_column == "cb_person_default_on_file":
        df[target_column] = df[target_column].map({"Y": 1, "N": 0})

    available_features = [col for col in df.columns if col != target_column]
    default_features = [
        "person_age",
        "person_income",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    selected_features = st.sidebar.multiselect(
        "Select feature columns",
        available_features,
        default=[f for f in default_features if f in available_features],
    )

    if len(selected_features) < 1:
        st.warning("Select at least one feature to train the model.")
        return

    test_size = st.sidebar.slider("Test set proportion", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random seed", value=42, min_value=0, step=1)

    X = prepare_features(df, selected_features)
    y = df[target_column].astype(int)

    st.subheader("Feature dataset")
    st.write(X.head(10))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = logistic_regression(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Performance metrics
    st.subheader("Model performance")
    st.write(f"**Target:** {target_column}")
    st.write(f"**Training samples:** {X_train.shape[0]}, **Test samples:** {X_test.shape[0]}")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"**ROC AUC:** {roc_auc_score(y_test, y_proba):.4f}")

    # Classification report
    st.write("### Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion matrix
    st.write("### Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    st.dataframe(cm_df)

    # Confusion matrix heatmap
    st.write("### Confusion Matrix Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    st.pyplot(fig)

    # ROC curve
    st.write("### ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Coefficients + odds ratios
    st.write("### Logistic Regression Coefficients")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0],
        "Odds Ratio": np.exp(model.coef_[0])
    }).sort_values("Odds Ratio", ascending=False)
    st.dataframe(coef_df)

    # Prediction probabilities
    st.write("### Prediction Probabilities")
    proba_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred, "Probability": y_proba})
    st.dataframe(proba_df.head(20))
    fig, ax = plt.subplots()
    sns.histplot(proba_df["Probability"], bins=20, kde=True, ax=ax, color="green")
    ax.set_title("Distribution of Predicted Probabilities")
    st.pyplot(fig)

    # Feature visualization
    if len(selected_features) >= 2:
        st.subheader("Feature visualization")
        x_axis = st.selectbox("X axis", selected_features, index=0)
        y_axis = st.selectbox("Y axis", selected_features, index=1)

        x_is_numeric = pd.api.types.is_numeric_dtype(df[x_axis])
        y_is_numeric = pd.api.types.is_numeric_dtype(df[y_axis])

        plot_data = df[[x_axis, y_axis, target_column]].copy()
        if target_column == "cb_person_default_on_file":
            plot_data[target_column] = plot_data[target_column].map({1: "Default", 0: "No Default"})

        if x_is_numeric and y_is_numeric:
            st.write("Scatter plot for two numeric features")
            st.vega_lite_chart(
                plot_data,
                {
                    "mark": "circle",
                    "encoding": {
                        "x": {"field": x_axis, "type": "quantitative"},
                        "y": {"field": y_axis, "type": "quantitative"},
                        "color": {"field": target_column, "type": "nominal"},
                        "tooltip": [{"field": x_axis}, {"field": y_axis}, {"field": target_column}],
                    },
                },
                use_container_width=True,
            )
        else:
            st.write("Column chart for categorical vs numeric")
            if x_is_numeric and not y_is_numeric:
                agg_data = plot_data.groupby(y_axis)[x_axis].mean().reset_index(name=f"avg_{x_axis}")
                st.bar_chart(agg_data.set_index(y_axis))
            elif y_is_numeric and not x_is_numeric:
                agg_data = plot_data.groupby(x_axis)[y_axis].mean().reset_index(name=f"avg_{y_axis}")
                st.bar_chart(agg_data.set_index(x_axis))
            else:
                count_data = plot_data.groupby(x_axis).size().reset_index(name="count")
                st.bar_chart(count_data.set_index(x_axis))

    st.sidebar.markdown("---")
    st.sidebar.write("Run this app with: `streamlit run logisticregression.py`")

if __name__ == "__main__":
    main()
