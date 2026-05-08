import pandas as pd # type: ignore
import streamlit as st # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score # type: ignore


# Load the CSV dataset from the specified file path.
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# Convert selected feature columns into numeric input data.
# This function uses one-hot encoding for categorical variables.
def prepare_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df[features], drop_first=True)


# Train a logistic regression model using training data.
def logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def main() -> None:
    # Configure the Streamlit page layout and title.
    st.set_page_config(page_title="Credit Risk Logistic Regression", layout="wide")
    st.title("Credit Risk Dataset Visualization and Logistic Regression")
    st.write(
        "Explore the cleaned credit risk dataset, select features, train a logistic regression model, and inspect predictions and metrics."
    )

    # Load the dataset once at app start.
    df = load_data("cleaned_credit_risk_dataset.csv")

    # Show a preview of the dataset and its shape.
    with st.expander("Dataset Preview"):
        st.dataframe(df.head(100))
        st.write("Shape:", df.shape)

    # Display summary statistics and target distribution counts.
    with st.expander("Dataset Summary"):
        st.write(df.describe(include="all"))
        st.subheader("Target value counts")
        if "loan_status" in df.columns:
            st.bar_chart(df["loan_status"].value_counts())

    # Sidebar inputs for model configuration.
    st.sidebar.header("Model configuration")
    target_column = st.sidebar.selectbox(
        "Choose target variable",
        ["loan_status", "cb_person_default_on_file"],
        index=0,
    )

    # Convert the default indicator column to numeric values for modeling.
    if target_column == "cb_person_default_on_file":
        df[target_column] = df[target_column].map({"Y": 1, "N": 0})

    # Prepare feature selection options excluding the target.
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

    # Require at least one feature before training.
    if len(selected_features) < 1:
        st.warning("Select at least one feature to train the model.")
        return

    # Sidebar controls for test split size and random seed.
    test_size = st.sidebar.slider("Test set proportion", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random seed", value=42, min_value=0, step=1)

    # Build the training feature matrix and target vector.
    X = prepare_features(df, selected_features)
    y = df[target_column].astype(int)

    # Show the processed feature matrix after categorical encoding.
    st.subheader("Feature dataset")
    st.write("Selected features after one-hot encoding:")
    st.write(X.head(10))

    # Split the data into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train the logistic regression model and score on the test set.
    model = logistic_regression(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Display model performance metrics.
    st.subheader("Model performance")
    st.write(f"**Target:** {target_column}")
    st.write(f"**Training samples:** {X_train.shape[0]}, **Test samples:** {X_test.shape[0]}")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")

    if y_proba is not None:
        st.write(f"**ROC AUC:** {roc_auc_score(y_test, y_proba):.4f}")

    # Show a classification report table for precision, recall, and F1-score.
    st.write("### Classification report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Display the confusion matrix for true vs predicted labels.
    st.write("### Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    st.dataframe(cm_df)

    # Show a scatter plot when at least two features are selected.
    if len(selected_features) >= 2:
        st.subheader("Feature visualization")
        x_axis = st.selectbox("X axis", selected_features, index=0)
        y_axis = st.selectbox("Y axis", selected_features, index=1 if len(selected_features) > 1 else 0)

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
            st.write("Column chart for numeric vs non-numeric features")
            if x_is_numeric and not y_is_numeric:
                agg_data = plot_data.groupby(y_axis)[x_axis].mean().reset_index(name=f"avg_{x_axis}")
                chart_spec = {
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": y_axis, "type": "nominal"},
                        "y": {"field": f"avg_{x_axis}", "type": "quantitative"},
                        "color": {"field": y_axis, "type": "nominal"},
                        "tooltip": [{"field": y_axis}, {"field": f"avg_{x_axis}"}],
                    },
                }
                st.vega_lite_chart(agg_data, chart_spec, use_container_width=True)
            elif y_is_numeric and not x_is_numeric:
                agg_data = plot_data.groupby(x_axis)[y_axis].mean().reset_index(name=f"avg_{y_axis}")
                chart_spec = {
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": x_axis, "type": "nominal"},
                        "y": {"field": f"avg_{y_axis}", "type": "quantitative"},
                        "color": {"field": x_axis, "type": "nominal"},
                        "tooltip": [{"field": x_axis}, {"field": f"avg_{y_axis}"}],
                    },
                }
                st.vega_lite_chart(agg_data, chart_spec, use_container_width=True)
            else:
                count_data = plot_data.groupby(x_axis).size().reset_index(name="count")
                chart_spec = {
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": x_axis, "type": "nominal"},
                        "y": {"field": "count", "type": "quantitative"},
                        "tooltip": [{"field": x_axis}, {"field": "count"}],
                    },
                }
                st.vega_lite_chart(count_data, chart_spec, use_container_width=True)

    # Add a small footer note in the sidebar.
    st.sidebar.markdown("---")
    st.sidebar.write("Run this app with: `streamlit run logisticregression.py`")


if __name__ == "__main__":
    main()


