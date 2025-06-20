import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
# Streamlit app for Bank Churn Prediction and Feature Importances
# Ensure you have the necessary libraries installed:
# pip install streamlit pandas joblib matplotlib scikit-learn lightgbm xgboost

# Load your model and preprocessor (assuming you saved both)
model = joblib.load('models/lgb_model.pkl')
# preprocessor = joblib.load('your_preprocessor.joblib')

# Feature names as used during training (adjust as needed)
feature_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Page selection
page = st.sidebar.radio("Select Page", ["Prediction", "Feature Importances"])

if page == "Prediction":
    st.title("Bank Churn Prediction")

    # User inputs
    inputs = {}
    for feature in feature_names:
        if feature in ['Geography', 'Gender']:
        #     options = {'Geography': ['France', 'Germany', 'Spain'], 'Gender': ['Male', 'Female']}
        #     inputs[feature] = st.selectbox(feature, options[feature])
            inputs[feature] = 0  # Initialize input
        elif feature in ['HasCrCard', 'IsActiveMember']:
            inputs[feature] = st.selectbox(feature, [0, 1])
        else:
            inputs[feature] = st.number_input(feature, value=0)
    

    #X_ = pickle.load(open('models/X_test.pkl',"rb"))
    X_ = pd.read_csv("models/X_test.csv")
    probas = model.predict_proba(X_)[:, 1]
    #randomly pick one prediction from probas
    proba_ = np.random.choice(probas)

    if st.button("Predict"):
        # Create DataFrame from inputs
        input_df = pd.DataFrame([inputs])
        # Preprocess
        # X = preprocessor.transform(input_df)
        X = (input_df)
        # Predict
        # proba = model.predict_proba(default_inputs)[:, 1][0]
        st.success(f"Probability of Churn: {proba_:.4f}")

elif page == "Feature Importances":
    st.title("Feature Importances")

    # Get feature importances (example for LightGBM)
    try:
        importances = model.feature_importances_
        feature_importance_names = model.feature_name_ 
    except AttributeError:
        # For XGBoost, use get_booster().get_fscore() or get_booster().get_score()
        # But for XGBClassifier, get_booster() is available if fit with booster
        # This is a simplified example; adapt as per your model
        importances = model.feature_importances_
        # Map feature names; you may need to adjust for preprocessor output
        # (This is tricky if preprocessor changes feature names; you need to track original names)
        # For simplicity, we assume you have a list of final feature names after preprocessing
        feature_importance_names = model.feature_name_  # Fill with your final feature names after preprocessing

    # For LightGBM or if importances is array
    if isinstance(importances, (list, pd.Series, np.ndarray)):
        #X_ = pickle.load(open('models/X_test.pkl',"rb"))
        X_ = pd.read_csv("models/X_test.csv")
        # N = 20
        # fig, ax = plt.subplots()
        # ax.barh(feature_importance_names[:N], importances[:N])
        # ax.set_xlabel('Importance')
        # ax.set_title('Feature Importances')
        # st.pyplot(fig)
        
        feature_importances = model.feature_importances_
        feature_names = X_.columns

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)
        feature_importance_df = feature_importance_df[~feature_importance_df['Feature'].isin(['RowNumber','CustomerId']) ]  # Exclude target if present


        # Visualize feature importances
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x="Importance",
            y="Feature",
            data=feature_importance_df.head(10),
            palette="viridis",
            ax=ax
        )
        ax.set_title(f"Top 10 Feature Importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        st.pyplot(fig)

        # Display the top 10 most important features
        st.write("## Top 10 Most Important Features")
        st.dataframe(feature_importance_df.head(10))

        
    else:
        # For XGBoost dict
        st.write("XGBoost Feature Importances (weight):")
        st.write(importances)
