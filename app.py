import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


st.set_page_config(
    page_title="Telecom Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("📊 Telecom Customer Churn Dashboard")
st.markdown("""
Welcome! Explore customer churn, visualize key trends, and see ML model predictions.  
Use the sidebar filters to interactively explore the dataset.
""")

# --- Load dataset ---
df = pd.read_csv('telecom_churn_dataset.csv')


st.sidebar.header("Filters")
selected_contract = st.sidebar.multiselect(
    "Contract Type",
    options=df['Contract'].unique(),
    default=df['Contract'].unique()
)
selected_payment = st.sidebar.multiselect(
    "Payment Method",
    options=df['PaymentMethod'].unique(),
    default=df['PaymentMethod'].unique()
)
selected_gender = st.sidebar.multiselect(
    "Gender",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

tenure_range = st.sidebar.slider(
    "Tenure (Months)",
    min_value=int(df['Tenure'].min()),
    max_value=int(df['Tenure'].max()),
    value=(int(df['Tenure'].min()), int(df['Tenure'].max()))
)

monthly_charges_range = st.sidebar.slider(
    "Monthly Charges",
    min_value=int(df['MonthlyCharges'].min()),
    max_value=int(df['MonthlyCharges'].max()),
    value=(int(df['MonthlyCharges'].min()), int(df['MonthlyCharges'].max()))
)


filtered_df = df[
    (df['Contract'].isin(selected_contract)) &
    (df['PaymentMethod'].isin(selected_payment)) &
    (df['Gender'].isin(selected_gender)) &
    (df['Tenure'].between(*tenure_range)) &
    (df['MonthlyCharges'].between(*monthly_charges_range))
]

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "EDA Charts", "Model Metrics", "Feature Importance"])

# --- TAB 1: Data Preview ---
with tab1:
    st.subheader("Filtered Data")
    st.dataframe(filtered_df)

    st.markdown("### Quick Statistics")
    st.write(filtered_df.describe())


with tab2:
    st.subheader("Churn Distribution")
    if st.checkbox("Show Churn Distribution"):
        fig, ax = plt.subplots()
        sns.countplot(x='Churn', data=filtered_df, palette='coolwarm', ax=ax)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Show Monthly Charges vs Churn"):
            fig2, ax2 = plt.subplots()
            sns.boxplot(x='Churn', y='MonthlyCharges', data=filtered_df, palette='Set2', ax=ax2)
            st.pyplot(fig2)
    with col2:
        if st.checkbox("Show Tenure vs Churn"):
            fig3, ax3 = plt.subplots()
            sns.boxplot(x='Churn', y='Tenure', data=filtered_df, palette='Set3', ax=ax3)
            st.pyplot(fig3)


with tab3:
    st.subheader("Prepare Data for ML")
    # Encode binary columns
    binary_cols = ['Gender','Partner','Dependents','PhoneService','PaperlessBilling',
                   'MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies']
    for col in binary_cols:
        filtered_df[col] = filtered_df[col].map({'Yes':1,'No':0,'Male':1,'Female':0})

   
    cat_cols = ['InternetService','Contract','PaymentMethod']
    le = LabelEncoder()
    for col in cat_cols:
        filtered_df[col] = le.fit_transform(filtered_df[col])

  
    X = filtered_df.drop(['CustomerID','Churn'], axis=1)
    y = filtered_df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:,1]

    st.write("**Random Forest Accuracy:**", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))


    if st.checkbox("Show Confusion Matrix"):
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        st.pyplot(fig_cm)


    if st.checkbox("Show ROC Curve"):
        st.subheader("ROC Curve")
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label='Random Forest (AUC = %.2f)' % roc_auc_score(y_test, y_proba))
        ax_roc.plot([0,1],[0,1],'--', color='gray')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)


    st.subheader("Download Predictions")
    pred_df = X_test.copy()
    pred_df['ActualChurn'] = y_test
    pred_df['PredictedChurn'] = y_pred
    st.download_button(
        label="Download Predictions as CSV",
        data=pred_df.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )


with tab4:
    st.subheader("Feature Importance from Random Forest")
    feat_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_fi, ax_fi = plt.subplots(figsize=(10,6))
    sns.barplot(x=feat_importance.values, y=feat_importance.index, palette='viridis', ax=ax_fi)
    st.pyplot(fig_fi)

with tab5:
    st.subheader("Combined Dashboard Preview")
    fig, axes = plt.subplots(2, 2, figsize=(16,12))
    sns.countplot(x='Churn', data=filtered_df, palette='coolwarm', ax=axes[0,0])
    axes[0,0].set_title('Churn Distribution')
    sns.boxplot(x='Churn', y='MonthlyCharges', data=filtered_df, palette='Set2', ax=axes[0,1])
    axes[0,1].set_title('Monthly Charges vs Churn')
    sns.boxplot(x='Churn', y='Tenure', data=filtered_df, palette='Set3', ax=axes[1,0])
    axes[1,0].set_title('Tenure vs Churn')
    sns.barplot(x=feat_importance.values, y=feat_importance.index, palette='viridis', ax=axes[1,1])
    axes[1,1].set_title('Feature Importance')
    plt.tight_layout()
    st.pyplot(fig)
