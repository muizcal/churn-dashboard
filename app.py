import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score


st.title("Telecom Customer Churn Dashboard")
st.markdown("Analyze customer churn, explore data, and see ML predictions.")


df = pd.read_csv('telecom_churn_dataset.csv')


st.sidebar.header("Filters")
selected_contract = st.sidebar.multiselect("Contract Type", df['Contract'].unique(), default=df['Contract'].unique())
selected_payment = st.sidebar.multiselect("Payment Method", df['PaymentMethod'].unique(), default=df['PaymentMethod'].unique())

filtered_df = df[df['Contract'].isin(selected_contract) & df['PaymentMethod'].isin(selected_payment)]

st.subheader("Filtered Data")
st.dataframe(filtered_df.head(20))


st.subheader("Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Churn', data=filtered_df, palette='coolwarm', ax=ax)
st.pyplot(fig)

st.subheader("Monthly Charges vs Churn")
fig2, ax2 = plt.subplots()
sns.boxplot(x='Churn', y='MonthlyCharges', data=filtered_df, palette='Set2', ax=ax2)
st.pyplot(fig2)

st.subheader("Tenure vs Churn")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Churn', y='Tenure', data=filtered_df, palette='Set3', ax=ax3)
st.pyplot(fig3)


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]


st.subheader("Model Metrics")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
st.pyplot(fig_cm)

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

st.subheader("Feature Importance")
feat_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
fig_fi, ax_fi = plt.subplots(figsize=(10,6))
sns.barplot(x=feat_importance.values, y=feat_importance.index, palette='viridis', ax=ax_fi)
st.pyplot(fig_fi)

