import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r"D:\niran's\UNH\projects\team 4 DS\data1.csv")

# Handling missing data
df['education'].fillna(df['education'].mode()[0], inplace=True)
df['cigsPerDay'].fillna(df['cigsPerDay'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)
df['totChol'].fillna(df['totChol'].median(), inplace=True)
df['glucose'].fillna(df['glucose'].median(), inplace=True)
df['BPMeds'].fillna(df['BPMeds'].mode()[0], inplace=True)
df['heartRate'].fillna(df['heartRate'].median(), inplace=True)

# Implementing train test split
X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Decision Tree Classifier
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Random Forest Classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Streamlit App
st.title("Cardiovascular Disease Prediction App")

# Display dataset
st.subheader("Dataset:")
st.dataframe(df.head())

# Display model accuracies
st.subheader("Model Accuracies:")
st.write(f"Logistic Regression Accuracy: {accuracy_lr:.2%}")
st.write(f"Decision Tree Accuracy: {accuracy_dt:.2%}")
st.write(f"Random Forest Accuracy: {accuracy_rf:.2%}")

# Sidebar for user input
st.sidebar.title("User Input")
user_input = st.sidebar.text_input("Enter a new data point (comma-separated):")

if user_input:
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([map(float, user_input.split(','))], columns=X.columns)

    # Make predictions
    prediction_lr = model_lr.predict(user_df)[0]
    prediction_dt = model_dt.predict(user_df)[0]
    prediction_rf = model_rf.predict(user_df)[0]

    # Display predictions
    st.sidebar.subheader("Predictions:")
    st.sidebar.write(f"Logistic Regression Prediction: {prediction_lr}")
    st.sidebar.write(f"Decision Tree Prediction: {prediction_dt}")
    st.sidebar.write(f"Random Forest Prediction: {prediction_rf}")
