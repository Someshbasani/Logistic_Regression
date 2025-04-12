import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Title
st.title("Model Deployment: Logistic Regression")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# Function to take user input
def user_input_features():
    pclass = st.selectbox('Passenger Class', [1, 2, 3])
    age = st.number_input('Age', min_value=0, max_value=100)
    sibsp = st.number_input('Number of Siblings/Spouses', min_value=0)
    parch = st.number_input('Number of Parents/Children', min_value=0)
    fare = st.number_input('Fare', min_value=0.0)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

    # Convert categorical inputs to numerical
    sex_encoded = 1 if sex == 'Male' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0

    # Create DataFrame
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex': [sex_encoded],
        'Embarked_Q': [embarked_Q],
        'Embarked_S': [embarked_S]
    })
    
    return data

# Get user input
df = user_input_features()
st.subheader("User Input Parameters")
st.write(df)

# Load and preprocess Titanic dataset
@st.cache_data
def load_and_preprocess_data():
    # Load datasets
    train = pd.read_csv(r"C:\Users\basan\Desktop\datascience class\Assignments\7.Logistic Regression\Logistic Regression\Titanic_train.csv")
    test = pd.read_csv(r"C:\Users\basan\Desktop\datascience class\Assignments\7.Logistic Regression\Logistic Regression\Titanic_test.csv")

    # Drop unnecessary columns
    train.drop(['PassengerId', 'Name', 'Cabin','Ticket'], axis=1, inplace=True)
    test.drop(['PassengerId', 'Name', 'Cabin','Ticket'], axis=1, inplace=True)

    # Handle missing values
    for df in [train, test]:
        df['Age'].fillna(df['Age'].mean(), inplace=True)
        df['Fare'].fillna(df['Fare'].mean(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Encode 'Sex'
    le_sex = LabelEncoder()
    train['Sex'] = le_sex.fit_transform(train['Sex'])
    test['Sex'] = le_sex.transform(test['Sex'])

    # One-hot encode 'Embarked'
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(train[['Embarked']])
    
    # Transform both datasets
    train_embarked = encoder.transform(train[['Embarked']])
    test_embarked = encoder.transform(test[['Embarked']])
    
    # Create DataFrames and concatenate
    train_embarked_df = pd.DataFrame(train_embarked, columns=encoder.get_feature_names_out(['Embarked']))
    test_embarked_df = pd.DataFrame(test_embarked, columns=encoder.get_feature_names_out(['Embarked']))
    
    train = pd.concat([train.drop('Embarked', axis=1), train_embarked_df], axis=1)
    test = pd.concat([test.drop('Embarked', axis=1), test_embarked_df], axis=1)

    return train, test

# Load and preprocess data
Data_train, Data_test = load_and_preprocess_data()

# Prepare training data
X_train = Data_train.drop('Survived', axis=1)
y_train = Data_train['Survived']

# Train Logistic Regression model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Ensure user input has all features
for feature in classifier.feature_names_in_:
    if feature not in df.columns:
        df[feature] = 0

# Reorder columns to match training data
df = df[classifier.feature_names_in_]

# Make predictions
prediction = classifier.predict(df)
prediction_proba = classifier.predict_proba(df)

# Display results
st.subheader('Prediction')
st.write('Survived' if prediction[0] == 1 else 'Did not survive')

st.subheader('Prediction Probability')
st.write(prediction_proba)