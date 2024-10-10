from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.pipeline import make_pipeline

# Load data
data = pd.read_csv(r'C:\Users\moham\Documents\Machine learning dataset\spam.csv', encoding='ISO-8859-1')

# Preprocess
X = data['Message']
y = data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load trained model (you should have the model saved)
st.title('Email Spam Classifier')

# Text input for the user to enter the message
user_input = st.text_area("Enter email text to classify")

# When user submits the form
if st.button('Classify'):
    prediction = model.predict([user_input])
    st.write(f'The email is classified as: {"SPAM" if prediction[0] == "spam" else "NOT SPAM"}.')

# Show model accuracy
st.write(f'Model Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix button
if st.button('Show Confusion Matrix'):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Plotting the confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax)
    st.pyplot(fig)
