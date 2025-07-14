# Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Data Collection & Pre-processing
raw_mail_data = pd.read_csv('mail_data.csv')
print(raw_mail_data.head())
print("Null values in each column:\n", raw_mail_data.isnull().sum())
print("Dataset shape:", raw_mail_data.shape)

# Label encoding: spam = 0, ham = 1
raw_mail_data['Category'] = raw_mail_data['Category'].map({'spam': 0, 'ham': 1})

# Separating data into texts and labels
X = raw_mail_data['Message']
Y = raw_mail_data['Category'].astype('int')

# Splitting the data: 80% train, 20% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=3)

# Feature Extraction using TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Prediction on training data
train_predictions = model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_predictions)
print('Training Accuracy:', train_accuracy)

# Prediction on test data
test_predictions = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_predictions)
print('Test Accuracy:', test_accuracy)

# Detailed Evaluation
print("\nClassification Report (on test data):\n")
print(classification_report(Y_test, test_predictions, target_names=["Spam", "Ham"]))

# Predictive System
input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfill my promise. You have been wonderful and a blessing at all times."]

# Convert input mail to feature vector
input_data_features = feature_extraction.transform(input_mail)

# Make prediction
prediction = model.predict(input_data_features)
print("\nPrediction for input mail:", prediction[0])

if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")


