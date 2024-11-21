import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the Dataset
# Assume 'fake_real' is a folder where the dataset files are stored
fake_df = pd.read_csv('fake_real/Fake.csv')
true_df = pd.read_csv('fake_real/True.csv')

# 2. Label the Data
# Add a 'label' column where 0 is for fake news and 1 is for real news
fake_df['label'] = 0
true_df['label'] = 1

# 3. Combine the DataFrames
df = pd.concat([fake_df, true_df], ignore_index=True)

# 4. Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# 5. Prepare data for training
X = df['text']  # Features: News articles
y = df['label']  # Labels: 0 for fake, 1 for real

# 6. Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# 7. Initialize and fit the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# 8. Initialize and train the Passive Aggressive Classifier model
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 9. Make predictions on the test set
y_pred = pac.predict(tfidf_test)

# 10. Evaluate the model (optional)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 11. Save the model and vectorizer to disk using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(pac, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

print("Model and vectorizer saved to 'model.pkl' and 'vectorizer.pkl'")
