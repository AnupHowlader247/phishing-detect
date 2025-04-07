import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("Dataset.csv")

# Select features and target
features = [
    'url_length',
    'number_of_dots_in_url',
    'having_repeated_digits_in_url',
    'number_of_special_char_in_url',
    'entropy_of_url',
    'entropy_of_domain'
]
X = df[features]
y = df['Type']

# Use first 1000 rows to avoid memory issues
X = X.head(1000)
y = y.head(1000)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200, solver='liblinear')
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
