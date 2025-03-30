import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Splitting data into features and target
X = data.drop(columns=['price_range'])
y = data['price_range']

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
with open("models/random_forest.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
