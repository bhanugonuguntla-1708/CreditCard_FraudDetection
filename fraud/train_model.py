import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv(r"C:\Users\Chitt\OneDrive\Desktop\creaditcard_frauddetection\fraud\creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved as model.pkl")