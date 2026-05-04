import os
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    """Train a Random Forest model on the Iris dataset."""
    print("Loading data...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # We will use DVC to track a CSV version later, but for now we load from sklearn
    os.makedirs('data', exist_ok=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc:.4f}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Model saved to models/rf_model.pkl")
    
    return acc

if __name__ == "__main__":
    train_model()
