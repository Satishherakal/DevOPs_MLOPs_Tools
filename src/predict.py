import pickle

def load_model(model_path='models/rf_model.pkl'):
    """Load the trained model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict(model, data):
    """Predict using the loaded model."""
    return model.predict(data)

if __name__ == "__main__":
    try:
        clf = load_model()
        # Sample data: Sepal Length, Sepal Width, Petal Length, Petal Width
        sample = [[5.1, 3.5, 1.4, 0.2]]
        pred = predict(clf, sample)
        print(f"Predicted class: {pred[0]}")
    except FileNotFoundError:
        print("Model not found. Please run train.py first.")
