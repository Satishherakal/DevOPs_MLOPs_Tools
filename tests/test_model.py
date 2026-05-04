import unittest
import numpy as np
import os
from src.train import train_model
from src.predict import load_model, predict

class TestRandomForestModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Train the model once before running tests."""
        cls.acc = train_model()
        cls.model_path = 'models/rf_model.pkl'

    def test_model_training(self):
        """Test if the model achieves reasonable accuracy."""
        self.assertGreater(self.acc, 0.8)
        
    def test_model_saving(self):
        """Test if the model file is created."""
        self.assertTrue(os.path.exists(self.model_path))
        
    def test_prediction(self):
        """Test prediction with valid data."""
        model = load_model(self.model_path)
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        pred = predict(model, sample)
        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], [0, 1, 2]) # Valid iris classes

if __name__ == '__main__':
    unittest.main()
