import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import random

num_folds = 5
seed = 0
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)

class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.scaler = StandardScaler()
        
        # Create ensemble of models with different parameters
        # Create ensemble of models with different parameters
        base_models = [
            RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample'),
            RandomForestClassifier(n_estimators=800, random_state=seed+1, class_weight='balanced'),
            RandomForestClassifier(n_estimators=1200, random_state=seed+2, class_weight='balanced_subsample')
        ]
        
        # Wrap each base model with MultiOutputClassifier
        self.models = [MultiOutputClassifier(model) for model in base_models]
        self.predictions = None
        self.confidence_scores = None
        self.data_transform()

    def train(self, data) -> None:
        """
        Train the Random Forest model using ensemble method
        :param data: Data object containing training data
        """
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(data.X_train)
        
        # Reshape y_train to 2D array if it's 1D
        y_train_reshaped = data.y_train.reshape(-1, 1) if len(data.y_train.shape) == 1 else data.y_train
        
        # Train each model in the ensemble
        for model in self.models:
            model.fit(X_train_scaled, y_train_reshaped)
            
        # Perform cross-validation
        cv_scores = cross_val_score(self.models[0], X_train_scaled, y_train_reshaped, cv=num_folds)
        print(f"\nCross-validation scores for {self.model_name}:")
        print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble of models
        :param X_test: Test features
        :return: Predicted labels
        """
        # Scale the test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions from each model
        all_predictions = []
        all_probabilities = []
        
        for model in self.models:
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)[0]  # Take first output's probabilities
            all_predictions.append(pred)
            all_probabilities.append(prob)
        
        # Combine predictions using majority voting
        final_predictions = np.zeros(len(X_test), dtype=np.int32)
        final_probabilities = np.zeros((len(X_test), len(np.unique(self.y))))
        
        for i in range(len(X_test)):
            # Get predictions from all models for this sample
            sample_predictions = [pred[i][0] for pred in all_predictions]  # Take first output
            sample_probabilities = [prob[i] for prob in all_probabilities]
            
            # Majority voting
            unique_labels, counts = np.unique(sample_predictions, return_counts=True)
            final_predictions[i] = unique_labels[np.argmax(counts)]
            
            # Average probabilities
            final_probabilities[i] = np.mean(sample_probabilities, axis=0)
        
        self.predictions = final_predictions
        self.confidence_scores = np.max(final_probabilities, axis=1)
        return final_predictions

    def print_results(self, data):
        """
        Print detailed evaluation metrics
        :param data: Data object containing test data
        """
        # Convert predictions back to original labels
        decoded_predictions = data.decode_predictions(self.predictions)
        decoded_y_test = data.decode_predictions(data.y_test)
        
        print(f"\nDetailed Results for {self.model_name}:")
        print("\nClassification Report:")
        print(classification_report(decoded_y_test, decoded_predictions))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(decoded_y_test, decoded_predictions))
        
        print("\nConfidence Score Statistics:")
        print(f"Mean Confidence: {np.mean(self.confidence_scores):.3f}")
        print(f"Std Confidence: {np.std(self.confidence_scores):.3f}")
        print(f"Min Confidence: {np.min(self.confidence_scores):.3f}")
        print(f"Max Confidence: {np.max(self.confidence_scores):.3f}")

    def data_transform(self) -> None:
        """
        Transform the input data
        - Scale features
        - Handle multilingual content
        """
        # Scale features
        self.embeddings = self.scaler.fit_transform(self.embeddings)

