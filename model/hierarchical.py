import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from mixins import LoggingMixin, ValidationMixin, MetricsMixin, SerializationMixin
from model.base import BaseModel
from typing import Dict
import random

seed = 0
np.random.seed(seed)
random.seed(seed)

class HierarchicalRandomForest(SerializationMixin, MetricsMixin, ValidationMixin, LoggingMixin, BaseModel):
    def __init__(self, model_name: str, embeddings: np.ndarray) -> None:
        """
        Initialize Hierarchical Random Forest Classifier
        :param model_name: Name of the model
        :param embeddings: Input feature embeddings
        """
        super(HierarchicalRandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.level_classifiers = {}
        self.predictions = None
        self.confidence_scores = {}

    def train(self, data) -> None:
        """
        Train the hierarchical model level by level using HierarchicalData
        :param data: HierarchicalData object containing training data
        """
        X_train, y_train_dict = data.get_type2_train_data()

        # Train classifiers for each hierarchical level
        for level, y_train in y_train_dict.items():
            self.log_info(f"Training classifier for level: {level}")

            clf = RandomForestClassifier(
                n_estimators=1000,
                random_state=seed,
                class_weight='balanced_subsample'
            )

            clf.fit(X_train, y_train)
            self.level_classifiers[level] = clf

            # Log training accuracy
            train_pred = clf.predict(X_train)
            accuracy = np.mean(train_pred == y_train)
            self.log_info(f"Training accuracy for {level}: {accuracy:.3f}")

    def predict(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make predictions using the hierarchical model
        :param X_test: Test features
        :return: Dictionary of predictions per hierarchy level
        """
        predictions = {}
        confidence_scores = {}

        for level, clf in self.level_classifiers.items():
            pred = clf.predict(X_test)
            prob = clf.predict_proba(X_test)

            predictions[level] = pred
            confidence_scores[level] = np.max(prob, axis=1)

        self.predictions = predictions
        self.confidence_scores = confidence_scores

        return predictions

    def print_results(self, data) -> None:
        """
        Print detailed evaluation metrics for each hierarchy level
        :param data: HierarchicalData object containing test data
        """
        _, y_test_dict = data.get_type2_test_data()

        for level, y_true in y_test_dict.items():
            self.log_info(f"\n=== Results for {level} ===")

            y_pred = self.predictions[level]

            self.log_info("\nClassification Report:")
            self.log_info(classification_report(y_true, y_pred))

            self.log_info("\nConfusion Matrix:")
            self.log_info(confusion_matrix(y_true, y_pred))

            confidence = self.confidence_scores[level]
            self.log_info("\nConfidence Score Statistics:")
            self.log_info(f"Mean Confidence: {np.mean(confidence):.3f}")
            self.log_info(f"Std Confidence: {np.std(confidence):.3f}")
            self.log_info(f"Min Confidence: {np.min(confidence):.3f}")
            self.log_info(f"Max Confidence: {np.max(confidence):.3f}")
