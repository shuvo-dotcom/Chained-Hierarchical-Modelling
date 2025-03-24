from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import re


class BaseModel(ABC):
    def __init__(self) -> None:
        self.defaults = {}
        self.confidence_scores = None

    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the input data
        :param X: Input features
        :return: Predicted labels
        """
        ...

    @abstractmethod
    def data_transform(self) -> None:
        """
        Transform the input data if needed
        """
        ...

    def build(self, values={}):
        """
        Build the model with given parameters
        :param values: Dictionary of parameters
        :return: self
        """
        values = values if isinstance(values, dict) else {}
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self

    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns
        :param text: Input text
        :return: Language code
        """
        # German patterns
        german_patterns = [r'ä', r'ö', r'ü', r'ß', r'der ', r'die ', r'das ']
        if any(re.search(pattern, text.lower()) for pattern in german_patterns):
            return 'de'
        
        # French patterns
        french_patterns = [r'é', r'è', r'ê', r'à', r'le ', r'la ', r'les ']
        if any(re.search(pattern, text.lower()) for pattern in french_patterns):
            return 'fr'
        
        # Spanish patterns
        spanish_patterns = [r'á', r'é', r'í', r'ó', r'ú', r'ñ', r'el ', r'la ']
        if any(re.search(pattern, text.lower()) for pattern in spanish_patterns):
            return 'es'
        
        # Italian patterns
        italian_patterns = [r'à', r'è', r'é', r'ì', r'ò', r'ù', r'il ', r'la ']
        if any(re.search(pattern, text.lower()) for pattern in italian_patterns):
            return 'it'
        
        # Portuguese patterns
        portuguese_patterns = [r'á', r'à', r'ã', r'é', r'ê', r'ó', r'ô', r'o ']
        if any(re.search(pattern, text.lower()) for pattern in portuguese_patterns):
            return 'pt'
        
        return 'en'  # Default to English

    def get_confidence_scores(self) -> Optional[np.ndarray]:
        """
        Get confidence scores for the last prediction
        :return: Array of confidence scores
        """
        return self.confidence_scores

    def get_prediction_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions with confidence scores
        :param X: Input features
        :return: Tuple of (predictions, confidence_scores)
        """
        predictions = self.predict(X)
        return predictions, self.confidence_scores
