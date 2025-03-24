from model.randomforest import RandomForest
from modelling.data_model import Data
from Config import *
import numpy as np
from typing import Dict, List, Tuple

class ChainedPredictor:
    def __init__(self, data: Data):
        self.data = data
        self.models = {}
        self.predictions = {}
        self.confidence_scores = {}
        
    def train_models(self):
        # Train Type2 model
        self.models['Type2'] = RandomForest("RandomForest_Type2", 
                                           self.data.get_embeddings(), 
                                           self.data.get_type_y_train())
        self.models['Type2'].train(self.data)
        
        # Train Type3 model
        self.models['Type3'] = RandomForest("RandomForest_Type3", 
                                           self.data.get_embeddings(), 
                                           self.data.get_type_y_train())
        self.models['Type3'].train(self.data)
        
        # Train Type4 model
        self.models['Type4'] = RandomForest("RandomForest_Type4", 
                                           self.data.get_embeddings(), 
                                           self.data.get_type_y_train())
        self.models['Type4'].train(self.data)
        
    def predict(self):
        # Predict Type2 with confidence scores
        self.models['Type2'].predict(self.data.get_X_test())
        self.predictions['Type2'] = self.models['Type2'].predictions
        self.confidence_scores['Type2'] = self.models['Type2'].confidence_scores
        
        # Predict Type3 with confidence scores
        self.models['Type3'].predict(self.data.get_X_test())
        self.predictions['Type3'] = self.models['Type3'].predictions
        self.confidence_scores['Type3'] = self.models['Type3'].confidence_scores
        
        # Predict Type4 with confidence scores
        self.models['Type4'].predict(self.data.get_X_test())
        self.predictions['Type4'] = self.models['Type4'].predictions
        self.confidence_scores['Type4'] = self.models['Type4'].confidence_scores
        
    def calculate_accuracy(self) -> Dict[str, float]:
        y_test = self.data.get_type_y_test()
        accuracies = {}
        
        # Calculate Type2 accuracy with confidence threshold
        type2_correct = np.sum((self.predictions['Type2'] == y_test) & (self.confidence_scores['Type2'] > 0.5))
        type2_accuracy = type2_correct / len(y_test)
        
        # Calculate Type2+Type3 accuracy with confidence threshold
        type2_type3_correct = 0
        for i in range(len(y_test)):
            if (self.predictions['Type2'][i] == y_test[i] and self.confidence_scores['Type2'][i] > 0.5):
                if (self.predictions['Type3'][i] == y_test[i] and self.confidence_scores['Type3'][i] > 0.5):
                    type2_type3_correct += 1
        type2_type3_accuracy = type2_type3_correct / len(y_test)
        
        # Calculate Type2+Type3+Type4 accuracy with confidence threshold
        type2_type3_type4_correct = 0
        for i in range(len(y_test)):
            if (self.predictions['Type2'][i] == y_test[i] and self.confidence_scores['Type2'][i] > 0.5):
                if (self.predictions['Type3'][i] == y_test[i] and self.confidence_scores['Type3'][i] > 0.5):
                    if (self.predictions['Type4'][i] == y_test[i] and self.confidence_scores['Type4'][i] > 0.5):
                        type2_type3_type4_correct += 1
        type2_type3_type4_accuracy = type2_type3_type4_correct / len(y_test)
        
        # Calculate final accuracy with confidence weighting
        final_accuracy = (type2_accuracy + type2_type3_accuracy + type2_type3_type4_accuracy) / 3
        
        # Calculate average confidence scores
        avg_confidence = {
            'Type2': np.mean(self.confidence_scores['Type2']),
            'Type3': np.mean(self.confidence_scores['Type3']),
            'Type4': np.mean(self.confidence_scores['Type4'])
        }
        
        return {
            'Type2_Accuracy': type2_accuracy,
            'Type2_Type3_Accuracy': type2_type3_accuracy,
            'Type2_Type3_Type4_Accuracy': type2_type3_type4_accuracy,
            'Final_Accuracy': final_accuracy,
            'Average_Confidence': avg_confidence
        }
