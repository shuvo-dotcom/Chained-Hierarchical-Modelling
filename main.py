from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from modelling.chained_prediction import ChainedPredictor
import random
import pandas as pd
import numpy as np
from typing import Dict, List
import json

seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    df = get_input_data()
    return df

def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    # Prepare data for each type
    type2_data = Data(data.get_embeddings(), df)
    type3_data = Data(data.get_embeddings(), df)
    type4_data = Data(data.get_embeddings(), df)
    
    # Create and train chained predictor
    predictor = ChainedPredictor(type2_data)
    predictor.train_models()
    
    # Make predictions
    predictor.predict()
    
    # Calculate and print accuracies
    results = predictor.calculate_accuracy()
    
    print(f"\nDetailed Results for {name}:")
    print(f"Type2 Accuracy: {results['Type2_Accuracy']:.2%}")
    print(f"Type2+Type3 Accuracy: {results['Type2_Type3_Accuracy']:.2%}")
    print(f"Type2+Type3+Type4 Accuracy: {results['Type2_Type3_Type4_Accuracy']:.2%}")
    print(f"Final Accuracy: {results['Final_Accuracy']:.2%}")
    
    print("\nAverage Confidence Scores:")
    for type_name, confidence in results['Average_Confidence'].items():
        print(f"{type_name}: {confidence:.3f}")
    
    # Save detailed results to JSON
    results['Group'] = name
    save_results(results)

def save_results(results: Dict):
    try:
        with open('results.json', 'r') as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = []
    
    all_results.append(results)
    
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

def analyze_results():
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
        
        print("\nOverall Analysis:")
        print("-" * 50)
        
        # Calculate average accuracies across all groups
        avg_accuracies = {
            'Type2': np.mean([r['Type2_Accuracy'] for r in results]),
            'Type2+Type3': np.mean([r['Type2_Type3_Accuracy'] for r in results]),
            'Type2+Type3+Type4': np.mean([r['Type2_Type3_Type4_Accuracy'] for r in results]),
            'Final': np.mean([r['Final_Accuracy'] for r in results])
        }
        
        print("\nAverage Accuracies Across All Groups:")
        for name, acc in avg_accuracies.items():
            print(f"{name}: {acc:.2%}")
        
        # Calculate average confidence scores
        avg_confidence = {
            'Type2': np.mean([r['Average_Confidence']['Type2'] for r in results]),
            'Type3': np.mean([r['Average_Confidence']['Type3'] for r in results]),
            'Type4': np.mean([r['Average_Confidence']['Type4'] for r in results])
        }
        
        print("\nAverage Confidence Scores Across All Groups:")
        for name, conf in avg_confidence.items():
            print(f"{name}: {conf:.3f}")
            
    except FileNotFoundError:
        print("No results file found. Run the model first to generate results.")

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED)
    
    for name, group_df in grouped_df:
        print(f"\nProcessing group: {name}")
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        perform_modelling(data, group_df, name)
    
    analyze_results()