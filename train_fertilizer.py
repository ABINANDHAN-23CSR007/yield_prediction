import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

from fertilizer_model import preprocess_and_feature_engineering_fertilizer, FertilizerRecommendationModel

def main():
    soil_path = "data_core.csv"
    
    if not os.path.exists(soil_path):
        print(f"Error: Required dataset file '{soil_path}' is missing in workspace.")
        return
        
    # 1. Load dataset
    df_raw = pd.read_csv(soil_path)
    df_raw = df_raw.drop_duplicates().reset_index(drop=True)
    
    # Drop rows where target count is too low to stratify (e.g., class '20-20' has only 1 sample)
    target_col = 'Fertilizer Name'
    counts = df_raw[target_col].value_counts()
    rare_classes = counts[counts < 2].index
    if len(rare_classes) > 0:
        print(f"Dropping rare classes with fewer than 2 samples: {list(rare_classes)}")
        df_raw = df_raw[~df_raw[target_col].isin(rare_classes)].reset_index(drop=True)
        
    # 2. Preprocessing & Feature Engineering
    df = preprocess_and_feature_engineering_fertilizer(df_raw)
    
    categorical_cols = ['Soil Type', 'Crop Type']
    numerical_cols = [
        'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium',
        'N_P_Ratio', 'N_K_Ratio', 'P_K_Ratio', 'Nutrient_Sum',
        'Temp_Hum_Interaction', 'Temp_Moisture_Interaction', 'Hum_Moisture_Interaction'
    ]
    
    # 3. Train Test Split (80/20 stratified by Fertilizer Name)
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df[target_col]
    )
    
    # 4. Fit encoders and scalers
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(train_df[categorical_cols])
    
    scaler = StandardScaler()
    scaler.fit(train_df[numerical_cols])
    
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[target_col])
    
    # 5. Initialize and train model
    model = FertilizerRecommendationModel(encoder, scaler, label_encoder, categorical_cols, numerical_cols)
    print("Training Hybrid Stacking Classifier for Fertilizer Recommendation...")
    model.fit(train_df, train_df[target_col])
    
    # 6. Evaluation
    test_preds = model.predict(test_df)
    test_true = test_df[target_col].values
    
    accuracy = accuracy_score(test_true, test_preds)
    print("\n" + "="*50)
    print("        FERTILIZER RECOMMENDATION MODEL EVALUATION")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.4%}")
    print("-"*50)
    print("Classification Report:")
    print(classification_report(test_true, test_preds))
    print("="*50 + "\n")
    # Model serialization
    os.makedirs("models", exist_ok=True)
    with open("models/fertilizer_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/fertilizer_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/fertilizer_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open("models/fertilizer_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
        
    print("Saved fertilizer recommendation model components to models/")
    
    # 7. Sample Prediction
    print("="*50)
    print("           PREDICTION FOR SAMPLE SOIL DATA")
    print("="*50)
    sample_soil = {
        'Soil Type': ['Loamy'],
        'Crop Type': ['Maize'],
        'Temparature': [28.0],
        'Humidity': [55.0],
        'Moisture': [40.0],
        'Nitrogen': [85],
        'Phosphorous': [42],
        'Potassium': [40]
    }
    sample_df = pd.DataFrame(sample_soil)
    sample_df_preprocessed = preprocess_and_feature_engineering_fertilizer(sample_df)
    
    pred_fertilizer = model.predict(sample_df_preprocessed)
    pred_prob = model.predict_proba(sample_df_preprocessed)[0]
    
    print("Sample Soil Input Parameters:")
    for k, v in sample_soil.items():
        print(f"  {k:<12}: {v[0]}")
        
    print(f"\nRecommended Fertilizer: {pred_fertilizer[0]}")
    print("Class Probabilities:")
    for class_name, prob in zip(label_encoder.classes_, pred_prob):
        print(f"  {class_name:<10}: {prob:.2%}")
    print("="*50)

if __name__ == '__main__':
    main()
