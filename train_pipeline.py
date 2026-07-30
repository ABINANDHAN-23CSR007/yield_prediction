import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd

# Suppress all log messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from hybrid_model import load_crop_data, HybridCropYieldModel, preprocess_and_feature_engineering

def calculate_metrics(y_true, y_pred, is_log_scale=True):
    """Calculates regression metrics (MAE, MSE, RMSE, R2) on the original scale."""
    if is_log_scale:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
        
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    mse = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def main():
    crop_path = "district-season-and-crop-wise-area-production-and-yield-statistics-for-tamil-nadu.xlsx"
    
    if not os.path.exists(crop_path):
        print("Error: Required dataset file is missing in workspace.")
        return
        
    df_raw = load_crop_data(crop_path)
    df = preprocess_and_feature_engineering(df_raw)
    
    categorical_cols = ['district', 'crop', 'season']
    numerical_cols = ['Log_Area', 'Crop_Mean', 'District_Mean', 'Season_Mean', 'Crop_District_Mean']
    
    # Train test split (80/20 stratified by crop type)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['crop'])
    
    # Setup encoders and scalers on train set
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()
    
    # Extract training target (log scale)
    y_train = np.log1p(train_df['crop_yield'].values)
    y_test_log = np.log1p(test_df['crop_yield'].values)
    y_test_orig = test_df['crop_yield'].values
    
    # Initialize and train Hybrid Model
    hybrid_model = HybridCropYieldModel(encoder, scaler, categorical_cols, numerical_cols)
    hybrid_model.fit(train_df, y_train, epochs=30, batch_size=64)
    
    # Model predictions on Test Data
    X_test_encoded = hybrid_model._transform_features(test_df)
    
    # Individual model predictions
    lstm_test_pred_log = hybrid_model.lstm_model.predict(X_test_encoded[:, np.newaxis, :], verbose=0).flatten()
    rf_test_pred_log = hybrid_model.rf_model.predict(X_test_encoded)
    xgb_test_pred_log = hybrid_model.xgb_model.predict(X_test_encoded)
    
    # Hybrid Model predictions (predicted yield is already in original scale)
    hybrid_test_pred_orig = hybrid_model.predict(test_df)
    
    # Metrics evaluation (Calculated on original Tons/Hectare scale)
    lstm_metrics = calculate_metrics(y_test_log, lstm_test_pred_log)
    rf_metrics = calculate_metrics(y_test_log, rf_test_pred_log)
    xgb_metrics = calculate_metrics(y_test_log, xgb_test_pred_log)
    hybrid_metrics = calculate_metrics(y_test_orig, hybrid_test_pred_orig, is_log_scale=False)
    
    # Print metrics table
    results_df = pd.DataFrame({
        'LSTM': lstm_metrics,
        'Random Forest': rf_metrics,
        'XGBoost': xgb_metrics,
        'Hybrid Model': hybrid_metrics
    }).T
    
    print("\n" + "="*50)
    print("           MODEL PERFORMANCE COMPARISON (ORIGINAL SCALE)")
    print("="*50)
    print(results_df.to_string())
    print("="*50 + "\n")
    
    # Save model binaries
    os.makedirs("models", exist_ok=True)
    with open("models/best_yield_model.pkl", "wb") as f:
        pickle.dump(hybrid_model, f)
    with open("models/hybrid_model.pkl", "wb") as f:
        pickle.dump(hybrid_model, f)
    with open("models/yield_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("models/yield_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    # 8. Predict on sample data
    print("="*50)
    print("           PREDICTION FOR SAMPLE DATA")
    print("="*50)
    
    sample_data = {
        'district': ['COIMBATORE'],
        'crop': ['Rice'],
        'season': ['Kharif'],
        'area': [2.5]
    }
    sample_df = pd.DataFrame(sample_data)
    sample_df_preprocessed = preprocess_and_feature_engineering(sample_df)
    
    sample_pred = hybrid_model.predict(sample_df_preprocessed)
    print("Sample Input Parameters:")
    for k, v in sample_data.items():
        print(f"  {k:<12}: {v[0]}")
    print(f"\nPredicted Yield : {sample_pred[0]:.4f} Tons/Hectare")
    print("="*50)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*50)
        print(" [INFO] Execution cancelled by user (Ctrl+C).")
        print("="*50)
