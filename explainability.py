import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
import lime
import lime.lime_tabular

def explain_model_global(hybrid_model, X_train_encoded, feature_names, output_dir):
    """
    Computes global feature contributions using SHAP KernelExplainer.
    Saves the SHAP summary plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define a prediction function for the preprocessed features
    def shap_predict_fn(X_encoded_batch):
        # Shape: (batch_size, encoded_dim)
        X_lstm = X_encoded_batch[:, np.newaxis, :]
        X_lstm_tensor = tf.cast(X_lstm, tf.float32)
        lstm_preds = hybrid_model.lstm_model(X_lstm_tensor, training=False).numpy().reshape(-1, 1)
        rf_preds = hybrid_model.rf_model.predict(X_encoded_batch).reshape(-1, 1)
        xgb_preds = hybrid_model.xgb_model.predict(X_encoded_batch).reshape(-1, 1)
        
        final_features = np.hstack([lstm_preds, rf_preds, xgb_preds])
        return hybrid_model.meta_model.predict(final_features)
    
    # Sample background data to speed up Kernel SHAP calculation
    background_data = shap.sample(X_train_encoded, 50)
    
    # Initialize KernelExplainer
    explainer = shap.KernelExplainer(shap_predict_fn, background_data)
    
    # Explain a subset of test records for visualization (e.g., 20 records for speed)
    test_sample = shap.sample(X_train_encoded, 20)
    shap_values = explainer.shap_values(test_sample, silent=True)
    
    # Save SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Contribution Summary (Hybrid Model)", fontsize=14, pad=20)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    return shap_values, test_sample

def explain_prediction_local(hybrid_model, X_train_encoded, sample_to_explain, feature_names, output_dir, file_prefix="lime_local_exp"):
    """
    Computes local feature contributions for a single sample using LIME.
    Saves explanation as an interactive HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define prediction function
    def lime_predict_fn(X_encoded_batch):
        X_lstm = X_encoded_batch[:, np.newaxis, :]
        X_lstm_tensor = tf.cast(X_lstm, tf.float32)
        lstm_preds = hybrid_model.lstm_model(X_lstm_tensor, training=False).numpy().reshape(-1, 1)
        rf_preds = hybrid_model.rf_model.predict(X_encoded_batch).reshape(-1, 1)
        xgb_preds = hybrid_model.xgb_model.predict(X_encoded_batch).reshape(-1, 1)
        
        final_features = np.hstack([lstm_preds, rf_preds, xgb_preds])
        return hybrid_model.meta_model.predict(final_features)
        
    # Initialize LIME Tabular Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_encoded,
        feature_names=feature_names,
        class_names=['crop_yield'],
        mode='regression',
        random_state=42
    )
    
    # Generate explanation
    exp = explainer.explain_instance(
        data_row=sample_to_explain,
        predict_fn=lime_predict_fn,
        num_features=10
    )
    
    # Save to HTML
    html_path = os.path.join(output_dir, f"{file_prefix}.html")
    exp.save_to_file(html_path)
    
    return exp
