import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def preprocess_and_feature_engineering_fertilizer(df):
    """
    Applies soil nutrient ratios, aggregate nutrients, and climate/moisture interactions.
    """
    df = df.copy()
    
    eps = 1e-5
    df['N_P_Ratio'] = df['Nitrogen'] / (df['Phosphorous'] + eps)
    df['N_K_Ratio'] = df['Nitrogen'] / (df['Potassium'] + eps)
    df['P_K_Ratio'] = df['Phosphorous'] / (df['Potassium'] + eps)
    df['Nutrient_Sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    
    # Climate / Moisture interaction features
    df['Temp_Hum_Interaction'] = df['Temparature'] * df['Humidity']
    df['Temp_Moisture_Interaction'] = df['Temparature'] * df['Moisture']
    df['Hum_Moisture_Interaction'] = df['Humidity'] * df['Moisture']
    
    return df

class FertilizerRecommendationModel:
    """
    Hybrid Stacking Classifier for Fertilizer Recommendation:
    - Preprocessing features via OneHotEncoder & StandardScaler
    - Random Forest Sub-Classifier
    - XGBoost Sub-Classifier
    - Stacking layer fusing predicted probabilities
    - Meta-Classifier (Logistic Regression)
    """
    def __init__(self, encoder, scaler, label_encoder, categorical_cols, numerical_cols):
        self.encoder = encoder
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        
        self.rf_model = None
        self.xgb_model = None
        self.meta_model = None

    def _transform_features(self, X_raw):
        """Encodes and scales inputs using pre-fitted encoders/scalers."""
        encoded_cats = self.encoder.transform(X_raw[self.categorical_cols])
        scaled_nums = self.scaler.transform(X_raw[self.numerical_cols])
        return np.hstack([encoded_cats, scaled_nums]).astype(np.float64)

    def fit(self, X_train_raw, y_train_labels):
        """
        Trains the sub-classifiers, extracts predicted probabilities, and fits the meta-classifier.
        y_train_labels should be raw string/categorical labels (which will be label-encoded).
        """
        X_train_encoded = self._transform_features(X_train_raw)
        y_train = self.label_encoder.transform(y_train_labels)
        
        # 1. Train Random Forest Classifier
        self.rf_model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_train_encoded, y_train)
        
        # 2. Train XGBoost Classifier
        self.xgb_model = xgb.XGBClassifier(
            max_depth=10, 
            n_estimators=1000, 
            random_state=42, 
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        self.xgb_model.fit(X_train_encoded, y_train)
        
        # 3. Stack predictions (using probabilities for soft voting stacking)
        rf_probs = self.rf_model.predict_proba(X_train_encoded)
        xgb_probs = self.xgb_model.predict_proba(X_train_encoded)
        
        stacked_features = np.hstack([rf_probs, xgb_probs])
        
        # 4. Train Meta-Classifier stacking layer
        self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
        self.meta_model.fit(stacked_features, y_train)

    def predict(self, X_input):
        """
        Predicts fertilizer recommendation.
        Returns predicted fertilizer name as raw string labels.
        """
        if isinstance(X_input, pd.DataFrame):
            X_encoded = self._transform_features(X_input)
        else:
            X_encoded = np.asarray(X_input, dtype=np.float64)

        rf_probs = self.rf_model.predict_proba(X_encoded)
        xgb_probs = self.xgb_model.predict_proba(X_encoded)
        
        stacked_features = np.hstack([rf_probs, xgb_probs])
        pred_encoded = self.meta_model.predict(stacked_features)
        
        return self.label_encoder.inverse_transform(pred_encoded)

    def predict_proba(self, X_input):
        """
        Returns predicted probability distribution.
        """
        if isinstance(X_input, pd.DataFrame):
            X_encoded = self._transform_features(X_input)
        else:
            X_encoded = np.asarray(X_input, dtype=np.float64)

        rf_probs = self.rf_model.predict_proba(X_encoded)
        xgb_probs = self.xgb_model.predict_proba(X_encoded)
        
        stacked_features = np.hstack([rf_probs, xgb_probs])
        return self.meta_model.predict_proba(stacked_features)
