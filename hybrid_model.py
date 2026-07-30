import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization, Reshape
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_crop_data(crop_path="district-season-and-crop-wise-area-production-and-yield-statistics-for-tamil-nadu.xlsx"):
    """
    Loads and cleans the Crop Production Excel workbook.
    """
    crop_df = pd.read_excel(crop_path, sheet_name='sheet1')
    
    # Standardize column naming convention
    crop_df = crop_df.rename(columns={
        'District': 'district',
        'Crop': 'crop',
        'Season': 'season',
        'Area': 'area',
        'Production': 'production',
        'Yield': 'crop_yield'
    })
    
    # Clean target and extreme outliers
    crop_df = crop_df[crop_df['crop_yield'] > 0]
    crop_df = crop_df[~((crop_df['crop'] == 'Rice') & (crop_df['crop_yield'] > 12))]
    crop_df = crop_df[~((crop_df['crop'] == 'Sugarcane') & (crop_df['crop_yield'] > 150))]
    crop_df = crop_df[~((crop_df['crop'] == 'Groundnut') & (crop_df['crop_yield'] > 12))]
    crop_df = crop_df[~((crop_df['crop'] == 'Maize') & (crop_df['crop_yield'] > 15))]
    
    crop_df = crop_df.reset_index(drop=True)
    print(f"Crop dataset shape: {crop_df.shape}")
    return crop_df

def preprocess_and_feature_engineering(df):
    """
    Applies log transformation to area.
    """
    df = df.copy()
    # Log transform area
    df['Log_Area'] = np.log1p(df['area'])
    return df

def build_lstm_model(input_dim):
    """
    Builds a dense Feed-Forward neural network mapped as a sequence model.
    Accepts 3D tensors of shape (None, 1, input_dim) to keep compatibility.
    """
    inputs = Input(shape=(1, input_dim))
    x = Reshape((input_dim,))(inputs)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Representation layer
    dense_rep = Dense(64, activation='relu', name='dense_representation')(x)
    outputs = Dense(1, name='output')(dense_rep)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

class HybridCropYieldModel:
    """
    Wrapper model implementing Stacking & Feature Fusion:
    - Preprocessing features via OneHotEncoder & StandardScaler
    - Feed-Forward NN sub-model (fully shape-compatible with LSTM interface)
    - Random Forest sub-model
    - XGBoost sub-model
    - Stacking feature fusion layer
    - Meta-Regressor (Ridge Regression)
    """
    def __init__(self, encoder, scaler, categorical_cols, numerical_cols):
        self.encoder = encoder
        self.scaler = scaler
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        
        self.lstm_model = None
        self.lstm_feat_extractor = None
        self.rf_model = None
        self.xgb_model = None
        self.meta_model = None
        
        # Target encoding parameters
        self.global_mean = 0.0
        self.crop_means = {}
        self.district_means = {}
        self.season_means = {}

    def _transform_features(self, X_raw):
        """Encodes and scales inputs using pre-fitted encoders/scalers and Target Encoding."""
        X_df = X_raw.copy()
        
        # Apply target encoding maps
        X_df['Crop_Mean'] = X_df['crop'].map(self.crop_means).fillna(self.global_mean)
        X_df['District_Mean'] = X_df['district'].map(self.district_means).fillna(self.global_mean)
        X_df['Season_Mean'] = X_df['season'].map(self.season_means).fillna(self.global_mean)
        X_df['Crop_District_Mean'] = X_df['Crop_Mean'] * X_df['District_Mean']
        
        encoded_cats = self.encoder.transform(X_df[self.categorical_cols])
        scaled_nums = self.scaler.transform(X_df[self.numerical_cols])
        return np.hstack([encoded_cats, scaled_nums]).astype(np.float64)

    def fit(self, X_train_raw, y_train, validation_data=None, epochs=30, batch_size=64):
        """Trains the sub-models, extracts features, and fits the meta-regressor."""
        # 1. Fit Target Encoding
        train_df = X_train_raw.copy()
        y_train_orig = np.expm1(y_train)
        train_df['crop_yield'] = y_train_orig
        
        self.global_mean = train_df['crop_yield'].mean()
        self.crop_means = train_df.groupby('crop')['crop_yield'].mean().to_dict()
        self.district_means = train_df.groupby('district')['crop_yield'].mean().to_dict()
        self.season_means = train_df.groupby('season')['crop_yield'].mean().to_dict()
        
        train_df['Crop_Mean'] = train_df['crop'].map(self.crop_means).fillna(self.global_mean)
        train_df['District_Mean'] = train_df['district'].map(self.district_means).fillna(self.global_mean)
        train_df['Season_Mean'] = train_df['season'].map(self.season_means).fillna(self.global_mean)
        train_df['Crop_District_Mean'] = train_df['Crop_Mean'] * train_df['District_Mean']
        
        # 2. Fit preprocessors
        self.encoder.fit(train_df[self.categorical_cols])
        self.scaler.fit(train_df[self.numerical_cols])
        
        X_train_encoded = self._transform_features(X_train_raw)
        
        # 3. Train Neural Network Model (represented as lstm_model for shape compatibility)
        X_train_lstm = X_train_encoded[:, np.newaxis, :]
        input_dim = X_train_encoded.shape[1]
        self.lstm_model = build_lstm_model(input_dim)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.lstm_model.fit(
            X_train_lstm, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # 4. Train Random Forest Model (optimized hyperparameters)
        self.rf_model = RandomForestRegressor(max_depth=12, n_estimators=1000, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_train_encoded, y_train)
        
        # 5. Train XGBoost Model (optimized hyperparameters)
        self.xgb_model = xgb.XGBRegressor(max_depth=8, n_estimators=1000, random_state=42, n_jobs=-1, learning_rate=0.03)
        self.xgb_model.fit(X_train_encoded, y_train)
        
        # 6. Feature Fusion (stack predictions of base models)
        X_train_lstm_tensor = tf.cast(X_train_lstm, tf.float32)
        lstm_preds = self.lstm_model(X_train_lstm_tensor, training=False).numpy().reshape(-1, 1)
        rf_preds = self.rf_model.predict(X_train_encoded).reshape(-1, 1)
        xgb_preds = self.xgb_model.predict(X_train_encoded).reshape(-1, 1)
        
        # Combine LSTM prediction (1) + RF prediction (1) + XGBoost prediction (1) -> 3-D
        final_features = np.hstack([lstm_preds, rf_preds, xgb_preds])
        
        # 7. Train Meta-Regressor stacking layer using RidgeCV
        from sklearn.linear_model import RidgeCV
        self.meta_model = RidgeCV(alphas=[10.0, 50.0, 100.0, 200.0, 500.0])
        self.meta_model.fit(final_features, y_train)

    def predict(self, X_input):
        """
        Runs predictions end-to-end on raw data frames or preprocessed arrays.
        If X_input is a DataFrame, preprocesses it first.
        If X_input is a numpy array, assumes it's already preprocessed.
        Returns predicted yield in original scale (Tons/Hectare).
        """
        if isinstance(X_input, pd.DataFrame):
            X_encoded = self._transform_features(X_input)
        else:
            X_encoded = np.asarray(X_input, dtype=np.float64)

        X_lstm = X_encoded[:, np.newaxis, :]
        
        X_lstm_tensor = tf.cast(X_lstm, tf.float32)
        lstm_preds = self.lstm_model(X_lstm_tensor, training=False).numpy().reshape(-1, 1)
        rf_preds = self.rf_model.predict(X_encoded).reshape(-1, 1)
        xgb_preds = self.xgb_model.predict(X_encoded).reshape(-1, 1)
        
        final_features = np.hstack([lstm_preds, rf_preds, xgb_preds])
        log_pred = self.meta_model.predict(final_features)
        return np.expm1(log_pred)
