import xgboost as xgb
import numpy as np

def train_meta_model(X_train, y_train, models):
    # Initialize arrays to hold meta features and labels
    meta_features_train = []

    # Generate meta features for each model
    trained_models = []
    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        meta_feature = model.predict(X_train)
        meta_feature = meta_feature.reshape(-1, 1) # reshape to 2D array with a single column
        meta_features_train.append(meta_feature)
        trained_models.append(model)

    # Concatenate meta features into a single array
    meta_features_train = np.concatenate(meta_features_train, axis=1)

    # Define XGBoost model and parameters
    meta_model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        objective='binary:logistic'
    )

    # Train XGBoost model on meta-features
    meta_model.fit(meta_features_train, y_train)

    return meta_model


