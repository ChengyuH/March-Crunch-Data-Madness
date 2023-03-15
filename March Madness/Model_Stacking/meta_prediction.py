import numpy as np

def predict_meta_model(X_test, meta_model, models):
    # Generate meta features for test data
    meta_features_test = []
    for model in models:
        meta_feature = model.predict(X_test)
        meta_feature = meta_feature.reshape(-1, 1) # reshape to 2D array with a single column
        meta_features_test.append(meta_feature)
    meta_features_test = np.concatenate(meta_features_test, axis=1)

    # Use meta-model to make predictions
    meta_predictions = meta_model.predict_proba(meta_features_test)[:, 1]

    return meta_predictions