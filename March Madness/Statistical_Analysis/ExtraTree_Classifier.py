from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def select_top_features(X, y, n_top=12):
    # 对分类变量进行编码
    categorical_cols = X.select_dtypes(include='object').columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # 使用ExtraTreesClassifier选择原始特征中最好的 k 个特征
    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(X, y)
    importances = model.feature_importances_
    top_features = importances.argsort()[::-1][:n_top]

    # 对 k 个特征进行 PCA 降维
    pca = PCA(n_components=n_top)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.iloc[:, top_features])
    X_pca = pca.fit_transform(X_scaled)

    # 获取每个主成分对应的原始特征
    feature_dict = {}
    for i in range(n_top):
        feature = X.columns[top_features[pca.components_[i].argmax()]]
        if feature not in feature_dict:
            feature_dict[feature] = i

    # 只保留每个原始特征的最大主成分所对应的特征
    features = []
    for feature, index in feature_dict.items():
        features.append(X.columns[top_features[pca.components_[index].argmax()]])

    return features
