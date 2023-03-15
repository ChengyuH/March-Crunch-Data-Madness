from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def random_forest(X_train, y_train):
    maximum_depth = 100
    minimum_depth = 1
    param_grid = {'max_depth': list(range(minimum_depth, maximum_depth + 1))}

    # The reason we choose RandomizedSearchCV instead of GridSearchCV because it's faster
    # The Random Forest Model
    gs_rf = GridSearchCV(
        RandomForestClassifier(criterion='entropy', random_state=1),
        param_grid, cv=3, n_jobs=4)

    gs_rf.fit(X_train, y_train)

    return  gs_rf
