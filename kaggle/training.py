import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, train_test_split
import cupy as cp
import optuna
from itertools import combinations
from functools import partial
import lightgbm as lgb
from catboost import CatBoostClassifier

def reduce_correlated_features(X, threshold=0.95):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropping {len(to_drop)} correlated features")
    return X.drop(columns=to_drop)

def engineer_features(X_train, X_test):
    """
    Perform feature engineering on both X_train and X_test to ensure they have identical features.
    """
    print("\nPerforming feature engineering...")

    X_train_new = X_train.copy()
    X_test_new = X_test.copy()

    # Ensure both datasets have the same columns initially
    missing_cols_in_test = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols_in_test:
        X_test_new[col] = 0  # Fill missing columns with 0 in X_test
    
    # Identify potential categorical features
    categorical_features = [col for col in X_train.columns if X_train[col].nunique() < 20]
    print(f"Identified {len(categorical_features)} potential categorical features")

    # Determine top features based on target (if available) or variance
    if 'target' in X_train.columns:
        temp_features = X_train.drop(columns=['target'])
        temp_target = X_train['target']
        dtrain = xgb.DMatrix(temp_features, label=temp_target)
        params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 42}
        simple_model = xgb.train(params, dtrain, num_boost_round=100)
        importance_scores = simple_model.get_score(importance_type='gain')
        importances = pd.DataFrame({
            'Feature': list(importance_scores.keys()),
            'Importance': list(importance_scores.values())
        }).sort_values('Importance', ascending=False)
        top_features = importances['Feature'].tolist()[:10]
    else:
        top_features = X_train.var().sort_values(ascending=False).head(10).index.tolist()

    # Ensure all top features exist in both datasets
    for feat in top_features:
        if feat not in X_test_new:
            X_test_new[feat] = 0  # Fill missing features with 0
    
    # Create pairwise interactions between top features
    for feat_i, feat_j in combinations(top_features, 2):
        X_train_new[f"{feat_i}_x_{feat_j}"] = X_train[feat_i] * X_train[feat_j]
        X_test_new[f"{feat_i}_x_{feat_j}"] = X_test_new[feat_i] * X_test_new[feat_j]

        X_train_new[f"{feat_i}_div_{feat_j}"] = X_train[feat_i] / (X_train[feat_j] + 1e-5)
        X_test_new[f"{feat_i}_div_{feat_j}"] = X_test_new[feat_i] / (X_test_new[feat_j] + 1e-5)

        X_train_new[f"{feat_i}_plus_{feat_j}"] = X_train[feat_i] + X_train[feat_j]
        X_test_new[f"{feat_i}_plus_{feat_j}"] = X_test_new[feat_i] + X_test_new[feat_j]

        X_train_new[f"{feat_i}_minus_{feat_j}"] = X_train[feat_i] - X_train[feat_j]
        X_test_new[f"{feat_i}_minus_{feat_j}"] = X_test_new[feat_i] - X_test_new[feat_j]

    # Create polynomial features for top 5 features
    for feat in top_features[:5]:
        X_train_new[f"{feat}_squared"] = X_train[feat] ** 2
        X_test_new[f"{feat}_squared"] = X_test_new[feat] ** 2

        X_train_new[f"{feat}_cubed"] = X_train[feat] ** 3
        X_test_new[f"{feat}_cubed"] = X_test_new[feat] ** 3

        if (X_train[feat] >= 0).all():
            X_train_new[f"sqrt_{feat}"] = np.sqrt(X_train[feat])
            X_test_new[f"sqrt_{feat}"] = np.sqrt(X_test_new[feat])

    # Log transform for highly skewed features
    skewed_features = [col for col in X_train.columns if X_train[col].skew() > 3]
    for feat in skewed_features[:5]:
        min_val = min(X_train[feat].min(), X_test[feat].min())
        shift = abs(min_val) + 1e-5 if min_val <= 0 else 0

        X_train_new[f"log_{feat}"] = np.log(X_train[feat] + shift)
        X_test_new[f"log_{feat}"] = np.log(X_test_new[feat] + shift)

    print(f"Original features: {X_train.shape[1]}, New features: {X_train_new.shape[1]}")
    print("Data has been engineered\n")
    return X_train_new, X_test_new, categorical_features

def load():
    # load testing and training data
    column_names = ['Id', 'Y'] + [f'f{i}' for i in range(1, 39)]
    train = pd.read_csv('training_data.csv', header=0, names=column_names)
    y = train['Y']
    X = train.drop(columns=['Id', 'Y'])

    column_names = ['Id'] + [f'f{i}' for i in range(1, 39)]
    test = pd.read_csv('testing_data.csv', header=0, names=column_names)
    test_ids = test['Id']
    X_t = test.drop(columns=['Id'])
    print("Data has been loaded\n")
    return X, X_t, y, test_ids

def objective(trial, X, y, cat):
    # Define the hyperparameters to tune
    # param = {
    #     'n_estimators': trial.suggest_int('n_estimators', 50, 300),  # Use suggest_int instead of randint
    #     'max_depth': trial.suggest_int('max_depth', 3, 15),  # Use suggest_int instead of randint
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # Use suggest_float instead of float
    #     'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    #     'gamma': trial.suggest_float('gamma', 0, 0.3),
    #     'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
    #     'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
    #     'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    #     'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10.38)
    # }
    
    #lgb
    # param = {
    # 'n_estimators': trial.suggest_int('n_estimators', 50, 300),
    # 'max_depth': trial.suggest_int('max_depth', 3, 15),
    # 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
    # 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),  # Avoid extreme subsampling
    # 'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),  # Ensure enough features
    # 'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),  # Reduce restrictions on splits
    # 'min_child_samples': trial.suggest_int('min_child_samples', 2, 10),  # Allow more leaf nodes
    # 'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),  # Reduce L1 regularization
    # 'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),  # Reduce L2 regularization
    # 'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10.0)  # Handle class imbalance
    # }

    #cat
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),  # Regularization
        'border_count': trial.suggest_int('border_count', 32, 255),  # Feature binning
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),  # Controls subsampling
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),  # Helps with overfitting
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),  # For class imbalance
    }

    # Create the model with the suggested hyperparameters
    # model = XGBClassifier(
    #     **param,
    #     objective='binary:logistic',
    #     eval_metric='auc',
    #     random_state=42,
    #     enable_categorical=True
    # )

    #lgb
    # model = lgb.LGBMClassifier(
    # **param,
    # random_state=42,
    # metric='auc',
    # objective='binary',
    # verbosity = 0
    # )

    model = CatBoostClassifier(
        **param,
        random_state=42,
        eval_metric='AUC',
        loss_function='Logloss',
        cat_features=cat,
        verbose=0
    )

    # Perform cross-validation (e.g., 8-fold CV)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)

    # Return the mean of the ROC AUC scores from the cross-validation
    return np.mean(cv_scores)

def main():
    # prompt
    print("TRAINING TIME")

    # load and preprocess data
    X, X_t, y, test_ids = load()
    X_new, X_tnew, cats = engineer_features(X, X_t)

    # optimize first xgboost
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(partial(objective, X=X_new, y=y, cat=cats), n_trials=100, n_jobs=-1)
    print("Parameters have been tuned for first xbg! \n")

    # best parameters and the best score
    print("Best Hyperparameters:", study.best_params)
    print(f"Best Cross-Validation AUC: {study.best_value:.4f}")

    with open('optuna_output.txt', 'w') as f:
        f.write("Optuna Optimization Results for xgb1\n")
        f.write("---------------------------\n")
        f.write(f"Best Hyperparameters: {study.best_params}\n")
        f.write(f"Best Cross-Validation AUC: {study.best_value:.4f}\n")

    ##################################################
    for col in cats:
        if col in X_new.columns:
            X_new[col] = X_new[col].astype('category')
        if col in X_tnew.columns:
            X_tnew[col] = X_tnew[col].astype('category')
    model = XGBClassifier(
        **study.best_params,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        enable_categorical=True
    )
    
    X_train = X_new
    y_train = y
    model.fit(X_train,y_train)


    importance_scores = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
    'Feature': list(importance_scores.keys()),
    'Importance': list(importance_scores.values())
    }).sort_values('Importance', ascending=False)
    #print("Top features by importance:")
    #print(importance_df.head(100))

    top50_features = importance_df['Feature'].head(50).tolist()
    X_train_top50 = X_train[top50_features]

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=30)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(partial(objective, X=X_train_top50, y=y), n_trials=100, n_jobs=-1)
    print("Parameters have been tuned for second xbg! \n")
    print("Best Hyperparameters:", study.best_params)
    print(f"Best Cross-Validation AUC: {study.best_value:.4f}")

    with open('optuna_output.txt', 'a') as f:
        f.write("Optuna Optimization Results for xgb2\n")
        f.write("---------------------------\n")
        f.write(f"Best Hyperparameters: {study.best_params}\n")
        f.write(f"Best Cross-Validation AUC: {study.best_value:.4f}\n")

    model2 = XGBClassifier(
        **study.best_params,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        enable_categorical=True
    )
    model2.fit(X_train_top50, y_train)

    predOut = model2.predict_proba(X_tnew[X_train_top50.columns])
    results = pd.DataFrame({
    'Id': test_ids,
    'Y': predOut[:, 1]
    })

    # Save the results to Google Drive
    results.to_csv('sub.csv', index=False)
    print("SUCCESS! \n")

    ##################################################

if __name__ == "__main__":
    main()
    
    # old parameters: {'n_estimators': 276, 'max_depth': 17, 'learning_rate': 0.025473270733730957, 'subsample': 0.9154136056690677, 'colsample_bytree': 0.5086224425714986, 'gamma': 0.07364474321239584, 'min_child_weight': 1, 'reg_alpha': 0.12583538105891215, 'reg_lambda': 0.2091061358983765, 'scale_pos_weight': 2.1833862588473414}
    # new parameters for xgb 1 (trial 447): parameters: {'n_estimators': 275, 'max_depth': 15, 'learning_rate': 0.036778436187159785, 'subsample': 0.9446728494447203, 'colsample_bytree': 0.5005311507529506, 'gamma': 0.24529159637562972, 'min_child_weight': 2, 'reg_alpha': 0.4064661143253671, 'reg_lambda': 2.159032799096768, 'scale_pos_weight': 3.650412987277388}.
    # new parameters for xgb 2 (trial 471): parameters: {'n_estimators': 238, 'max_depth': 15, 'learning_rate': 0.04797228141757646, 'subsample': 0.7155438787616168, 'colsample_bytree': 0.5269440962507546, 'gamma': 0.11834748451214913, 'min_child_weight': 1, 'reg_alpha': 0.009181517172902736, 'reg_lambda': 1.6872456066629262, 'scale_pos_weight': 1.7308351937873647}
