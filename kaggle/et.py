# XGBoost Implementation for Kaggle Binary Classification Competition
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_data():
    """
    Load training and testing data
    """
    train_data = pd.read_csv('training_data.csv')
    test_data = pd.read_csv('testing_data.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    
    return train_data, test_data

def engineer_features(X_train, X_test):
    """
    Perform feature engineering
    """
    print("\nPerforming feature engineering...")
    
    # Copy the original dataframes
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    
    # 1. Identify potential categorical features
    categorical_features = []
    for col in X_train.columns:
        if X_train[col].nunique() < 20:
            categorical_features.append(col)
    
    print(f"Identified {len(categorical_features)} potential categorical features")
    
    # 2. Train a simple XGBoost to get feature importances
    # Create a copy for XGBoost training to avoid modifying the original
    temp_features = X_train.copy()
    temp_target = temp_features.pop('target') if 'target' in temp_features else np.zeros(len(temp_features))
    dtrain = xgb.DMatrix(temp_features, label=temp_target)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': RANDOM_STATE
    }
    simple_model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Get feature importances
    importance_scores = simple_model.get_score(importance_type='gain')
    # Convert to dataframe for easier handling
    importances = pd.DataFrame({
        'Feature': list(importance_scores.keys()),
        'Importance': list(importance_scores.values())
    }).sort_values('Importance', ascending=False)
    
    # 3. Create pairwise interactions between top important features
    top_features = importances['Feature'].tolist()[:10]  # Top 10 features
    
    interaction_count = 0
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            feat_i = top_features[i]
            feat_j = top_features[j]
            
            # Multiplication interaction
            interaction_name = f"{feat_i}_x_{feat_j}"
            X_train_new[interaction_name] = X_train[feat_i] * X_train[feat_j]
            X_test_new[interaction_name] = X_test[feat_i] * X_test[feat_j]
            interaction_count += 1
            
            # Division interaction (with safety check)
            if (X_train[feat_j] != 0).all() and (X_test[feat_j] != 0).all():
                interaction_name = f"{feat_i}_div_{feat_j}"
                X_train_new[interaction_name] = X_train[feat_i] / (X_train[feat_j] + 1e-5)
                X_test_new[interaction_name] = X_test[feat_i] / (X_test[feat_j] + 1e-5)
                interaction_count += 1
            
            # Sum interaction
            interaction_name = f"{feat_i}_plus_{feat_j}"
            X_train_new[interaction_name] = X_train[feat_i] + X_train[feat_j]
            X_test_new[interaction_name] = X_test[feat_i] + X_test[feat_j]
            interaction_count += 1
            
            # Difference interaction
            interaction_name = f"{feat_i}_minus_{feat_j}"
            X_train_new[interaction_name] = X_train[feat_i] - X_train[feat_j]
            X_test_new[interaction_name] = X_test[feat_i] - X_test[feat_j]
            interaction_count += 1
    
    # 4. Create polynomial features for top 5 features
    for feat in top_features[:5]:
        # Square
        X_train_new[f"{feat}_squared"] = X_train[feat] ** 2
        X_test_new[f"{feat}_squared"] = X_test[feat] ** 2
        
        # Cube
        X_train_new[f"{feat}_cubed"] = X_train[feat] ** 3
        X_test_new[f"{feat}_cubed"] = X_test[feat] ** 3
        
        # Square root (with safety check)
        if (X_train[feat] >= 0).all() and (X_test[feat] >= 0).all():
            X_train_new[f"sqrt_{feat}"] = np.sqrt(X_train[feat])
            X_test_new[f"sqrt_{feat}"] = np.sqrt(X_test[feat])
    
    # 5. Log transform for highly skewed features (optional)
    # Identify skewed features
    skewed_features = []
    for col in X_train.columns:
        if X_train[col].skew() > 3:  # Highly skewed
            skewed_features.append(col)
    
    for feat in skewed_features[:5]:  # Limit to top 5 most skewed
        # Make sure all values are positive by adding minimum + small constant
        min_val = min(X_train[feat].min(), X_test[feat].min())
        if min_val <= 0:
            shift = abs(min_val) + 1e-5
        else:
            shift = 0
            
        X_train_new[f"log_{feat}"] = np.log(X_train[feat] + shift)
        X_test_new[f"log_{feat}"] = np.log(X_test[feat] + shift)
    
    print(f"Created {interaction_count} interaction features")
    print(f"Original features: {X_train.shape[1]}, New features: {X_train_new.shape[1]}")
    
    return X_train_new, X_test_new, categorical_features

def cross_validate_xgb(X_train, y_train, params=None):
    """
    Perform cross-validation for XGBoost
    """
    print("\nPerforming cross-validation for XGBoost...")
    
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),  # Handle imbalance
            'seed': RANDOM_STATE
        }
    
    # Create cross-validation folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Store predictions and scores
    oof_predictions = np.zeros(len(X_train))
    fold_scores = []
    
    # Loop through folds
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        print(f"\nFold {fold+1}/5")
        
        # Split data for this fold
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
        dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Make predictions for this fold's validation set
        fold_preds = model.predict(dval)
        oof_predictions[val_idx] = fold_preds
        
        # Calculate AUC for this fold
        fold_auc = roc_auc_score(y_fold_val, fold_preds)
        fold_scores.append(fold_auc)
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")
    
    # Calculate overall AUC
    overall_auc = roc_auc_score(y_train, oof_predictions)
    print(f"\nCross-validation AUC scores: {fold_scores}")
    print(f"Mean CV AUC: {np.mean(fold_scores):.4f} (Â±{np.std(fold_scores):.4f})")
    print(f"Overall CV AUC: {overall_auc:.4f}")
    
    return overall_auc, oof_predictions

def tune_xgb_hyperparameters(X_train, y_train):
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV
    """
    print("\nTuning XGBoost hyperparameters...")
    start_time = time()
    
    # Parameter grid for RandomizedSearchCV
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [4, 5, 6, 7, 8],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [0, 0.1, 1, 10],
        'scale_pos_weight': [(y_train == 0).sum() / (y_train == 1).sum()]
    }
    
    # Initialize XGBClassifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,  # Will be increased later
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='auc'
    )
    
    # Initialize RandomizedSearchCV
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=15,  # Try 15 combinations
        scoring='roc_auc',
        cv=cv,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=-1
    )
    
    # Fit RandomizedSearchCV
    random_search.fit(X_train, y_train)
    
    # Print results
    print(f"Best parameters found: {random_search.best_params_}")
    print(f"Best AUC score: {random_search.best_score_:.4f}")
    print(f"Hyperparameter tuning completed in {(time() - start_time)/60:.2f} minutes")
    
    # Convert best parameters to XGBoost param format
    best_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': random_search.best_params_['learning_rate'],
        'max_depth': random_search.best_params_['max_depth'],
        'min_child_weight': random_search.best_params_['min_child_weight'],
        'subsample': random_search.best_params_['subsample'],
        'colsample_bytree': random_search.best_params_['colsample_bytree'],
        'gamma': random_search.best_params_['gamma'],
        'alpha': random_search.best_params_['reg_alpha'],
        'lambda': random_search.best_params_['reg_lambda'],
        'scale_pos_weight': random_search.best_params_['scale_pos_weight'],
        'seed': RANDOM_STATE
    }
    
    return best_params

def train_final_xgb(X_train, y_train, X_test, params=None):
    """
    Train final XGBoost model with all data
    """
    print("\nTraining final XGBoost model...")
    
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),  # Handle imbalance
            'seed': RANDOM_STATE
        }
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,  # We'll use early stopping
        verbose_eval=100
    )
    
    # Generate predictions
    test_preds = model.predict(dtest)
    
    # Feature importance
    try:
        importance_scores = model.get_score(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': list(importance_scores.keys()),
            'Importance': list(importance_scores.values())
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 important features:")
        print(importance_df.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('xgb_feature_importance.png')
        print("Feature importance plot saved as 'xgb_feature_importance.png'")
    except:
        print("Could not generate feature importance plot - some features may not have been used")
    
    return model, test_preds

def main():
    """
    Main execution function
    """
    print("Starting XGBoost implementation for high Kaggle score...")
    start_time = time()
    
    # Load data
    train_data, test_data = load_data()
    
    # Split features and target
    X_train = train_data.drop(['Id', 'Y'], axis=1)
    y_train = train_data['Y']
    X_test = test_data.drop(['Id'], axis=1)
    test_ids = test_data['Id']
    
    # Check class imbalance
    class_counts = y_train.value_counts()
    print(f"\nClass distribution:")
    print(class_counts)
    print(f"Class imbalance ratio: 1:{class_counts[1]/class_counts[0]:.2f}")
    
    # Add target to X_train for feature engineering (as a separate copy)
    X_train_with_target = X_train.copy()
    X_train_with_target['target'] = y_train
    
    # Step 1: Engineer features
    X_train_engineered, X_test_engineered, categorical_features = engineer_features(X_train_with_target, X_test)
    
    # Remove target from engineered features if it exists
    y_train_engineered = X_train_engineered.pop('target') if 'target' in X_train_engineered else y_train
    
    # Step 2: Tune XGBoost hyperparameters
    best_params = tune_xgb_hyperparameters(X_train_engineered, y_train_engineered)
    
    # Step 3: Perform cross-validation with tuned parameters
    cv_auc, oof_preds = cross_validate_xgb(X_train_engineered, y_train_engineered, best_params)
    
    # Step 4: Train final model and predict
    final_model, test_preds = train_final_xgb(X_train_engineered, y_train_engineered, X_test_engineered, best_params)
    
    # Create submission file
    submission = pd.DataFrame({
        'Id': test_ids,
        'Y': test_preds  # Using probabilities (soft predictions), not class labels
    })
    
    # Save submission
    submission_file = 'xgb_submission.csv'
    submission.to_csv(submission_file, index=False)
    print(f"Submission file created: {submission_file}")
    print(f"Sample probability predictions: {submission['Y'].head().tolist()}")
    
    # Plot histogram of predictions to check distribution
    plt.figure(figsize=(10, 6))
    plt.hist(test_preds, bins=50)
    plt.title('Distribution of Predicted Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.savefig('xgb_prediction_distribution.png')
    print("Created prediction distribution plot: 'xgb_prediction_distribution.png'")
    
    # Save model for future use
    joblib.dump(final_model, 'xgb_model.pkl')
    print("Model saved as 'xgb_model.pkl'")
    
    # Print total runtime
    total_runtime = (time() - start_time) / 60
    print(f"\nTotal runtime: {total_runtime:.2f} minutes")
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()