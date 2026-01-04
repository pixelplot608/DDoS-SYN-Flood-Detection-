import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("traffic_detection.log"), logging.StreamHandler()]
)
logger = logging.getLogger("traffic_detector")

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variable to optimize parallel processing
os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, os.cpu_count() - 1))


def load_data(train_path, test_path):
    """Load and prepare datasets with error handling"""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        logger.info(f"Loaded training data: {train_data.shape} and testing data: {test_data.shape}")
        logger.info(f"Training labels: {train_data['Label'].value_counts().to_dict()}")
        logger.info(f"Testing labels: {test_data['Label'].value_counts().to_dict()}")
        
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def preprocess_data(train_data, test_data):
    """Preprocess the data for training"""
    # Separate features and labels
    X_train = train_data.drop('Label', axis=1)
    y_train = train_data['Label']
    X_test = test_data.drop('Label', axis=1)
    y_test = test_data['Label']
    
    feature_names = X_train.columns.tolist()
    
    # Encode labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    logger.info("Label Encoding Mapping:")
    label_mapping = {label: i for i, label in enumerate(encoder.classes_)}
    logger.info(str(label_mapping))
    
    malicious_labels = [i for i, label in enumerate(encoder.classes_) if label.lower() == 'syn']
    logger.info(f"Malicious label indices: {malicious_labels}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, 
            X_train, X_test, feature_names, encoder, malicious_labels)


def feature_selection(X_train_scaled, y_train_encoded, X_test_scaled, feature_names):
    """Select most important features using Random Forest"""
    logger.info("Performing feature selection...")
    
    # Use Random Forest for feature selection
    selector = RandomForestClassifier(n_estimators=100, random_state=42)
    selector.fit(X_train_scaled, y_train_encoded)
    
    # Get feature importances and select top features
    importances = selector.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Log top 10 features
    logger.info("Top 10 features by importance:")
    for i in range(min(10, len(feature_names))):
        logger.info(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Select features using the model
    selection_model = SelectFromModel(selector, threshold="mean", prefit=True)
    X_train_selected = selection_model.transform(X_train_scaled)
    X_test_selected = selection_model.transform(X_test_scaled)
    
    # Get names of selected features
    selected_indices = selection_model.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    logger.info(f"Selected {len(selected_features)} out of {len(feature_names)} features")
    
    return X_train_selected, X_test_selected, selected_features


def build_advanced_models():
    """Build dictionary of advanced ML models"""
    # XGBoost classifier with good defaults
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        n_jobs=-1,
        random_state=42
    )
    
    # LightGBM classifier with good defaults
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        n_jobs=-1,
        random_state=42
    )
    
    # Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    # SVM with optimized settings
    svm_model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )
    
    # Create a voting ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model),
            ('svm', svm_model)
        ],
        voting='soft'
    )
    
    models = {
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
        "Random Forest": rf_model,
        "SVM": svm_model,
        "Ensemble": ensemble_model
    }
    
    return models


def optimize_hyperparameters(X_train, y_train, model_name, model):
    """Perform hyperparameter optimization for selected model"""
    logger.info(f"Optimizing hyperparameters for {model_name}...")
    
    # Define parameter grids for each model type
    param_grids = {
        "XGBoost": {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100],
            'subsample': [0.8, 1.0]
        },
        "LightGBM": {
            'num_leaves': [15, 31],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100],
            'subsample': [0.8, 1.0]
        },
        "Random Forest": {
            'max_depth': [5, 10, None],
            'n_estimators': [50, 100],
            'min_samples_split': [2, 5]
        },
        "SVM": {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1]
        }
    }
    
    # Skip ensemble and any model not in param_grids
    if model_name not in param_grids:
        logger.info(f"No hyperparameter optimization defined for {model_name}")
        return model
    
    # Configure and run grid search
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, encoder, model_name):
    """Evaluate model and return metrics"""
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities for ROC AUC (if the model supports it)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        roc_auc = roc_auc_score(y_test, y_prob)
    except (AttributeError, IndexError):
        roc_auc = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
    
    # Log results
    logger.info(f"\nResults for {model_name}:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    if roc_auc:
        logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Print classification report in a formatted way
    logger.info("Classification Report:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            logger.info(f"{label}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}")
    
    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": y_pred,
        "model": model
    }


def prevent_attack(X_test_original, y_pred, malicious_labels, feature_names, classifier_name, threshold=0.8):
    """Enhanced attack prevention mechanism with scoring"""
    # Identify source IP-related features more robustly
    source_feature_indices = []
    for i, name in enumerate(feature_names):
        lower_name = name.lower()
        if any(term in lower_name for term in ['ip', 'source', 'src', 'addr', 'port']):
            source_feature_indices.append(i)
    
    # If no source features found, use first few features
    if not source_feature_indices:
        source_feature_indices = list(range(min(5, len(feature_names))))
    
    # Create a blacklist with confidence scores
    blacklist = {}
    blocked_count = 0
    
    for i, prediction in enumerate(y_pred):
        if prediction in malicious_labels:
            # Extract source information
            source_info = tuple(X_test_original.iloc[i, source_feature_indices].values)
            
            # Increment count for this source
            if source_info in blacklist:
                blacklist[source_info] += 1
            else:
                blacklist[source_info] = 1
            
            blocked_count += 1
    
    # Calculate confidence scores (normalized by max count)
    max_count = max(blacklist.values()) if blacklist else 1
    blacklist_with_scores = {source: count/max_count for source, count in blacklist.items()}
    
    # Filter by threshold
    final_blacklist = {source: score for source, score in blacklist_with_scores.items() if score >= threshold}
    
    logger.info(f"Using {classifier_name}: Blocked {blocked_count}/{len(y_pred)} entries " 
                f"({(blocked_count / len(y_pred)) * 100:.2f}%)")
    logger.info(f"Final blacklist size: {len(final_blacklist)} sources with confidence ≥ {threshold}")
    
    return final_blacklist


def main():
    """Main execution function"""
    # Define file paths - use configuration instead of hardcoded paths
    config = {
        'train_path': r"C:\Users\main\OneDrive\Documents\16\program related stuff\modified\Syn-training-balanced.csv",
        'test_path': r"C:\Users\main\OneDrive\Documents\16\program related stuff\modified\Noisy_Syn_Testing.csv",
        'output_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
        'model_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    }
    
    # Create output directories if they don't exist
    for directory in [config['output_dir'], config['model_dir']]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Load and preprocess data
    logger.info("Starting cloud traffic detection system...")
    train_data, test_data = load_data(config['train_path'], config['test_path'])
    
    (X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, 
     X_train_orig, X_test_orig, feature_names, encoder, malicious_labels) = preprocess_data(train_data, test_data)
    
    # Feature selection
    X_train_selected, X_test_selected, selected_features = feature_selection(
        X_train_scaled, y_train_encoded, X_test_scaled, feature_names
    )
    
    # Build and train models
    models = build_advanced_models()
    
    # Results storage
    results = {}
    best_model = None
    best_accuracy = 0
    
    # Train, optimize and evaluate each model
    for name, model in models.items():
        logger.info(f"\nTraining and evaluating {name}...")
        
        # Only optimize individual models, not ensemble
        if name != "Ensemble":
            model = optimize_hyperparameters(X_train_selected, y_train_encoded, name, model)
        
        # Train the model
        model.fit(X_train_selected, y_train_encoded)
        
        # Evaluate
        result = evaluate_model(model, X_test_selected, y_test_encoded, encoder, name)
        results[name] = result
        
        # Save model
        model_path = os.path.join(config['model_dir'], f"{name.lower().replace(' ', '_')}_model.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Track best model
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model = name
    
    # Execute attack prevention
    logger.info("\nRunning attack prevention mechanism...")
    all_blacklists = {}
    for name, result in results.items():
        blacklist = prevent_attack(
            X_test_orig, result['predictions'], malicious_labels, feature_names, name
        )
        all_blacklists[name] = blacklist
    
    # Create consensus blacklist
    if len(models) > 1:
        logger.info("\nGenerating consensus blacklist...")
        
        # Count sources flagged by different models
        source_counts = {}
        for name, blacklist in all_blacklists.items():
            for source, score in blacklist.items():
                if source not in source_counts:
                    source_counts[source] = {'count': 0, 'total_score': 0}
                source_counts[source]['count'] += 1
                source_counts[source]['total_score'] += score
        
        # Create consensus with average scores
        min_classifiers = max(1, len(models) // 2)  # At least half of classifiers must agree
        consensus_blacklist = {
            source: data['total_score'] / data['count'] 
            for source, data in source_counts.items() 
            if data['count'] >= min_classifiers
        }
        
        logger.info(f"Consensus Blacklist: {len(consensus_blacklist)} unique sources flagged by {min_classifiers}+ classifiers")
        
        # Save blacklist to file
        blacklist_file = os.path.join(config['output_dir'], "consensus_blacklist.txt")
        with open(blacklist_file, 'w') as f:
            for source, score in consensus_blacklist.items():
                f.write(f"{source}: {score:.4f}\n")
        logger.info(f"Consensus blacklist saved to {blacklist_file}")
    
    # Summarize best model
    logger.info(f"\nBest performing model: {best_model} with accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()