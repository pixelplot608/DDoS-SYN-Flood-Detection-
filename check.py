import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# load datasets
train_data = pd.read_csv(r'C:\Users\main\OneDrive\Documents\ddos cloud flood\datset\modified\Syn-training-balanced.csv')
test_data = pd.read_csv(r'C:\Users\main\OneDrive\Documents\ddos cloud flood\datset\modified\Syn-testing.csv')

# Print the unique values in the Label column to understand what labels we have
print("Unique labels in training data:", train_data['Label'].unique())
print("Unique labels in testing data:", test_data['Label'].unique())

# separate features and labels
X_train = train_data.drop('Label', axis=1)
y_train = train_data['Label']

X_test = test_data.drop('Label', axis=1)
y_test = test_data['Label']

# Store original feature names for later reference
feature_names = X_train.columns.tolist()
print(f"First few feature names: {feature_names[:5]}")

# Encode labels
encoder = LabelEncoder()
encoder.fit(pd.concat([y_train, y_test]).unique())
y_train_encoded = encoder.transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Print encoded label mapping for clarity
print("\nLabel Encoding Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{label} -> {i}")

# Identify malicious label indices (assuming 'SYN' and 'Syn' are malicious)
malicious_labels = [i for i, label in enumerate(encoder.classes_) if label.lower() == 'syn']
print(f"Malicious label indices: {malicious_labels}")

# standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Keeping a copy of the original test data for reference in prevention mechanism
X_test_original = X_test.copy()

# initialize classifiers
classifiers = {
    "LS-SVM": SVC(kernel='linear'),
    "Naive Bayes": GaussianNB(),
    "K-Nearest": KNeighborsClassifier(n_neighbors=5),
    "Multilayer Perceptron": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

# train and evaluate classifiers
results = {}
for name, clf in classifiers.items():
    print(f"\nTraining {name} classifier...")
    clf.fit(X_train_scaled, y_train_encoded)
    y_pred = clf.predict(X_test_scaled)

    # evaluate performance
    accuracy = accuracy_score(y_test_encoded, y_pred)
    cm = confusion_matrix(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred, target_names=encoder.classes_)

    results[name] = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": y_pred
    }

# print results
print("\nClassifier Performance:")
for name, result in results.items():
    print(f"\nClassifier: {name}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Classification Report:")
    print(result['classification_report'])

# Improved prevention mechanism
def prevent_attack(X_test_original, y_pred, malicious_labels, feature_names, classifier_name):
    """
    Identify and block potentially malicious traffic based on classifier predictions
    
    Args:
        X_test_original: Original test features (not scaled)
        y_pred: Predicted labels
        malicious_labels: List of label indices considered malicious
        feature_names: Names of features in the dataset
        classifier_name: Name of the classifier used
    
    Returns:
        blacklist: Set of tuples containing source information of blocked traffic
    """
    blacklist = set()
    
    # Select relevant source identification features (assuming first few features identify the source)
    source_feature_indices = [i for i, name in enumerate(feature_names) 
                             if any(id_term in name.lower() for id_term in ['ip', 'source', 'src', 'addr', 'port'])]
    
    # If no source features identified, use first 3 features as fallback
    if not source_feature_indices:
        source_feature_indices = list(range(min(3, len(feature_names))))
        print(f"No explicit source features found. Using first {len(source_feature_indices)} features as source identifiers.")
    else:
        print(f"Using {len(source_feature_indices)} source identification features: {[feature_names[i] for i in source_feature_indices]}")
    
    # Track statistics
    total_traffic = len(y_pred)
    blocked_count = 0
    
    # Process each traffic entry
    for i, prediction in enumerate(y_pred):
        if prediction in malicious_labels:  # detected as malicious
            # Create a tuple of source identifying features
            source_info = tuple(X_test_original.iloc[i, source_feature_indices].values)
            blacklist.add(source_info)
            blocked_count += 1
    
    block_rate = (blocked_count / total_traffic) * 100 if total_traffic > 0 else 0
    print(f"Using {classifier_name}:")
    print(f"  - Total traffic entries: {total_traffic}")
    print(f"  - Blocked entries: {blocked_count} ({block_rate:.2f}%)")
    
    # Most frequently blocked sources (top 5)
    if blocked_count > 0:
        print("  - Sample of blocked sources (first 5):")
        for i, source in enumerate(list(blacklist)[:5]):
            print(f"    Source {i+1}: {source}")
    
    return blacklist

print("\nPrevention Mechanism Results:")
all_blacklists = {}
for name, result in results.items():
    blacklist = prevent_attack(
        X_test_original, 
        result['predictions'], 
        malicious_labels, 
        feature_names,
        name
    )
    all_blacklists[name] = blacklist

# Calculate consensus blacklist (sources flagged by majority of classifiers)
if len(classifiers) > 1:
    # Convert each blacklist to a set of sources
    blacklist_sets = list(all_blacklists.values())
    
    # Find sources flagged by at least half of the classifiers
    min_classifiers = max(1, len(classifiers) // 2)
    
    # Count occurrences of each source across all blacklists
    source_counts = {}
    for blacklist in blacklist_sets:
        for source in blacklist:
            source_counts[source] = source_counts.get(source, 0) + 1
    
    # Create consensus blacklist
    consensus_blacklist = {source for source, count in source_counts.items() 
                          if count >= min_classifiers}
    
    print(f"\nConsensus Blacklist (sources flagged by {min_classifiers}+ classifiers):")
    print(f"Total unique blocked sources: {len(consensus_blacklist)}")
    
    # Display some statistics on agreement among classifiers
    agreement_counts = {count: 0 for count in range(1, len(classifiers) + 1)}
    for count in source_counts.values():
        agreement_counts[count] += 1
    
    print("Classifier agreement statistics:")
    for count, num_sources in agreement_counts.items():
        if num_sources > 0:
            print(f"  Sources flagged by exactly {count} classifier(s): {num_sources}")

# cross-validation
print("\nCross-Validation Results:")
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_train_scaled, y_train_encoded, cv=5, scoring='accuracy')
    print(f"{name} cross-validation accuracy: {np.mean(scores):.4f} (±{np.std(scores):.4f})")

# Implementation of a simple firewall rule generation function
def generate_firewall_rules(blacklist, feature_names, source_feature_indices):
    """Generate sample firewall rules based on the blacklisted sources"""
    print("\nSample Firewall Rules (conceptual):")
    for i, source in enumerate(list(blacklist)[:10]):  # Show first 10 rules
        rule = f"RULE {i+1}: BLOCK traffic FROM "
        for j, feature_idx in enumerate(source_feature_indices):
            feature_value = source[j]
            feature_name = feature_names[feature_idx]
            rule += f"{feature_name}={feature_value}"
            if j < len(source_feature_indices) - 1:
                rule += " AND "
        print(rule)
    
    if len(blacklist) > 10:
        print(f"... and {len(blacklist) - 10} more rules")

# Generate sample firewall rules for the best performing classifier's blacklist
if all_blacklists:
    best_classifier = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nGenerating firewall rules based on {best_classifier} classifier:")
    
    # Determine source feature indices (assuming same logic as in prevent_attack)
    source_feature_indices = [i for i, name in enumerate(feature_names) 
                             if any(id_term in name.lower() for id_term in ['ip', 'source', 'src', 'addr', 'port'])]
    if not source_feature_indices:
        source_feature_indices = list(range(min(3, len(feature_names))))
    
    generate_firewall_rules(all_blacklists[best_classifier], feature_names, source_feature_indices)