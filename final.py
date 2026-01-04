import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set environment variable to silence joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Adjust to match CPU core count

# Load datasets
train_data = pd.read_csv("Syn-training-balanced.csv")
test_data = pd.read_csv("Syn-testing-hard.csv")

#train_data = pd.read_csv(r"C:\Users\main\Downloads\Hard-to-Detect-Syn.csv")

# Analyze dataset
print("Unique labels in training data:", train_data['Label'].unique())
print("Unique labels in testing data:", test_data['Label'].unique())

print("\nTraining data counts:")
print(train_data['Label'].value_counts())
print("\nTesting data counts:")
print(test_data['Label'].value_counts())

# Separate features and labels
X_train = train_data.drop('Label', axis=1)
y_train = train_data['Label']
X_test = test_data.drop('Label', axis=1)
y_test = test_data['Label']

feature_names = X_train.columns.tolist()
print(f"First few feature names: {feature_names[:5]}")

# Encode labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

print("\nLabel Encoding Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{label} -> {i}")

malicious_labels = [i for i, label in enumerate(encoder.classes_) if label.lower() == 'syn']
print(f"Malicious label indices: {malicious_labels}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_test_original = X_test.copy()

# Classifiers
target_classifiers = {
    "LS-SVM": SVC(kernel='linear'),
    "Naive Bayes": GaussianNB(),
    "K-Nearest": KNeighborsClassifier(n_neighbors=5),
    "Multilayer Perceptron": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
}

# Train and evaluate
results = {}
for name, clf in target_classifiers.items():
    print(f"\nTraining {name} classifier...")
    clf.fit(X_train_scaled, y_train_encoded)
    y_pred = clf.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    cm = confusion_matrix(y_test_encoded, y_pred)
    report = classification_report(y_test_encoded, y_pred, target_names=encoder.classes_)
    
    results[name] = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": y_pred
    }

# Display results
print("\nClassifier Performance:")
for name, result in results.items():
    print(f"\nClassifier: {name}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Classification Report:")
    print(result['classification_report'])

# Attack prevention mechanism
def prevent_attack(X_test_original, y_pred, malicious_labels, feature_names, classifier_name):
    blacklist = set()
    source_feature_indices = [i for i, name in enumerate(feature_names) if any(term in name.lower() for term in ['ip', 'source', 'src', 'addr', 'port'])]
    
    if not source_feature_indices:
        source_feature_indices = list(range(min(3, len(feature_names))))
    
    blocked_count = 0
    for i, prediction in enumerate(y_pred):
        if prediction in malicious_labels:
            source_info = tuple(X_test_original.iloc[i, source_feature_indices].values)
            blacklist.add(source_info)
            blocked_count += 1
    
    print(f"Using {classifier_name}: Blocked {blocked_count}/{len(y_pred)} entries ({(blocked_count / len(y_pred)) * 100:.2f}%)")
    return blacklist

print("\nPrevention Mechanism Results:")
all_blacklists = {}
for name, result in results.items():
    blacklist = prevent_attack(X_test_original, result['predictions'], malicious_labels, feature_names, name)
    all_blacklists[name] = blacklist

# Consensus blacklist
if len(target_classifiers) > 1:
    blacklist_sets = list(all_blacklists.values())
    min_classifiers = max(1, len(target_classifiers) // 2)
    
    source_counts = {}
    for blacklist in blacklist_sets:
        for source in blacklist:
            source_counts[source] = source_counts.get(source, 0) + 1
    
    consensus_blacklist = {source for source, count in source_counts.items() if count >= min_classifiers}
    print(f"\nConsensus Blacklist: {len(consensus_blacklist)} unique sources flagged by {min_classifiers}+ classifiers")

# Cross-validation
print("\nCross-Validation Results:")
for name, clf in target_classifiers.items():
    scores = cross_val_score(clf, X_train_scaled, y_train_encoded, cv=3, scoring='accuracy', n_jobs=-1)
    print(f"{name} cross-validation accuracy: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
