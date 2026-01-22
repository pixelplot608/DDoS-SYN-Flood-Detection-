import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load datasets
train_data = pd.read_csv("Syn-training-balanced.csv")
test_data = pd.read_csv("Syn-testing-hard.csv")

# Dataset info
print("Training labels:\n", train_data['Label'].value_counts())
print("Testing labels:\n", test_data['Label'].value_counts())

# Split features & labels
X_train = train_data.drop('Label', axis=1)
y_train = train_data['Label']
X_test = test_data.drop('Label', axis=1)
y_test = test_data['Label']

feature_names = X_train.columns.tolist()

# Encode labels
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

print("\nLabel Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{label} -> {i}")

malicious_labels = [i for i, lbl in enumerate(encoder.classes_) if lbl.lower() == 'syn']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_test_original = X_test.copy()

models = {
    "Linear SVM": LinearSVC(C=1.0, max_iter=3000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train_enc)

    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test_enc, y_pred)
    cm = confusion_matrix(y_test_enc, y_pred)
    report = classification_report(y_test_enc, y_pred, target_names=encoder.classes_)

    results[name] = y_pred

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

# 🔐 Prevention mechanism
def prevent_attack(X_test_original, y_pred, malicious_labels, feature_names, classifier_name):
    blacklist = set()
    src_idx = [i for i, f in enumerate(feature_names)
               if any(x in f.lower() for x in ['src', 'ip', 'port'])]

    if not src_idx:
        src_idx = list(range(min(3, len(feature_names))))

    blocked = 0
    for i, pred in enumerate(y_pred):
        if pred in malicious_labels:
            source = tuple(X_test_original.iloc[i, src_idx].values)
            blacklist.add(source)
            blocked += 1

    print(f"{classifier_name}: Blocked {blocked}/{len(y_pred)} attacks")
    return blacklist

print("\nPrevention Results:")
blacklists = []
for name, preds in results.items():
    blacklists.append(prevent_attack(
        X_test_original, preds, malicious_labels, feature_names, name
    ))

print("\nExecution completed successfully.")
