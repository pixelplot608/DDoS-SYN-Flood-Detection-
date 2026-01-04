import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# Given results
detected = {
    "LS-SVM": {"accuracy": 0.9978, "tp": 532, "fp": 1, "fn": 1, "tn": 373},
    "Naive Bayes": {"accuracy": 0.8710, "tp": 416, "fp": 0, "fn": 117, "tn": 374},
    "K-Nearest": {"accuracy": 0.9945, "tp": 532, "fp": 1, "fn": 1, "tn": 370},
    "Multilayer Perceptron": {"accuracy": 0.9967, "tp": 532, "fp": 1, "fn": 2, "tn": 372},
}

prevention = {
    "LS-SVM": 533/907 * 100,
    "Naive Bayes": 416/907 * 100,
    "K-Nearest": 536/907 * 100,
    "Multilayer Perceptron": 534/907 * 100,
}

# Accuracy Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(detected.keys(), [v['accuracy'] * 100 for v in detected.values()], color=['blue', 'orange', 'green', 'red'])
plt.ylabel("Accuracy (%)")
plt.title("Classifier Accuracy Comparison")
plt.ylim(80, 100)
plt.show()

# Confusion Matrix Heatmaps
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (clf, values) in zip(axes.flatten(), detected.items()):
    cm = np.array([[values['tn'], values['fp']], [values['fn'], values['tp']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Syn'], yticklabels=['Benign', 'Syn'], ax=ax)
    ax.set_title(f"Confusion Matrix - {clf}")
plt.tight_layout()
plt.show()

# False Positives & False Negatives
plt.figure(figsize=(8, 5))
models = list(detected.keys())
false_positives = [v['fp'] for v in detected.values()]
false_negatives = [v['fn'] for v in detected.values()]
plt.bar(models, false_positives, label='False Positives', color='purple')
plt.bar(models, false_negatives, bottom=false_positives, label='False Negatives', color='pink')
plt.ylabel("Count")
plt.title("False Positives & False Negatives per Model")
plt.legend()
plt.show()

# Prevention Mechanism Effectiveness
plt.figure(figsize=(8, 5))
plt.bar(prevention.keys(), prevention.values(), color=['cyan', 'magenta', 'yellow', 'black'])
plt.ylabel("Percentage Blocked (%)")
plt.title("Prevention Mechanism Effectiveness")
plt.ylim(40, 100)
plt.show()
