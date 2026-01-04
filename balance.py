import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

# load datasets
train_path = 'C:/Users/main/OneDrive/Documents/ddos cloud flood/datset/modified/Syn-training.csv'
save_path = os.path.dirname(train_path) + '/Syn-training-balanced.csv'

df_train = pd.read_csv(train_path)

# separate features and target
X_train = df_train.drop('Label', axis=1)
y_train = df_train['Label']

# check original class distribution
print("Original Class Distribution:", Counter(y_train))

# apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# check new class distribution
print("Resampled Class Distribution:", Counter(y_train_resampled))

# combine back to DataFrame
df_train_balanced = pd.concat([pd.DataFrame(X_train_resampled, columns=X_train.columns), pd.DataFrame(y_train_resampled, columns=['Label'])], axis=1)

# save the balanced dataset in the same folder
df_train_balanced.to_csv(save_path, index=False)
print(f"Balanced dataset saved to: {save_path}")
