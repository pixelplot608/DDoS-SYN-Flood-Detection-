import pandas as pd

# load the datasets
# correct file paths
train_path = 'C:/Users/main/OneDrive/Documents/ddos cloud flood/datset/modified/Syn-training.csv'
test_path = 'C:/Users/main/OneDrive/Documents/ddos cloud flood/datset/modified/Syn-testing.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# basic info
print("Training Data Info:")
print(df_train.info())
print(df_train.head())

print("\nTesting Data Info:")
print(df_test.info())
print(df_test.head())

# check for missing values
print("\nMissing Values in Training Data:")
print(df_train.isnull().sum())

print("\nMissing Values in Testing Data:")
print(df_test.isnull().sum())

# check for duplicates
print("\nDuplicate Rows in Training Data:", df_train.duplicated().sum())
print("Duplicate Rows in Testing Data:", df_test.duplicated().sum())

# check class balance (assuming 'label' is the target column, change if needed)
print("\nClass Distribution in Training Data:")
print(df_train['Label'].value_counts())

print("\nClass Distribution in Testing Data:")
print(df_test['Label'].value_counts())
