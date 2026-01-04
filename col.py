import pandas as pd

train_path = 'C:/Users/main/OneDrive/Documents/ddos cloud flood/datset/modified/Syn-training.csv'

df_train = pd.read_csv(train_path)

print(df_train.columns)
