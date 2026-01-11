import pandas as pd
import numpy as np

# during translation some sentences may not have been translated
# removing sentences with nan (or empty) translation from A and B
train_a = pd.read_csv('datasets/dataset_a/train.csv')
valid_a = pd.read_csv('datasets/dataset_a/valid.csv')
test_a = pd.read_csv('datasets/dataset_a/test.csv')

train_a = train_a.replace(r'^\s*$', np.nan, regex=True).dropna()
valid_a = valid_a.replace(r'^\s*$', np.nan, regex=True).dropna()
test_a = test_a.replace(r'^\s*$', np.nan, regex=True).dropna()

train_b = pd.read_csv('datasets/dataset_b/train.csv')
valid_b = pd.read_csv('datasets/dataset_b/valid.csv')
test_b = pd.read_csv('datasets/dataset_b/test.csv')

train_b = train_b.replace(r'^\s*$', np.nan, regex=True).dropna()
valid_b = valid_b.replace(r'^\s*$', np.nan, regex=True).dropna()
test_b = test_b.replace(r'^\s*$', np.nan, regex=True).dropna()

# i want that len(valid_a) == len(test_a) == len(valid_b) == len(test_b)
# and that len(train_a) == len(train_b)

min_valid_test_len = min(len(valid_a), len(test_a), len(valid_b), len(test_b))
valid_a = valid_a[0:min_valid_test_len]
test_a = test_a[0:min_valid_test_len]
valid_b = valid_b[0:min_valid_test_len]
test_b = test_b[0:min_valid_test_len]

min_train_len = min(len(train_a), len(train_b))


print(len(train_a))
print(len(train_b))

print(min_train_len)

train_a = train_a[0:min_train_len]
train_b = train_b[0:min_train_len]

print(len(train_a))
print(len(train_b))

print(min_train_len)

train_a.to_csv('datasets/dataset_a/train.csv', index=False)
valid_a.to_csv('datasets/dataset_a/valid.csv', index=False)
test_a.to_csv('datasets/dataset_a/test.csv', index=False)

train_b.to_csv('datasets/dataset_b/train.csv', index=False)
valid_b.to_csv('datasets/dataset_b/valid.csv', index=False)
test_b.to_csv('datasets/dataset_b/test.csv', index=False)