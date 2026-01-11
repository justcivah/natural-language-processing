import pandas as pd
import numpy as np
import os


# fixing seed so that the training can be reproducible
rng = np.random.default_rng(1001001)

# create dataset directory if it doesn't exist
os.makedirs('datasets/dataset_ab', exist_ok=True)


train_a = pd.read_csv('datasets/dataset_a/train.csv')
valid_a = pd.read_csv('datasets/dataset_a/valid.csv')
test_a = pd.read_csv('datasets/dataset_a/test.csv')

train_b = pd.read_csv('datasets/dataset_b/train.csv')
valid_b = pd.read_csv('datasets/dataset_b/valid.csv')
test_b = pd.read_csv('datasets/dataset_b/test.csv')

half_train_len = len(train_a) // 2
half_valid_len = len(valid_a) // 2
half_test_len = len(test_a) // 2

print(len(train_a))
print(half_train_len)

# create dataset AB divided between train, validation and test
# using 50% of sentences coming from A, and 50% from B
train_ab = pd.concat([
    pd.DataFrame(train_a[0:half_train_len]),
    pd.DataFrame(train_b[0:half_train_len])
], ignore_index=True)
train_ab = train_ab.sample(frac=1, random_state=rng).reset_index(drop=True)

valid_ab = pd.concat([
    pd.DataFrame(valid_a[0:half_valid_len]),
    pd.DataFrame(valid_b[0:half_valid_len])
], ignore_index=True)
valid_ab = valid_ab.sample(frac=1, random_state=rng).reset_index(drop=True)

test_ab = pd.concat([
    pd.DataFrame(test_a[0:half_test_len]),
    pd.DataFrame(test_b[0:half_test_len])
], ignore_index=True)
test_ab = test_ab.sample(frac=1, random_state=rng).reset_index(drop=True)

train_ab.to_csv('datasets/dataset_ab/train.csv', index=False)
valid_ab.to_csv('datasets/dataset_ab/valid.csv', index=False)
test_ab.to_csv('datasets/dataset_ab/test.csv', index=False)