import pandas as pd
import numpy as np


# fixing seed so that the training can be reproducible
rng = np.random.default_rng(1001001)


# create dataset A divided between train, validation and test
df_a = pd.read_csv('datasets/dataset_a/raw.csv')

# shuffle all unique sentences
sents_a = df_a['english'].unique()
rng.shuffle(sents_a)

train_a = pd.DataFrame({'english': sents_a[0:35000]})
valid_a = pd.DataFrame({'english': sents_a[35000:40000]})
test_a = pd.DataFrame({'english': sents_a[40000:45000]})

#train_a.to_csv('datasets/dataset_a/train.csv', index=False)
#valid_a.to_csv('datasets/dataset_a/valid.csv', index=False)
#test_a.to_csv('datasets/dataset_a/test.csv', index=False)


# create dataset B divided between train, validation and test
df_b = pd.read_csv('datasets/dataset_b/raw.csv', sep='\t', names=['english'])

# shuffle all unique sentences
sents_b = df_b['english'].unique()
rng.shuffle(sents_b)

train_b = pd.DataFrame({'english': sents_b[0:35000]})
valid_b = pd.DataFrame({'english': sents_b[35000:40000]})
test_b = pd.DataFrame({'english': sents_b[40000:45000]})

train_b.to_csv('datasets/dataset_b/train.csv', index=False)
valid_b.to_csv('datasets/dataset_b/valid.csv', index=False)
test_b.to_csv('datasets/dataset_b/test.csv', index=False)