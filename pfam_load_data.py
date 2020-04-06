"""
Protein Sequence Classification on pfam dataset
testing from
https://github.com/ronakvijay/Protein_Sequence_Classification/blob/master/Pfam_protein_sequence_classification.ipynb

"""

import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from prettytable import PrettyTable
from IPython.display import Image

from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, GlobalMaxPooling1D

# data is randomly split in three folders [train(80%), test(10%), dev(10%)]
# reading and concatenating data for each folder.

def read_data(partition):
 data = []
 data_path = '/u2/home_u2/dc1321/DL_protein_seqs/random_split/'
 #print('Available data', os.listdir(data_path))
 #print('for partition', partition)
 for fn in os.listdir(os.path.join(data_path, partition)):
  with open(os.path.join(data_path, partition, fn)) as f:
   data.append(pd.read_csv(f, index_col=None))
   return pd.concat(data)

# plotting from the dataset #
def plot_seq_count(df, data_name):
 sns.distplot(df['seq_char_count'].values)
 plt.title(f'Sequence char count: {data_name}')
 plt.grid(True)

 # Prints no. of unique classes in data sets #

def calc_unique_cls(train, test, val):
 train_unq = np.unique(train['family_accession'].values)
 val_unq = np.unique(val['family_accession'].values)
 test_unq = np.unique(test['family_accession'].values)

 print('Number of unique classes in Train: ', len(train_unq))
 print('Number of unique classes in Val: ', len(val_unq))
 print('Number of unique classes in Test: ', len(test_unq))

# getting frequency of amino acids #
def get_code_freq(df, data_name):
 df = df.apply(lambda x: " ".join(x))

 codes = []
 for i in df:  # concatenation of all codes
  codes.extend(i)

 codes_dict = Counter(codes)
 codes_dict.pop(' ')  # removing white space

 print(f'Codes: {data_name}')
 print(f'Total unique codes: {len(codes_dict.keys())}')

 df = pd.DataFrame({'Code': list(codes_dict.keys()), 'Freq': list(codes_dict.values())})
 return df.sort_values('Freq', ascending=False).reset_index()[['Code', 'Freq']]

# plotting histogram of frequencies #
def plot_code_freq(df, data_name):

  plt.title(f'Code frequency: {data_name}')
  sns.barplot(x='Code', y='Freq', data=df)


# reading all data_partitions
df_train = read_data('train')
df_val = read_data('dev')
df_test = read_data('test')

# printing out info of the dataset
"""
print(df_train.info())
print(df_train.head())
print(df_train.head(1)['sequence'].values[0])
"""
"""
print(df_val.info())
print(df_val.head())
print(df_val.head(1)['sequence'].values[0])
"""
"""
print(df_test.info())
print(df_test.head())
print(df_test.head(1)['sequence'].values[0])
"""


#print("size of train, val and test")

#print('Train size: ', len(df_train))
#print('Val size: ', len(df_val))
#print('Test size: ', len(df_test))


# Length of sequence in train data
df_train['seq_char_count']= df_train['sequence'].apply(lambda x: len(x))
df_val['seq_char_count']= df_val['sequence'].apply(lambda x: len(x))
df_test['seq_char_count']= df_test['sequence'].apply(lambda x: len(x))


#calc_unique_cls(df_train, df_test, df_val)

"""
plt.subplot(1, 3, 1)
plot_seq_count(df_train, 'Train')

plt.subplot(1, 3, 2)
plot_seq_count(df_val, 'Val')

plt.subplot(1, 3, 3)
plot_seq_count(df_test, 'Test')

plt.subplots_adjust(right=3.0)
plt.show()

"""

# train code sequence

train_code_freq = get_code_freq(df_train['sequence'], 'Train')
val_code_freq = get_code_freq(df_val['sequence'], 'Val')
test_code_freq = get_code_freq(df_test['sequence'], 'Test')
#print(train_code_freq)
#print(val_code_freq)
#print(test_code_freq)

"""
plt.subplot(131)
plot_code_freq(train_code_freq, 'Train')

plt.subplot(132)
plot_code_freq(val_code_freq, 'Val')

plt.subplot(133)
plot_code_freq(test_code_freq, 'Test')

plt.subplots_adjust(right=3.0)
plt.show()
"""

# Most observed family group #
#print(df_test.groupby('family_id').size().sort_values(ascending=False).head(20))

# Considering top 1000 classes based on most observations because of limited computational power.

classes = df_train['family_accession'].value_counts()[:1000].index.tolist()
print(len(classes))

# Filtering data based on considered 1000 classes.
train_sm = df_train.loc[df_train['family_accession'].isin(classes)].reset_index()
val_sm = df_val.loc[df_val['family_accession'].isin(classes)].reset_index()
test_sm = df_test.loc[df_test['family_accession'].isin(classes)].reset_index()

print('Data size after considering 1000 classes for each data split:')
print('Train size :', len(train_sm))
print('Val size :', len(val_sm))
print('Test size :', len(test_sm))

print("No. of unique classes after reducing the data size")

calc_unique_cls(train_sm, test_sm, val_sm)