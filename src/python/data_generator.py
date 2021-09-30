# based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
import os, sys
import numpy as np
import pandas as pd
import keras
import re

from . import helper as hp

class OneHotEncode():
    def __init__(self, max_len_model, n_chars, indices_token, token_indices, pad_char, start_char, end_char):
        'Initialization'
        self.max_len_model = max_len_model
        self.n_chars = n_chars
        
        self.pad_char = pad_char
        self.start_char = start_char
        self.end_char = end_char

        self.indices_token = indices_token
        self.token_indices = token_indices

    def one_hot_encode(self, token_list, n_chars):
        output = np.zeros((token_list.shape[0], n_chars))
        for j, token in enumerate(token_list):
            output[j, token] = 1
        return output
    
    def smi_to_int(self, smi):
        """
        this will turn a list of smiles in string format
        and turn them into a np array of int, with padding
        """
        token_list = hp.smi_tokenizer(smi)
        token_list = [self.start_char] + token_list + [self.end_char]
        padding = [self.pad_char]*(self.max_len_model - len(token_list))
        token_list.extend(padding)
        int_list = [self.token_indices[x] for x in token_list]
        return np.asarray(int_list)
    
    def int_to_smile(self, array):
        """ 
        From an array of int, return a list of 
        molecules in string smile format
        Note: remove the padding char
        """
        all_smi = []
        for seq in array:
            new_mol = [self.indices_token[int(x)] for x in seq]
            all_smi.append(''.join(new_mol).replace(self.pad_char, ''))
        return all_smi

    def clean_smile(self, smi):
        """ remove return line symbols """
        smi = smi.replace('\n', '')
        return smi
    
    def smile_to_onehot(self, path_data):
        
        f = open(path_data)
        lines = f.readlines()
        n_data = len(lines)
        
        X = np.empty((n_data, self.max_len_model, self.n_chars), dtype=int)
        
        for i,smi in enumerate(lines):
            # remove return line symbols
            smi = self.clean_smile(smi)
            # tokenize
            int_smi = self.smi_to_int(smi)
            # one hot encode
            X[i] = self.one_hot_encode(int_smi, self.n_chars)
            
        return X
    
    def generator_smile_to_onehot(self, smi):
        
        smi = self.clean_smile(smi)
        int_smi = self.smi_to_int(smi)
        one_hot = self.one_hot_encode(int_smi, self.n_chars)
        return one_hot

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, max_len_model, path_data, n_chars, 
                 indices_token, token_indices, pad_char, start_char, end_char, shuffle=True):
        'Initialization'
        self.max_len_model = max_len_model
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.path_data = path_data
        self.n_chars = n_chars
                
        self.OneHotEncoder = OneHotEncode(max_len_model, n_chars, 
                                          indices_token, token_indices, 
                                          pad_char, start_char, end_char)

        self.on_epoch_end()
        
        f=open(self.path_data)
        self.lines=f.readlines()
        
        self.indices_token = indices_token
        self.token_indices = token_indices

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    
    def __data_generation(self, list_IDs_temp):
        'Generates batch of data containing batch_size samples' 
        
        switch = 1
        y = np.empty((self.batch_size, self.max_len_model-switch, self.n_chars), dtype=int)
        X = np.empty((self.batch_size, self.max_len_model-switch, self.n_chars), dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            smi = self.lines[ID]
            one_hot_smi = self.OneHotEncoder.generator_smile_to_onehot(smi)
            X[i] = one_hot_smi[:-1]
            y[i] = one_hot_smi[1:]
            
        return X, y
    