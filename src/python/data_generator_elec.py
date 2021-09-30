# based on: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
import os, sys
import numpy as np
import random
import pandas as pd
import keras
import re

from . import helper as hp

class OneHotEncode():
    def __init__(self, max_len, n_chars, indices_token, token_indices, pad_char):
        'Initialization'
        self.max_len = max_len
        self.n_chars = n_chars
        
        self.pad_char = pad_char

        self.indices_token = indices_token
        self.token_indices = token_indices

    def one_hot_encode(self, token_list, n_chars):
        output = np.zeros((token_list.shape[0], n_chars))
        for j, token in enumerate(token_list):
            output[j, token] = 1
        return output
    
    def process_tokenized_smi(self, token_list):
        """
       
        """
        padding = [self.pad_char]*(self.max_len - len(token_list))
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
    
    def do_data_corruption(self, smi, tokens, tokens_probs, ratio=0.15):
        """
        smi: SMILES in str format
        tokens: list of tokens in the vocab
        tokens_proba: proba of picking each token
        in tokens
        ratio: ratio of tokens to corupt in smi
        
        return: 
        corrupted SMILES in str format
        list of bool. 1 if corrupted, else 0
        """     
        corrupted_tokenized = hp.smi_tokenizer(smi)
        len_tokenized = len(corrupted_tokenized)
        n = round(len_tokenized*ratio)
        
        idx_to_corrupt = random.sample(range(0, len_tokenized), n)
        corrupted_tokens = np.random.choice(tokens, n, p=tokens_probs)
        
        # corrupt the smi
        truth = np.zeros([len_tokenized, 1])
        for j,idx in enumerate(idx_to_corrupt):
            old = corrupted_tokenized[idx]
            corrupted_tokenized[idx] = corrupted_tokens[j]
            if old==corrupted_tokens[j]:
                truth[idx] = 0
            else:
                truth[idx] = 1
    
        return corrupted_tokenized, truth
    
    def generator_smi_to_onehot(self, tokenized_smi):
        
        int_p_tokenized_smi = self.process_tokenized_smi(tokenized_smi)
        one_hot = self.one_hot_encode(int_p_tokenized_smi, self.n_chars)
        
        return one_hot
    
    def generator_smi_to_onehot_for_exp(self, smi):
        
        smi = self.clean_smile(smi)
        tokenized_smi = hp.smi_tokenizer(smi)
        int_p_tokenized_smi = self.process_tokenized_smi(tokenized_smi)
        one_hot = self.one_hot_encode(int_p_tokenized_smi, self.n_chars)
        
        return one_hot

    
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, max_len, path_data, n_chars, 
                 indices_token, token_indices, pad_char, 
                 tokens, tokens_probs, ratio, shuffle=True):
        'Initialization'
        self.max_len = max_len
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.path_data = path_data
        self.n_chars = n_chars
        self.tokens = tokens
        self.tokens_probs = tokens_probs
        self.ratio = ratio
                
        self.OneHotEncoder = OneHotEncode(max_len, n_chars, 
                                          indices_token, token_indices, 
                                          pad_char)

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
        
        X = np.empty((self.batch_size, self.max_len, self.n_chars), dtype=int)
        y = np.zeros((self.batch_size, self.max_len, 1), dtype=int)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            smi = self.lines[ID]
            smi = self.OneHotEncoder.clean_smile(smi)
            corrupted_tokenized, truth = self.OneHotEncoder.do_data_corruption(smi, 
                                                                               self.tokens, 
                                                                               self.tokens_probs, 
                                                                               ratio=self.ratio)
            one_hot = self.OneHotEncoder.generator_smi_to_onehot(corrupted_tokenized)
            
            X[i] = one_hot
            y[i, 0:len(truth)] = truth
            
        return X, y
    
    
    
    
    
   