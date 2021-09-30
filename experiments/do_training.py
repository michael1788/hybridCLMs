import os, sys
import argparse
import time
import configparser
import ast
import warnings

import keras
from keras import backend as K
from keras.layers import Dense, LSTM, TimeDistributed, BatchNormalization
from keras.optimizers import Adam
from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('-c','--configfile', type=str, help='Path to config file', required=True)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)

class CLM():
    """Class for the CLM with autoregressive training"""
    def __init__(self, n_chars, max_length, layers, dropouts, trainables, lr):  
        
        self.n_chars = n_chars
        self.max_length = max_length
        
        self.layers = layers
        self.dropouts = dropouts
        self.trainables = trainables
        self.lr = lr
        
        self.model = None
        self.build_model()
        
    def build_model(self):
        
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(None, self.n_chars)))
        
        for neurons, dropout, trainable in zip(self.layers, self.dropouts, self.trainables):
            self.model.add(LSTM(neurons, unit_forget_bias=True, dropout=dropout, 
                                trainable=trainable, return_sequences=True))        
        self.model.add(BatchNormalization())
        self.model.add(TimeDistributed(Dense(self.n_chars, activation='softmax')))
        
        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        
class ELECTRA():
    """Class for the CLM with ELECTRA pretraining"""
    def __init__(self, n_chars, max_length, layers, dropouts, trainables, lr):  
        
        self.n_chars = n_chars
        self.max_length = max_length
        
        self.layers = layers
        self.dropouts = dropouts
        self.trainables = trainables
        self.lr = lr
        
        self.model = None
        self.build_model()
        
    def build_model(self):
        
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(None, self.n_chars)))
        for neurons, dropout, trainable in zip(self.layers, self.dropouts, self.trainables):
            self.model.add(LSTM(neurons, unit_forget_bias=True, dropout=dropout, 
                                trainable=trainable, return_sequences=True))        
        self.model.add(BatchNormalization())
        self.model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        
        optimizer = Adam(lr=self.lr)

        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)
        
def create_model_checkpoint(period, save_path):
    """ Function to save the trained model during training """
    filepath = save_path + '{epoch:02d}.h5' 
    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=0,
                                   save_best_only=False,
                                   period=period)

    return checkpointer



if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    verbose = True
    configfile = args['configfile']
    config = configparser.ConfigParser()
    config.read(configfile)
    repeat = args['repeat']
    
    # get back the experiment parameters
    min_len = int(config['PROCESSING']['min_len'])
    max_len = int(config['PROCESSING']['max_len'])
    mode = config['EXPERIMENTS']['mode']
    
    if verbose: print('\nSTART TRAINING')
    ####################################
    
    
    
    
    ####################################
    # Path to save the checkpoints
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    if repeat>0:
        save_path = f'{dir_exp}/{mode}/{exp_name}/models/{repeat}/'
    else:
        save_path = f'{dir_exp}/{mode}/{exp_name}/models/'
    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    
    
    ####################################
    # Neural net parameters
    patience_lr = int(config['MODEL']['patience_lr'])
    batch_size = int(config['MODEL']['batch_size'])
    print(f'\nBatch_size used: {batch_size}')
    epochs = int(config['MODEL']['epochs'])
    period = int(config['MODEL']['period'])
    n_workers = int(config['MODEL']['n_workers'])
    min_lr = float(config['MODEL']['min_lr'])
    factor = float(config['MODEL']['factor'])
    ####################################
    
    
    
    
    ####################################
    # Generator parameters
    max_len = int(config['PROCESSING']['max_len'])
    pad_char = FP.PROCESSING_FIXED['pad_char']
    ####################################
    
    
    
    
    ####################################
    # Define monitoring
    monitor = 'val_loss'
    lr_reduction = ReduceLROnPlateau(monitor=monitor, 
                                     patience=patience_lr, 
                                     verbose=0, 
                                     factor=factor, 
                                     min_lr=min_lr)
    ####################################
    
    
    
    
    ####################################
    # Path to the data
    aug = int(config['AUGMENTATION']['fold'])
    dir_data = str(config['DATA']['dir'])
    name_data = str(config['DATA']['name'])
    name_data = f'{name_data}/{min_len}_{max_len}_x{aug}/'
    dir_split_data = f'{dir_data}{name_data}'
    if verbose: print(f'Data path : {dir_split_data}')
    
    # load partitions
    partition = {}
    path_partition_train = f'{dir_split_data}idx_tr'
    path_partition_valid = f'{dir_split_data}idx_val'

    partition['train'] = hp.load_obj(path_partition_train)
    partition['val'] = hp.load_obj(path_partition_valid)
        
    # get back the name of the training data from parameters
    path_data = f'{dir_split_data}{min_len}_{max_len}_x{aug}.txt'
    ####################################
    
    
   

    ####################################
    # Create the generators
    if mode == 'lm_elec':
        from python import data_generator_elec as data_generator
        
        indices_token = FP.INDICES_TOKEN_ELEC
        token_indices  =  FP.TOKEN_INDICES_ELEC
        vocab_size = len(indices_token)
        # Define what policy we use to corrupt the input SMILES
        mode_tokens_probs = str(config['ELEC']['mode_tokens_probs'])
        if mode_tokens_probs=='naive':
            print('mode_tokens_probs is naive')
            tokens = list(token_indices.keys())
            tokens_probs = [1/len(tokens)]*len(tokens)
        elif mode_tokens_probs=='proportional':
            print('mode_tokens_probs is proportional')
            dict_prop = FP.PROP_ELEC
            tokens = []
            tokens_probs = []
            for tok, val in dict_prop.items():
                tokens.append(tok)
                tokens_probs.append(val)
        else:
            raise ValueError('mode token probs not valid')
        ratio = float(config['ELEC']['ratio'])
        
        tr_generator = data_generator.DataGenerator(partition['train'],
                                                    batch_size, 
                                                    max_len,
                                                    path_data,
                                                    vocab_size,
                                                    indices_token,
                                                    token_indices,
                                                    pad_char,
                                                    tokens, 
                                                    tokens_probs, 
                                                    ratio,
                                                    shuffle=True)
    
        val_generator = data_generator.DataGenerator(partition['val'], 
                                                     batch_size, 
                                                     max_len, 
                                                     path_data,
                                                     vocab_size,
                                                     indices_token,
                                                     token_indices,
                                                     pad_char,
                                                     tokens, 
                                                     tokens_probs, 
                                                     ratio,
                                                     shuffle=True)
    elif mode in ['clm', 'clm_ft']:
        from python import data_generator as data_generator
        
        indices_token = FP.INDICES_TOKEN
        token_indices  =  FP.TOKEN_INDICES
        vocab_size = len(indices_token)
        # +2 for start and end characters
        max_len_model = max_len+2
        start_char = FP.PROCESSING_FIXED['start_char']
        end_char = FP.PROCESSING_FIXED['end_char']
        
        tr_generator = data_generator.DataGenerator(partition['train'],
                                                    batch_size, 
                                                    max_len_model,
                                                    path_data,
                                                    vocab_size,
                                                    indices_token,
                                                    token_indices,
                                                    pad_char,
                                                    start_char,
                                                    end_char,
                                                    shuffle=True)
    
        val_generator = data_generator.DataGenerator(partition['val'], 
                                                     batch_size, 
                                                     max_len_model, 
                                                     path_data,
                                                     vocab_size,
                                                     indices_token,
                                                     token_indices,
                                                     pad_char,
                                                     start_char,
                                                     end_char,
                                                     shuffle=True)
        
    else:
        raise ValueError('Your experiment mode is not valid')
    ####################################
    
    
    
    
    ####################################
    # Create the checkpointer, the model and train.
    # Note: pretrained weights are loaded if we do
    # a fine-tuning experiment
    checkpointer = create_model_checkpoint(period, save_path)
    
    layers = ast.literal_eval(config['MODEL']['neurons'])
    dropouts = ast.literal_eval(config['MODEL']['dropouts'])
    trainables = ast.literal_eval(config['MODEL']['trainables'])
    lr = float(config['MODEL']['lr'])
    
    if mode == 'lm_elec':
        seqmodel = ELECTRA(vocab_size, max_len, layers, dropouts, trainables, lr)
    elif mode == 'clm':
        seqmodel = CLM(vocab_size, max_len, layers, dropouts, trainables, lr)
    elif mode == 'clm_ft':
        seqmodel = CLM(vocab_size, max_len, layers, dropouts, trainables, lr)
        # we load the pretrained weights
        path_model = config['FINETUNING']['pretrained_clm']
        if path_model[-2:]=='h5':
            pre_model = load_model(path_model)
            pre_weights = pre_model.get_weights()
            seqmodel.model.set_weights(pre_weights)
            print(f'Weights loaded: {path_model}')
        else:
            raise ValueError('You did not provide a path to a pretrained CLM for fine-tuning (.h5 file)')
    
    if verbose:
        seqmodel.model.summary()
    
    history = seqmodel.model.fit_generator(generator=tr_generator,
                                           validation_data=val_generator,
                                           use_multiprocessing=True,
                                           epochs=epochs,
                                           callbacks=[checkpointer,
                                                      lr_reduction],
                                           workers=n_workers)    
    
    # Save the loss history
    hp.save_obj(history.history, f'{save_path}history')
    
    end = time.time()
    print(f'TRAINING DONE in {end - start:.05} seconds')
    ####################################
    
    