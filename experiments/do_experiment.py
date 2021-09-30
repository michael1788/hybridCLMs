import os, sys
import argparse
import configparser
import ast
import random
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, LSTM, BatchNormalization, Input, Dropout
from keras.optimizers import Adam
from keras.models import Sequential, load_model, Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import matplotlib
import matplotlib.pyplot as plt

import do_data_processing as ddp

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Run combined losses training')
parser.add_argument('-c','--configfile', type=str, help='Path to config file', required=True)
parser.add_argument('-s','--split', type=int, help='Data split', required=False)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)

class SeqModel():
    def __init__(self, config, n_chars, mode):
        
        self.n_chars = n_chars
        self.lr = float(config['MODEL']['lr'])
        
        # LM model
        self.layers = ast.literal_eval(config['MODEL']['neurons'])
        self.dropouts = ast.literal_eval(config['MODEL']['dropouts'])
        self.trainables = ast.literal_eval(config['MODEL']['trainables']) 
        
        # Head
        self.head_layers = ast.literal_eval(config['HEAD']['head_neurons'])
        self.head_dropouts = ast.literal_eval(config['HEAD']['head_dropouts'])
        self.head_trainables = ast.literal_eval(config['HEAD']['head_trainables']) 
        self.first_dropout = float(config['HEAD']['first_dropout'])
        self.head_output_neurons = int(config['HEAD']['head_output_neurons'])
        
        self.model = None
        self.build_model()
        
    def build_model(self):
        
        # Core (shared)
        LM_input = Input(shape=(None, self.n_chars), name='LM_input')
        x = BatchNormalization()(LM_input)
        
        n_iter = 0
        for neurons, dropout, trainable in zip(self.layers, self.dropouts, self.trainables):
            if n_iter==len(self.layers)-1:
                x = LSTM(neurons,
                     unit_forget_bias=True,
                     dropout=dropout,
                     trainable=trainable,
                     return_sequences=False)(x)
            else:
                x = LSTM(neurons,
                         unit_forget_bias=True,
                         dropout=dropout,
                         trainable=trainable,
                         return_sequences=True)(x)
            n_iter+=1
            
        x = BatchNormalization()(x)
        
        # Head
        r = Dropout(self.first_dropout)(x)
        if self.head_layers[0]!=0:
            for neurons, dropout, trainable in zip(self.head_layers, self.head_dropouts, self.head_trainables):
                r = Dense(neurons, trainable=trainable)(r)
                r = Dropout(dropout)(r)
                    
        # Ranked classification, so sigmoid and
        # multiple outputs
        head_output = Dense(self.head_output_neurons, 
                            name='head_output',
                            activation='sigmoid')(r)
        # Join
        self.model = Model(inputs=[LM_input], outputs=[head_output])
        
        # Compile
        optimizer = Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy')

    
            
def create_model_checkpoint(period, save_path):
    """ Function to save the trained model during training """
    filepath = save_path + '{epoch:02d}.h5'
    checkpointer = ModelCheckpoint(filepath=filepath,
                                   verbose=0,
                                   save_best_only=False,
                                   period=period)
    return checkpointer


def save_history_plots(history, save_path):
    
    min_val_loss = min(history.history['val_loss'])
    
    fig, ax = plt.subplots(figsize=(10,6))
    plt.plot(history.history['loss'], c='r')
    plt.plot(history.history['val_loss'], c='b')
    plt.title(f'Losses\n Min val loss: {min_val_loss:.03}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'Validation loss'], loc='upper left')
    plt.savefig(f'{save_path}losses.png')
    plt.close()

def augment_dataset(data_ori, data_target, augmentation, min_len, max_len, verbose=False):
    """ 
    Function to augment a dataset, with the corresponding target 
    
    Parameters:
    - data_ori (list): list of SMILES string to augment.
    - data_target (list): target output for SMILES in data_ori.
    - augmentation (int): number of alternative SMILES to create.
    - min_len (int): minimum length of alternative SMILES.
    - max_len (int): maximum length of alternative SMILES.
    
    return: a list alternative SMILES representations of data_ori
    """
    
    all_alternative_smi = []
    all_alternative_target = []
    for i,x in enumerate(data_ori):
        alternative_smi = ddp.smi_augmentation(x, augmentation, min_len, max_len)
        all_alternative_smi.extend(alternative_smi)
        
        n_current = len(alternative_smi)
        all_alternative_target.extend([data_target[i]]*n_current)
        
        if verbose and i%50000==0:
            print(f'augmentation is at step {i}')
    if verbose:
        print(f'data augmentation done; number of new SMILES: {len(all_alternative_smi)}')
    assert len(all_alternative_smi)==len(all_alternative_target)
    
    return all_alternative_smi, all_alternative_target

def up_sample(data, target, dict_n):
    """
    Return an up sampled version of 
    data and target according to dict_n
    - Dict_n (dict): 
        key: class to augment
        value: n data to randomly sample
    """
    
    # form a dict with key:class, value:idx_smi
    dict_byclass_smi = {}
    i=0
    for smi,clas in zip(data,target):
        if clas in dict_byclass_smi:
            dict_byclass_smi[clas].append(i)
        else:
            dict_byclass_smi[clas] = [i]
        i+=1
    
    smi_to_add = []
    tar_to_add = []
    for class_,n in dict_n.items():
        if n>0:
            for n_ in range(n):
                idx_picked = random.choice(dict_byclass_smi[class_])
                smi_to_add.append(data[idx_picked])
                tar_to_add.append(target[idx_picked])
                
    return data+smi_to_add, target+tar_to_add

    
if __name__ == '__main__':
    
    ####################################
    args = vars(parser.parse_args())
    
    verbose = True
    configfile = args['configfile']
    config = configparser.ConfigParser()
    config.read(configfile)
    repeat = args['repeat']
    split = int(args['split'])
    if split==16:
        print(f'Run on all the data {split}\n')
    elif split==4:
        print(f'Test set run {split}\n')
    else:
        print(f'Current CV fold: {split}\n')
    
    mode = str(config['EXPERIMENTS']['mode'])
    
    if 'elec' in mode:
        from python import data_generator_elec as dg
        indices_token = FP.INDICES_TOKEN_ELEC
        token_indices = FP.TOKEN_INDICES_ELEC
        pad_char = FP.PROCESSING_FIXED['pad_char']
        n_chars = len(indices_token)
        max_len = int(config['PROCESSING']['max_len'])
        max_len_model = max_len
        
        onehotencoder = dg.OneHotEncode(max_len, n_chars, 
                                        indices_token, token_indices, 
                                        pad_char)
    elif 'clm' in mode:
        from python import data_generator as dg
        indices_token = FP.INDICES_TOKEN
        token_indices = FP.TOKEN_INDICES
        start_char = FP.PROCESSING_FIXED['start_char']
        end_char = FP.PROCESSING_FIXED['end_char']
        pad_char = FP.PROCESSING_FIXED['pad_char']
        n_chars = len(indices_token)
        max_len = int(config['PROCESSING']['max_len'])
        max_len_model = max_len + 2
        
        onehotencoder = dg.OneHotEncode(max_len_model, n_chars, 
                                        indices_token, token_indices, 
                                        pad_char, start_char, end_char)

    
    if verbose: print('\nSTART EXPERIMENT')
    ########################
    
    
    
    
    ####################################
    # Path to save the checkpoints
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
 
    if repeat>0:
        save_path = f'{dir_exp}/{mode}/{exp_name}/models/{split}/{repeat}/'
    else:
        save_path = f'{dir_exp}/{mode}/{exp_name}/models/{split}/'

    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    
  
    ######################## 
    path_pretrained = str(config['MODEL']['pretrained_model'])
    if path_pretrained is not None:
        print('\nLoading pretrained weights')
        pretrained_model = load_model(path_pretrained)
    else:
        raise ValueError('No pretrained model path given')
    ########################
    

    
    
    ########################
    # load and encode the data
    data_dir = str(config['DATA']['dir'])
    data_name = str(config['DATA']['data_name'])
    data_path = f'{data_dir}{data_name}'
    data = pd.read_csv(data_path)
    
    if split==4:
        data_val = data[data.group == split]
        data_tr = data[data.group != split]    
    elif split==16:
        # that's for the alldata mode
        # We create a temp value to
        # get all the data
        tmp = 4
        data_val = data[data.group == tmp]
        data_tr = data[data.group != tmp]
    else:
        # remove test set
        data = data[data.group != 4]
        # fold argument is the fold used
        # for validation
        data_val = data[data.group == split]
        data_tr = data[data.group != split]
        
    # check   
    assert len(data_val)<len(data_tr)
    assert len(data_val)!=0
    assert len(data_tr)!=0
    
    tr_input = list(data_tr['smiles_no_salt_canon'])
    val_input = list(data_val['smiles_no_salt_canon']) 
    
    tr_output = list(data_tr['class'])
    val_output = list(data_val['class'])  
  
    # up sample the data if True
    if config.getboolean('UPSAMPLE', 'upsample'):
        upvalues = ast.literal_eval(config['UPSAMPLE']['upvalues'])
        dict_upvalues = dict(enumerate(upvalues))
        print(f'\nUp values dict: ', dict_upvalues)
        tr_input, tr_output = up_sample(tr_input, tr_output, dict_upvalues)
        print(f'\nLen tr (in,out) data after up sampling: {len(tr_input)}, {len(tr_output)}')
    
    # augment dataset if option is on
    aug = int(config['AUGMENTATION']['fold']) 
    if aug>1:
        min_len = int(config['PROCESSING']['min_len'])
        max_len = int(config['PROCESSING']['max_len'])
        
        tr_input, tr_output = augment_dataset(tr_input, tr_output,
                                              aug, min_len, max_len_model, verbose=True)
        print(f'\nDataset len after aug: \ntr: {len(tr_input)}, \val: {len(val_input)}')
        
    
    # tr data
    model_tr_input = np.empty((len(tr_input), max_len_model, n_chars), dtype=int)
    for i,x in enumerate(tr_input):
        model_tr_input[i] = onehotencoder.generator_smi_to_onehot_for_exp(x)
    
    # val data
    model_val_input = np.empty((len(val_input), max_len_model, n_chars), dtype=int)
    for i,x in enumerate(val_input):
        model_val_input[i] = onehotencoder.generator_smi_to_onehot_for_exp(x)
    ########################
    
    
    
    
    ########################
    # Concert target label to ordinal regression
    def convert(mode, label, n_output):
        """
        label: class integer
        """
        if n_output==3:
            conv_dict = {0:np.array([1,0,0]),
                         1:np.array([1,1,0]),
                         2:np.array([1,1,1])}
        else:
            raise ValueError('n_output not valid in convert(label)')
    
        return conv_dict[label]
    
    tr_output_ordi = []
    val_output_ordi = []
    
    for x in tr_output:
        tr_output_ordi.append(convert(mode, x, int(config['HEAD']['head_output_neurons'])))
    for x in val_output:
        val_output_ordi.append(convert(mode, x, int(config['HEAD']['head_output_neurons'])))
    ########################
    
    
    
   
    ########################
    # monitoring
    if split==16:
        monitor='loss'
    else:
        monitor='val_loss'
    lr_reduction = ReduceLROnPlateau(monitor=monitor, 
                                     patience=float(config['MODEL']['patience_lr']), 
                                     verbose=False, 
                                     factor=float(config['MODEL']['factor']), 
                                     min_lr=float(config['MODEL']['min_lr']))
    ########################
    
    
    
    
    ########################
    # create the model
    seqmodel = SeqModel(config, n_chars, mode)
    print(seqmodel.model.summary())
        
    if path_pretrained is not None:
        print('\nPretrained model summary:')
        print(pretrained_model.summary())
        n_l_pre = len(pretrained_model.layers)
        for i,layer in enumerate(pretrained_model.layers):
            if i!=n_l_pre-1:
                weights = layer.get_weights()
                # +1 because of the input layer added
                layer_shift = 1
                idx_layer = i+layer_shift
                seqmodel.model.layers[idx_layer].set_weights(weights)
                print(f'Pretrained layer {i} loaded in layer {idx_layer}')
    ########################
            
    
    
    
    ########################
    # start training
    batch_size = int(config['MODEL']['batch_size'])
    if config.getboolean('AUGMENTATION', 'aug_bs'):
        assert aug!=0
        batch_size = batch_size*aug
        print(f'\nNew batch size (adapted for augmentation): {batch_size}')
    
    period = int(config['MODEL']['period'])
    checkpointer = create_model_checkpoint(period, save_path)
    
    # Check if we do a training run on all the data
    if split==16:
        print('Train on all the data ON')
        model_input = np.concatenate((model_tr_input, model_val_input))
        model_output = tr_output_ordi + val_output_ordi
        n_ensemble_models = int(config['MODEL']['n_ensemble_models'])
        
        for model_n in range(n_ensemble_models):
            history = seqmodel.model.fit(x=model_input,
                                         y=np.array(model_output),
                                         epochs=int(config['MODEL']['epochs']),
                                         batch_size=batch_size,
                                         shuffle=True,
                                         callbacks=[lr_reduction,
                                                    checkpointer])
            # Save history and the model
            seqmodel.model.save(f'{save_path}{model_n}.h5')
            
            dir_history = f'{save_path}/history/'
            os.makedirs(dir_history, exist_ok=True)
            hp.save_obj(history.history, f'{dir_history}{model_n}_history') 
            
    else:
        history = seqmodel.model.fit(x=model_tr_input,
                                     y=np.array(tr_output_ordi),
                                     epochs=int(config['MODEL']['epochs']),
                                     batch_size=batch_size,
                                     shuffle=True,
                                     validation_data=(model_val_input, np.array(val_output_ordi)),
                                     callbacks=[lr_reduction,
                                                checkpointer])
        # Save history and the model
        seqmodel.model.save(f'{save_path}last.h5')
        
        dir_history = f'{save_path}/history/'
        os.makedirs(dir_history, exist_ok=True)
        hp.save_obj(history.history, f'{dir_history}history')
        
        save_history_plots(history, save_path)
    
        # load last model
        print('\nLoading best model')
        seqmodel.model = load_model(f'{save_path}last.h5')
        print('\nWeights loaded')
        
        # Save the preds of the last model
        Y_pred = seqmodel.model.predict(model_val_input)
        hp.save_obj(Y_pred, f'{save_path}Y_pred')
        hp.save_obj(val_output_ordi, f'{save_path}y_true')
        hp.write_in_file(f'{save_path}val_smi.txt', val_input)

        
    if verbose: print('\nTRAINING DONE')
    ########################
    

                                      
