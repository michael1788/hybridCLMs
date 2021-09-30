import os, sys
import argparse
import configparser
import numpy as np
import csv

import keras
from keras.models import load_model 

from do_analysis import convert_pred 

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP
from python import data_generator_elec as dg

parser = argparse.ArgumentParser(description='Get predictions from an ensemble of models')
parser.add_argument('-c','--configfile', type=str, help='Path to config file', required=True)
    
if __name__ == '__main__':
    
    ########################
    args = vars(parser.parse_args())
    
    verbose = True
    configfile = args['configfile']
    config = configparser.ConfigParser()
    config.read(configfile)
    
    indices_token = FP.INDICES_TOKEN_ELEC
    token_indices = FP.TOKEN_INDICES_ELEC
    n_chars = len(indices_token)
    max_len = int(config['PROCESSING']['max_len'])
    pad_char = FP.PROCESSING_FIXED['pad_char']
    n_output = 3
    
    if verbose: print('\nSTART ENSEMBLE EXPERIMENT')
    ########################
    
    
    
    
    ########################
    # Path to save the checkpoints
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
 
    save_path = f'{dir_exp}/ensemble/{exp_name}/'

    os.makedirs(save_path, exist_ok=True)
    ########################
    

    
    
    ########################
    # load and encode the data
    # and load the model and get predictions
    onehotencoder = dg.OneHotEncode(max_len, n_chars, 
                                    indices_token, token_indices, 
                                    pad_char)
    
    data_dir = str(config['DATA']['data_dir'])
    all_smis = []
    # we put all the SMILES strings together
    for fn in os.listdir(data_dir):
        if fn.endswith('.txt'):
            current = hp.read_with_pd(f'{data_dir}{fn}')
            all_smis.extend(current)
    all_smis = list(set(all_smis))
    
    # We separate data in batches to deal 
    # with big inputs
    n = 512
    chunks = [all_smis[i:i + n] for i in range(0, len(all_smis), n)]
    
    thres = float(config['MODEL']['threshold'])
    all_results = {}
    
    # Iterate over the deep ensembles
    dir_model = str(config['MODEL']['path'])
    for fn in os.listdir(dir_model):
        if fn.endswith('.h5'):
            print(f'Current model: {fn}')
            model_id = fn.split('.')[0]
            trained_model = load_model(f'{dir_model}{fn}')
            
            for j,ch_inputs in enumerate(chunks):
                s_chunks = {}
                model_input = np.empty((len(ch_inputs), max_len, n_chars), dtype=int)
                for i,smi in enumerate(ch_inputs):
                    model_input[i] = onehotencoder.generator_smi_to_onehot_for_exp(smi)
                    s_chunks[i] = smi
                all_pred_arrays = trained_model.predict(model_input)
                
                # We iterave over the predictions to save the 
                # final results, i.e. a dict with as keys SMILES,
                # and value the number of model in the ensemble
                # predicting the highest class
                for idx,smi in s_chunks.items():
                    pred_array = all_pred_arrays[idx]
                    pred_class = convert_pred(pred_array, thres, n_output)
                    # n_output-1 is the highest class
                    if pred_class==n_output-1:
                        if smi not in all_results:
                            all_results[smi]=1
                        else:
                            all_results[smi]+=1
    
    # We safe both a pickle file with the dict, and a csv
    sorted_results = sorted(all_results.items(), key=lambda kv: kv[1], reverse=True)
    with open(f'{save_path}ensemble_results.csv', 'w') as f:  
        w = csv.writer(f)
        w.writerow(['SMILES', 'Ensemble (n prediction(s) in most active class)'])
        for x in sorted_results:
            w.writerow([x[0], x[1]])
    
    if verbose: print('\nENSEMBLE PREDICTIONS DONE')
    ########################
    

                                      
