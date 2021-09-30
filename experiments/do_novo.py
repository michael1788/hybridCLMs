import os, sys
import time
import argparse
import configparser
import ast
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Run novo analysis')
parser.add_argument('-c','--configfile', type=str, help='Path to config file', required=True)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)

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
    mode = config['EXPERIMENTS']['mode']
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    min_len = int(config['PROCESSING']['min_len'])
    max_len = int(config['PROCESSING']['max_len'])
    aug = int(config['AUGMENTATION']['fold'])
    dir_data = str(config['DATA']['dir'])
    name_data = str(config['DATA']['name'])
    
    # experiment parameters depending on the mode
    aug = int(config['AUGMENTATION']['fold'])
    
    if verbose: print('\nSTART NOVO ANALYSIS')
    ####################################
   
    
    
    
    ####################################
    # Path to save the novo analysis and to the generated data
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    
    if repeat>0:
        save_path = f'{dir_exp}/{mode}/{exp_name}/novo_molecules/{repeat}/'
        path_gen = f'{dir_exp}/{mode}/{exp_name}/generated_data/{repeat}/'
    else:
        save_path = f'{dir_exp}/{mode}/{exp_name}/novo_molecules/'
        path_gen = f'{dir_exp}/{mode}/{exp_name}/generated_data/'
    
    os.makedirs(save_path, exist_ok=True)
    ####################################    
    
    
    
    ####################################
    # Load the pretraining data and fine-tuning
    # data to compare the novelty of the generated
    # SMILES strings
    
    # pretraining data
    dir_data_pretraining = str(config['DATA']['dir_data_pretraining'])
    full_path_pretraining = f'{dir_data_pretraining}{min_len}_{max_len}_x{aug}/'
    with open(f'{full_path_pretraining}data_tr.txt', 'r') as f:
        data_training = f.readlines()
    with open(f'{full_path_pretraining}data_val.txt', 'r') as f:
        data_validation = f.readlines()
    full_pretraining_data = data_training + data_validation
    
    # fine-tuning data
    dir_data_ft = str(config['DATA']['dir'])
    name_data_ft = str(config['DATA']['name'])
    full_path_ft = f'{dir_data_ft}{name_data_ft}/{min_len}_{max_len}_x{aug}/'
    with open(f'{full_path_ft}data_tr.txt', 'r') as f:
        ft_data_training = f.readlines()
    with open(f'{full_path_ft}data_val.txt', 'r') as f:
        ft_data_validation = f.readlines()
    full_ft_data = ft_data_training + ft_data_validation
    
    # Merge everything together
    data_training = list(set(full_pretraining_data + full_ft_data))
    ####################################
    
    
    
    
    ####################################
    # Start iterating over the files
    t0 = time.time()
    for filename in os.listdir(path_gen):
        if filename.endswith('.pkl'):
            name = filename.replace('.pkl', '')
            data = hp.load_obj(path_gen + name)
                        
            valids = []
            n_valid = 0
            
            for gen_smile in data:
                if len(gen_smile)!=0 and isinstance(gen_smile, str):
                    gen_smile = gen_smile.replace(pad_char,'')
                    gen_smile = gen_smile.replace(end_char,'')
                    gen_smile = gen_smile.replace(start_char,'')
                    
                    mol = Chem.MolFromSmiles(gen_smile)
                    if mol is not None: 
                        cans = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                        if len(cans)>=1:
                            n_valid+=1
                            valids.append(cans)
                    
            if n_valid!=0:
                # Now let's pruned our valid guys
                unique_set = set(valids)
                n_unique = len(unique_set)
                novo_tr = list(unique_set - set(data_training))
                n_novo_tr = len(novo_tr)
                novo_val = list(unique_set - set(data_validation))
                n_novo_val = len(novo_val)
                novo_analysis = {'n_valid': n_valid,
                                 'n_unique': n_unique,
                                 'n_novo_tr': n_novo_tr,
                                 'n_novo_val': n_novo_val,
                                 'novo_tr': novo_tr}
                
                # we save the novo molecules also as .txt
                novo_name = f'{save_path}molecules_{name}'
                with open(f'{novo_name}.txt', 'w+') as f:
                    for item in novo_tr:
                        f.write("%s\n" % item)
                        
                hp.save_obj(novo_analysis, novo_name)
                
                if verbose: print(f'sampling analysis for {name} done')
            else:
                print(f'There are n {n_valid} valids SMILES for {name}')
                
    
    end = time.time()
    if verbose: print(f'NOVO ANALYSIS DONE in {end - start:.04} seconds')
    ####################################
    