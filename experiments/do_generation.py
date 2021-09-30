import os, sys
import time
import argparse
import configparser
import ast
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='SMILES generation')
parser.add_argument('-c','--configfile', type=str, help='path to config file', required=True)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)


def one_hot_encode(token_lists, n_chars):
    output = np.zeros((len(token_lists), len(token_lists[0]), n_chars))
    for i, token_list in enumerate(token_lists):
        for j, token in enumerate(token_list):
            output[i, j, int(token)] = 1
    return output

def topk_topp_sample(model, temp, start_char, end_char, max_len, indices_token, token_indices, top_k, top_p):
    
    generated = ""
    seed_token = []
    n_chars = len(indices_token)
    all_proba = np.ones([n_chars, max_len])*-1
    
    for i in range(len(start_char)):
        t = list(start_char)[i]
        generated += t
        seed_token += [token_indices[t]]
    
    loop = 0
    while generated[-1] != end_char and len(generated) < max_len:
        x_seed = one_hot_encode([seed_token], n_chars)
        full_preds = model.predict(x_seed, verbose=0)[0]
        logits = full_preds[-1]
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        
        probas, next_char_ind = get_token_proba(logits, temp)
        
        all_proba[:,loop] = probas
        
        next_char = indices_token[next_char_ind]
        generated += next_char
        seed_token += [next_char_ind]
        
        loop+=1
    
    return generated, all_proba[:, :len(generated)-1]


def get_token_proba(preds, temp):
    
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    
    probas = exp_preds / np.sum(exp_preds)
    char_ind = np.argmax(np.random.multinomial(1, probas, 1))
    
    return probas, char_ind

def softmax(preds):
    return np.exp(preds)/np.sum(np.exp(preds))

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=0):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    based on https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.shape[0])  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = np.argsort(logits)[::-1][top_k:]
        logits[indices_to_remove] = filter_value
               
    if top_p > 0.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        proba_logits = softmax(sorted_logits)
        cumulative_probs = np.cumsum(proba_logits)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        idx_to_change = np.where(sorted_indices_to_remove == True)[0][0]
        sorted_indices_to_remove[idx_to_change] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    return logits

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
    
    mode = str(config['EXPERIMENTS']['mode'])
    
    if verbose: print('\nSTART SAMPLING')
    ####################################
    
    
    
    ####################################
    # paths to save data and to checkpoints
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    if repeat>0:
        save_path = f'{dir_exp}/{mode}/{exp_name}/generated_data/{repeat}/'
        dir_ckpts = f'{dir_exp}/{mode}/{exp_name}/models/{repeat}/'
    else:
        save_path = f'{dir_exp}/{mode}/{exp_name}/generated_data/'
        dir_ckpts = f'{dir_exp}/{mode}/{exp_name}/models/'
    
    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    
    
    ####################################
    # Parameters to sample novo smiles
    temp = float(config['SAMPLING']['temp'])
    n_sample = int(config['SAMPLING']['n_sample']) 
    top_k = int(config['SAMPLING']['top_k']) 
    top_p = float(config['SAMPLING']['top_p'])
    
    max_len = int(config['PROCESSING']['max_len'])
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES
    ####################################
    
    
    
    ####################################
    # start SMILES sampling
    for filename in os.listdir(dir_ckpts):
        if filename.endswith('.h5'):            
            epoch = filename.split('/')[-1].replace('.h5', '')
            if verbose: print(f'Sampling from model saved at epoch {epoch}')
            model_path = os.path.join(dir_ckpts, filename)
            model = load_model(model_path)
            
            generated_smi = []
            for n in range(n_sample):
                sampled_smi, _ = topk_topp_sample(model, temp, 
                                                  start_char, end_char, 
                                                  max_len+1, 
                                                  indices_token, token_indices, 
                                                  top_k, top_p)
                generated_smi.append(sampled_smi)
        
            hp.save_obj(generated_smi, f'{save_path}epoch{epoch}_temp{temp}_top_k{top_k}_top_p{top_p}')
    
    end = time.time()
    if verbose: print(f'SAMPLING DONE for model from epoch {epoch} in {end-start:.2f} seconds')  
    ####################################
        