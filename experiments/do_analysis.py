import os, sys
import time
import argparse
import configparser
import ast
import numpy as np
import csv
import warnings
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Run analysis')
parser.add_argument('-c','--configfile', type=str, help='Path to config file', required=True)
parser.add_argument('-r','--repeat', type=int, help='Number of repeats', required=True)
parser.add_argument('--test_set', dest='test', help='Compute stats for test set', action='store_true')
parser.add_argument('--not_test_set', dest='test', help='Do not compute stats for test set', action='store_false')
parser.set_defaults(test=False)

def get_stats_analysis(path, n_class, n_output, thres=0.5, n_folds=4, class_=2, test_set=False):
    
    all_TPR = []
    all_FPR = []

    for i in range(n_folds+1):
        if test_set and i!=4:
            # If we compute the stats for the 
            # test set, we just want fold 4,
            # which is the one of the test set
            continue
        elif not test_set and i==4:
            # If we compute the stats for the
            # CV, we don't take the test set fold
            continue

        cofm = np.zeros([3,3])

        Y_pred = hp.load_obj(f'{path}{i}/Y_pred.pkl')
        y_true = hp.load_obj(f'{path}{i}/y_true.pkl')

        Y_pred_formatted = []
        y_true_formatted = []

        for p,t in zip(Y_pred, y_true):
            p_conv = convert_pred(p, thres, n_output)
            Y_pred_formatted.append(p_conv)

            t_conv = convert_true(t, n_output)
            y_true_formatted.append(t_conv)

        confm = confusion_matrix(y_true_formatted, Y_pred_formatted)
        # We get the TPR and FPR
        # for the "highly active" class, i.e. class 2
        # with respect to the "inactive" class, i.e. class 0 (arg name: base)
        TPR = get_TPR(confm, class_)
        FPR = get_FPR(confm, class_, base=0)

        all_TPR.append(TPR)
        all_FPR.append(FPR)
    
    return np.mean(all_TPR), np.mean(all_FPR)
    
def get_TPR(confm, class_):
    TP_and_FN = np.sum(confm[class_], axis=0)
    TP = confm[class_][class_]
    TPR = TP/TP_and_FN
    return TPR

def get_FPR(confm, class_, base):
    """
    FPR refers to the "base" class
    (in the study the "inactive" class) molecules 
    misclassified as the class "class_"
    (in the study the "highly active" class)
    """
    denominator = np.sum(confm[base], axis=0)
    numerator = confm[base][class_]
    FPR = numerator/denominator
    return FPR
        
def convert_pred(array, thres, n_output):
    """
    Get the class prediction in function of the output
    of the model after the sigmoid functions (3 outputs),
    and the threshold (float between 0 and 1). 
    
    """
    if thres<0 or thres>1.0:
        raise ValueError('The threshold must be between 0 and 1') 
    
    if n_output==3:
        if array[2]>=thres and array[1]>=thres and array[0]>=thres:
            return 2
        elif array[2]<thres and array[1]>=thres and array[0]>=thres:
            return 1
        elif array[2]<thres and array[1]<thres and array[0]>=thres:
            return 0
        else:
            warnings.warn('Warning; the predicted values could not be translated into a class:')
            print('Array predictions: ', array)
            print('Threshold: ', thres)
            print('Default class value used: class 0 (inactive)')
            return 0
    else:
        raise ValueError('You must have three outputs')
            
def convert_true(array, n_output):
    if n_output==3:
        if array[2]==1:
            return 2
        elif array[1]==1:
            return 1
        elif array[0]==1:
            return 0
        else:
            raise ValueError('array values not valid')  
    else:
        raise ValueError('You must have three outputs') 
        
def do_plot(x, y, savepath):
        
    fig, ax = plt.subplots(figsize=(6,6))
    label_font_sz = 18
    labelsize = 14
    
    ax.set_ylabel('True positive rate', fontsize=label_font_sz)
    ax.set_xlabel('False positive rate',  fontsize=label_font_sz)
    ax.tick_params(axis="x", labelsize=labelsize)
    ax.tick_params(axis="y", labelsize=labelsize)
    ax.set_ylim([-0.1,1.05])
    ax.set_xlim([-0.1,1.05])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.scatter(x, y, linestyle='None', marker='o', s=20, color='k')
    plt.xlim(right=1.05)
    
    fig.savefig(f'{savepath}plot.svg', 
                format='svg', 
                dpi=1200,
                bbox_inches='tight')
        
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
    test = args['test']
    
    # get back the experiment parameters
    mode = config['EXPERIMENTS']['mode']
   
    
    if verbose: print('\nSTART NOVO ANALYSIS')
    ####################################
   
    
    
    
    ####################################
    # Path to save the novo analysis and to the generated data
    dir_exp = str(config['EXPERIMENTS']['dir'])
    exp_name = configfile.split('/')[-1].replace('.ini','')
    
    if repeat>0:
        save_path = f'{dir_exp}{mode}/{exp_name}/analysis/{repeat}/'
        path_exp = f'{dir_exp}{mode}/{exp_name}/models/{repeat}/'
    else:
        save_path = f'{dir_exp}{mode}/{exp_name}/analysis/'
        path_exp = f'{dir_exp}{mode}/{exp_name}/models/'
    
    os.makedirs(save_path, exist_ok=True)
    ####################################    
    
   


    ####################################
    # Let's get back the statistics
    
    # Fixed paramters for out study
    # You can update them if you use the pipeline
    # on something different, e.g. if you have only two
    # classes for activity
    classes = ['None', 'Low', 'High']
    n_class = len(classes)
    n_output = 3
    n_folds = 4
    
    # We iterate over a range of thresholds to find the best 
    # settings. Note that we use the same threshold at the 
    # output of each of the three sigmoids (one for of the three classes)
    # PS: don't forget that we do an ordinal classification,
    # which is why we have sigmoids intead of a softmax
    d_results_TPR = {}
    d_results_FPR = {}
    range_ = [round(x*0.01, 3) for x in range(10, 70+1)]
    for thres in range_:
        m_TPR, m_anti_FPR = get_stats_analysis(path_exp, n_class, n_output, 
                                               thres=thres, n_folds=n_folds, test_set=test)
        d_results_TPR[thres] = m_TPR
        d_results_FPR[thres] = m_anti_FPR
        
    hp.save_obj(d_results_TPR, f'{save_path}d_results_TPR.pkl')
    hp.save_obj(d_results_FPR, f'{save_path}d_results_FPR.pkl')
    
    def rounding(v, n=3):
        return round(v,n)
    
    x_FPR = []
    y_TPR = []
    thresholds = []
    # Note: here, we round the values
    for k,v in d_results_TPR.items():
        y_TPR.append(rounding(v))
        x_FPR.append(rounding(d_results_FPR[k]))
        thresholds.append(k)
    do_plot(x_FPR, y_TPR, save_path)
    
    # We also save the results as a CSV file
    with open(f'{save_path}results.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['thresholds', 'x_FPR', 'y_TPR'])
        w.writerows(zip(thresholds, x_FPR, y_TPR))
    
    end = time.time()
    if verbose: print(f'ANALYSIS DONE in {end - start:.04} seconds')
    ####################################
    