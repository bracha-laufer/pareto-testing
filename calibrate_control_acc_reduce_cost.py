import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch

from bounds import hb_p_value
from data_utils.dataset import max_seq_length, n_test, n_cals, n_cals1

def is_pareto(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any((costs[:i])>=c, axis=1)) and np.all(np.any((costs[i+1:])>=c, axis=1))
    return is_efficient

def fixed_sequence_testing(h_sorted, p_vals):
    list_rejected = []
    for b in range(len(h_sorted)):
        xx, yy, zz = np.unravel_index(h_sorted[b], p_vals.shape)  
        if p_vals[xx, yy, zz] < args.delta:
            list_rejected.append((xx+1,yy+1,zz+1))
        else:
            break    

    return list_rejected


def main(args):
   
    data_len = args.n_test

    acc_per_samp_file = os.path.join(args.res_folder, f'acc_per_samp_{args.data_type}_len_{data_len}')    
    cost_per_samp_file = os.path.join(args.res_folder, f'cost_per_samp_{args.data_type}_len_{data_len}')    

    acc_per_samp = torch.load(acc_per_samp_file) 
    cost_per_samp = torch.load(cost_per_samp_file) 

    N = acc_per_samp.shape[0]
    M = acc_per_samp.shape[1]
    K = acc_per_samp.shape[2]

    
    n_cal = args.n_cal
    n_cal1 = args.n_cal1
    n_cal2 = n_cal - n_cal1
    alphas = [float(f) for f in args.alphas.split(',')]
    n_alphas = len(alphas)
    
    methods = ['Pareto Testing']
    method = methods[0]
    n_methods = len(methods)

    diff_accs = {method: np.zeros((n_alphas,args.n_trials)) for method in methods}
    costs = {method: np.zeros((n_alphas,args.n_trials)) for method in methods} 

    
    for t in tqdm(range(args.n_trials)):

        all_idx = np.arange(data_len)
        np.random.shuffle(all_idx)  
        cal_idx = all_idx[:n_cal]
        test_idx = all_idx[n_cal:]

        acc_per_samp_cal = acc_per_samp[:,:,:,cal_idx]
        cost_per_samp_cal = cost_per_samp[:,:,:,cal_idx]

        acc_per_samp_cal1 = acc_per_samp_cal[:,:,:,:n_cal1]
        cost_per_samp_cal1 = cost_per_samp_cal[:,:,:,:n_cal1]
        
        acc_per_samp_cal2 = acc_per_samp_cal[:,:,:,n_cal1:]
        cost_per_samp_cal2 = cost_per_samp_cal[:,:,:,n_cal1:]
        
        acc_per_samp_test = acc_per_samp[:,:,:,test_idx]
        cost_per_samp_test = cost_per_samp[:,:,:,test_idx]

        acc_per_h_cal1 = acc_per_samp_cal1.mean(-1)
        cost_per_h_cal1 = cost_per_samp_cal1.mean(-1) 

        acc_per_h_cal2 = acc_per_samp_cal2.mean(-1)
        cost_per_h_cal2 = cost_per_samp_cal2.mean(-1) 

        acc_per_h_test = acc_per_samp_test.mean(-1)
        cost_per_h_test = cost_per_samp_test.mean(-1)

        ##########################################
        ############# Pareto Frontier ############
        ##########################################
        
        acc1 = acc_per_h_cal1.reshape(-1) 
        cost1 = cost_per_h_cal1.reshape(-1)
        utilities = np.stack((-acc1, cost1), axis=1)

        is_efficient = is_pareto(utilities) 

        all_ids = np.arange(acc1.shape[0])
        efficient_ids = all_ids[is_efficient]
        efficent_sorted = efficient_ids[np.argsort(-acc1[is_efficient])]

        for a, alpha in enumerate(alphas):
            print(f'{a}: alpha = {alpha}')

            ##########################################
            ######## Fixed Sequence Testing ##########
            ##########################################

            risk2 = acc_per_h_cal2[0,0,0] - acc_per_h_cal2
            p_vals2 = hb_p_value(risk2, n_cal2, alpha)

            list_rejected = fixed_sequence_testing(efficent_sorted, p_vals2)

            ##########################################
            ######## Select ##########
            ##########################################
            score = [-cost_per_h_cal2[id_rej[0]-1, id_rej[1]-1, id_rej[2]-1] for id_rej in list_rejected]
            if len(score)>0:
                id_max_score = score.index(max(score))
                hx, hy, hz = list_rejected[id_max_score][0]-1, list_rejected[id_max_score][1]-1, list_rejected[id_max_score][2]-1
            else:
                hx, hy, hz = 0, 0, 0
          

            diff_accs[method][a,t] = acc_per_h_test[0,0,0] - acc_per_h_test[hx, hy, hz]
            costs[method][a,t] = cost_per_h_test[hx, hy, hz]           

    
    flat_diff_accs = np.concatenate([v.reshape(-1) for v in diff_accs.values()], axis=0)
    flat_costs = np.concatenate([v.reshape(-1) for v in costs.values()], axis=0)
    flat_methods = np.repeat(methods, args.n_trials*n_alphas)
    flat_alphas = np.repeat(alphas, args.n_trials).tolist()*n_methods

    res_df = pd.DataFrame({
        'Diff-Accuracy': flat_diff_accs,
        'Relative-Cost': flat_costs,
        r'$\alpha$': flat_alphas,
        'Method': flat_methods
    })

    res_df.to_csv(os.path.join(args.res_folder, f'{args.task}_cal_res.csv'))


    plt.figure()
    sns.set(style="whitegrid")
    sns.lineplot(x=alphas, y=alphas, dashes=[(2,2)], color='black', label='diagonal')
    sns.lineplot(x=r'$\alpha$', y='Diff-Accuracy', hue="Method", style="Method", markers=True, dashes=False, data=res_df)
    plt.savefig(os.path.join(args.res_folder, f'{args.task}_acc.jpg'))  

    plt.figure()
    sns.set(style="whitegrid")
    sns.lineplot(x=r'$\alpha$', y='Relative-Cost', hue="Method", style="Method", markers=True,  dashes=False, data=res_df)
    plt.savefig(os.path.join(args.res_folder, f'{args.task}_cost.jpg'))                    

    res_df['Risk-Violations'] = (res_df['Diff-Accuracy'] > res_df[r'$\alpha$']).astype(int)
    plt.figure()
    sns.barplot(x=r'$\alpha$',y='Risk-Violations',hue='Method',data=res_df, ci=None)
    plt.savefig(os.path.join(args.res_folder, f'{args.task}_violationes.jpg'))  

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="ag")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--res_folder", type=str, default='ag_pruning_results')
    parser.add_argument("--n_test", type=int, default=7600)
    parser.add_argument("--n_cal", type=int, default=5000)
    parser.add_argument("--n_cal1", type=int, default=2500)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--alphas", type=str, default='0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2')

    args = parser.parse_args()
    print(args.alphas)

    args.n_test = n_test[f'{args.task}']
    args.n_cal = n_cals[f'{args.task}']
    args.n_cal1 = n_cals1[f'{args.task}']


    main(args)
