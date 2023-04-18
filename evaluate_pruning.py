
from data_utils.dataset import NLPDataset
from data_utils.dataset import num_labels

from transformers import BertConfig
from modeling.my_modeling_bert_pruning import BertForSequenceClassification

import os
import pandas as pd
import seaborn as sns
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from pruning_utils import predict, early_exit, compute_cost, ModelWrapper
from data_utils.dataset import max_seq_length, n_test

do_plot = True

def main(args):

    th_np_ee = np.linspace(args.min_th_ee, args.max_th_ee, args.n_ee)
    th_list_ee = [round(th_np_ee[i],2) for i in range(args.n_ee)]

    th_np_token = np.linspace(args.min_th_token, args.max_th_token, args.n_token)
    th_list_token = [round(th_np_token[i],2) for i in range(args.n_token)]

    th_np_head = np.linspace(args.min_th_head, args.max_th_head, args.n_head)
    th_list_head = [round(th_np_head[i],2) for i in range(args.n_head)]

    config = BertConfig.from_pretrained(args.model_type1)

    config.add_cp = True
    config.add_ee = True
    config.cp_loss = False
    config.exit_loss = True

    config.batchnorm_size = max_seq_length[args.task]


    model = BertForSequenceClassification.from_pretrained(
                                       args.model_type1, config=config)

    model2 = BertForSequenceClassification.from_pretrained(
                                       args.model_type2, config=config)  

    own_state = model.state_dict()
    for name, param in model2.state_dict().items():
        if 'explanation_heads' not in name:
                continue
        if isinstance(param, torch.nn.parameter.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)                                                                     

    modelw = ModelWrapper(model)
           
    eval_dataset = NLPDataset(args.task, args.data_type, n_max=n_test[args.task])
    label_ids = eval_dataset.labels
    
    head_importance = np.load(f'head_importance_{args.task}.npy')
    head_importance = head_importance/ np.sum(head_importance, axis=1, keepdims=True)    

    accs = np.zeros((len(th_list_token), len(th_list_head),len(th_list_ee)))
    rel_costs = np.zeros((len(th_list_token), len(th_list_head), len(th_list_ee)))
    predictions = np.empty((args.n_token, args.n_head), dtype=object)
    lengths = np.empty((args.n_token, args.n_head), dtype=object)

    acc_per_samp = np.zeros((len(th_list_token), len(th_list_head),len(th_list_ee), len(eval_dataset)))
    cost_per_samp = np.zeros((len(th_list_token), len(th_list_head), len(th_list_ee), len(eval_dataset)))
    pred_per_samp = np.zeros((len(th_list_token), len(th_list_head), len(th_list_ee), len(eval_dataset), num_labels[args.task]))

    for i, th_token in enumerate(th_list_token):
        for j, th_head in enumerate(th_list_head):
            print(f'i = {i}, j = {j}')
            print(f'Evaluate with token pruning threshold {th_token} and head threshold {th_head}')
            
            
            head_mask_np = (head_importance > th_head).astype(int)
            head_mask = torch.tensor(head_mask_np)
            modelw.model.to_CP(mode='token_prune', delta=th_token)
            modelw.head_mask = head_mask
            
            predictions[i,j], lengths[i,j] = predict(modelw, eval_dataset, args)    
            
            
            for k, th_ee in enumerate(th_list_ee):

                acc_per_samp[i,j,k,:], exit_per_samp, pred_per_samp[i,j,k,:,:] = early_exit(
                                                            predictions[i,j], 
                                                            label_ids, th_ee)
 
                cost_per_samp[i,j,k,:] = compute_cost(lengths[i,j],
                                                        exit_per_samp, 
                                                        head_mask_np)

                accs[i,j,k] =  acc_per_samp[i,j,k,:].mean()   
                rel_costs[i,j,k] = cost_per_samp[i,j,k,:].mean()
                                        
                
                print(accs[i,j,k] , rel_costs[i,j,k])

      
    if not os.path.exists(args.res_folder):
        os.makedirs(args.res_folder)

    preds_file = os.path.join(args.res_folder, f'preds_{args.data_type}_len_{len(eval_dataset)}')    
    torch.save(predictions, preds_file)   

    lens_file = os.path.join(args.res_folder, f'lens_{args.data_type}_len_{len(eval_dataset)}')    
    torch.save(lengths, lens_file)  

    label_file = os.path.join(args.res_folder, f'label_{args.data_type}_len_{len(eval_dataset)}')    
    torch.save(label_ids, label_file)  

    acc_per_samp_file = os.path.join(args.res_folder, f'acc_per_samp_{args.data_type}_len_{len(eval_dataset)}')    
    cost_per_samp_file = os.path.join(args.res_folder, f'cost_per_samp_{args.data_type}_len_{len(eval_dataset)}')    
    pred_per_samp_file = os.path.join(args.res_folder, f'pred_per_samp_{args.data_type}_len_{len(eval_dataset)}')    

    torch.save(acc_per_samp, acc_per_samp_file) 
    torch.save(cost_per_samp, cost_per_samp_file)   
    torch.save(pred_per_samp, pred_per_samp_file)   
  

    th_ee_repeat = []
    for th in th_list_ee:
        th_ee_repeat.extend([th]*args.n_token*args.n_head)

    th_head_repeat = []
    for th in th_list_head:
        th_head_repeat.extend([th]*args.n_token)

    th_head_repeat = th_head_repeat*args.n_ee    


    df = pd.DataFrame({
    'EE-Threshold': th_ee_repeat,  
    'Head-Threshold': th_head_repeat, 
    'Token-Threshold': th_list_token*args.n_head*args.n_ee, 
    'Accuracy':  accs.reshape(-1,order='F'), 
    'Relative Cost': rel_costs.reshape(-1,order='F'), 
    })

    print(df)

    df_file = os.path.join(args.res_folder, f'three_res_{args.data_type}_len_{len(eval_dataset)}.csv')    

    df.to_csv(df_file)
    
    plot_res(df, args.task, args.data_type, len(eval_dataset), args.res_folder)
    print('Evaluation Finished')
    
def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)


def plot_res(df, task, data_type, data_len, res_folder):
    grid = sns.FacetGrid(df, col = "Head-Threshold", col_wrap=3)
    grid.map_dataframe(draw_heatmap, 'EE-Threshold', 'Token-Threshold', 'Accuracy',vmin=0.45, vmax=0.95) 
    grid.fig.subplots_adjust(top=0.9)
    grid.fig.suptitle('Accuracy')
    plt.savefig(os.path.join(res_folder, f'{task}_res_acc_{data_type}_len_{data_len}.jpg')) 


    grid = sns.FacetGrid(df, col = "Head-Threshold", col_wrap=3)
    grid.map_dataframe(draw_heatmap, 'EE-Threshold', 'Token-Threshold', 'Relative Cost',vmin=0.0, vmax=1.0) 
    grid.fig.subplots_adjust(top=0.9)
    grid.fig.suptitle('Relative Cost')
    plt.savefig(os.path.join(res_folder, f'{task}_res_cost_{data_type}_len_{data_len}.jpg')) 

    
    palette = sns.color_palette("mako_r", len(df["EE-Threshold"].unique()))
    ax3 = plt.figure(figsize=(8,5))
    ax3 = sns.scatterplot(data=df, x='Relative Cost', y='Accuracy', 
                        hue="EE-Threshold", 
                        size="Token-Threshold",
                        style="Head-Threshold" ,
                        palette=palette)
    ax3.legend(loc='lower right', ncol=4)                     
    ax3.set_title('Accuracy vs Relative Cost')
    plt.savefig(os.path.join(res_folder, f'{task}_res_acc_vs_cost_{data_type}_len_{data_len}.jpg'))     
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type=str, default="ag")
    parser.add_argument("--data_type", type=str, default="test")
    parser.add_argument("--res_folder", type=str, default="ag_pruning_res")
    parser.add_argument("--model_type1", type=str, default='early_exit_ag_bert')
    parser.add_argument("--model_type2", type=str, default='token_contribution_ag_bert')
    parser.add_argument("--output_dir", type=str, default='finetune_ag_bert')
    parser.add_argument("--n_ee", type=int, default=18)
    parser.add_argument("--min_th_ee", type=float, default=0.0)
    parser.add_argument("--max_th_ee", type=float, default=0.8)
    parser.add_argument("--n_token", type=int, default=20)
    parser.add_argument("--min_th_token", type=float, default=-2.0)
    parser.add_argument("--max_th_token", type=float, default=1.0)
    parser.add_argument("--n_head", type=int, default=18)
    parser.add_argument("--min_th_head", type=float, default=0.0)
    parser.add_argument("--max_th_head", type=float, default=1/6)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)

    args = parser.parse_args()

    token_bounds = np.load(f'token_bounds_{args.task}.npy')
    print(token_bounds)
    args.min_th_token = np.min(token_bounds[:,0]) 
    args.max_th_token = np.max(token_bounds[:,1])

    main(args)


