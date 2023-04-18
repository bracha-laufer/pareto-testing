import sys
sys.path.append('../pareto_testing')

from data_utils import NLPDataset
from modeling.my_modeling_bert_pruning import BertForSequenceClassification

from transformers import Trainer, TrainingArguments, EvalPrediction, BertConfig, set_seed

import numpy as np
import scipy

import os
import argparse
from data_utils.dataset import max_seq_length 

def compute_metrics(p: EvalPrediction):
    
    labels = p.label_ids/np.sum(p.label_ids,axis=1,keepdims=True)

    len_seqs = np.sum(p.label_ids>0, axis=1)
    
    n_samp = p.label_ids[0].shape[0]
    n_layers = len(p.predictions[0])
    print(n_samp, n_layers)
    
    pr_all = np.zeros((n_layers, n_samp))
    sp_all = np.zeros((n_layers, n_samp))
    mse_all = np.zeros((n_layers, n_samp))

    for ll in range(n_layers):
        preds = np.exp(p.predictions[0][ll])/np.sum(np.exp(p.predictions[0][ll]),axis=-1,keepdims=True)
        for ii in range(n_samp):
            pr_all[ll,ii] = scipy.stats.pearsonr(preds[ii,:len_seqs[ii]],labels[ii,:len_seqs[ii]])[0]
            sp_all[ll,ii] = scipy.stats.spearmanr(preds[ii,:len_seqs[ii]],labels[ii,:len_seqs[ii]])[0]
            mse_all[ll,ii] = np.mean((preds[ii,:len_seqs[ii]]-labels[ii,:len_seqs[ii]])**2)
            
            
    pr = np.mean(pr_all, axis=1)
    sp = np.mean(sp_all, axis=1)
    mse = np.mean(mse_all, axis=1)
   
    return {
        "pearson": pr,
        "spearman": sp,
        "mse": mse
      }


def main(args):

    training_args = TrainingArguments(
        output_dir = args.output_dir,          
        num_train_epochs = args.num_train_epochs,
        per_device_train_batch_size = args.per_device_train_batch_size, 
        per_device_eval_batch_size = args.per_device_eval_batch_size,  
        learning_rate = args.learning_rate,
        save_strategy = 'no',
        report_to = 'none' 
    )

    set_seed(training_args.seed)

    config = BertConfig.from_pretrained(args.model_type)
    config.loss_type = 'ce'
    config.ztw = True
    config.early_pooler_hidden_size = args.early_pooler_hidden_size 
    config.add_cp = True
    config.add_ee = False
    config.cp_loss = True
    config.exit_loss = False
    config.batchnorm_size = max_seq_length[args.task]

    model = BertForSequenceClassification.from_pretrained(args.model_type, config=config)

    
    for name, param in model.named_parameters(): 
        if 'explanation_heads' not in name:
            param.requires_grad = False
            print(name, param.shape) 
    
    
    print('Load dataset...')
    train_dataset = NLPDataset(args.task, 'train', attributions=True)
    val_dataset = NLPDataset(args.task, 'val', attributions=True)
    test_dataset = NLPDataset(args.task, 'test', n_max = 5000, attributions=True)

    
    print('Train...')
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset,           
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model()  

    
    val_res = trainer.evaluate()
    print('Validation Results')
    print(val_res)

    outputs = trainer.predict(val_dataset)
    token_bounds = np.zeros((12,2))
    for i, out in enumerate(outputs.predictions[1]):        
         token_bounds[i,0]=np.min(out)
         token_bounds[i,1]=np.max(out)

    print(token_bounds)
    np.save(f'token_bounds_{args.task}.npy', token_bounds)     
    
    test_res = trainer.evaluate(test_dataset)
    print('Test Results')
    print(test_res)

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type=str, default='ag')
    parser.add_argument("--model_type", type=str, default='finetune_ag_bert')
    parser.add_argument("--output_dir", type=str, default='token_contribution_ag_bert')
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-2)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--early_pooler_hidden_size", type=int, default=32)
    
    args = parser.parse_args()
    
    main(args)


