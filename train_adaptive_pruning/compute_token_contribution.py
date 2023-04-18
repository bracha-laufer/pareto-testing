import sys
sys.path.append('../pareto_testing')

from data_utils import NLPDataset, max_seq_length
from attribution_utils import ModelWrapper, get_saliency


import os
import numpy as np
import argparse


from transformers import BertForSequenceClassification, BertConfig

import torch

from captum.attr import InputXGradient



def main(args):
    data_path=f'data/{args.task}'

    config = BertConfig.from_pretrained(args.model_type)

    model = BertForSequenceClassification.from_pretrained(args.model_type, config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    modelw = ModelWrapper(model)

    ablator = InputXGradient(modelw)

    attributions = {}

    for data_type in ['train', 'val', 'test']:
        print(f'Data - {data_type}')
        
        print('Load dataset...')
        dataset = NLPDataset(args.task, data_type)
        
        print('Computing attributions...')
        attributions[data_type] = get_saliency(modelw, ablator, dataset, 
                                               max_seq_length = max_seq_length[args.task],
                                               batch_size = args.batch_size)
        
        cached_features_file = os.path.join(data_path, 'cached_{}_{}_{}_{}_inputXgradient'.format(
            data_type,
            'bert',
            str(max_seq_length[args.task]),
            str(args.task)))
    
        print("Saving features into cached file", cached_features_file)
        torch.save(attributions[data_type], cached_features_file)    
       
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type=str, default="ag")
    parser.add_argument("--model_type", type=str, default='finetune_ag_bert')
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    
    main(args)







