
import sys
sys.path.append('../pareto_testing')

from data_utils import NLPDataset

from transformers import BertForSequenceClassification

import torch
from torch.utils.data import SequentialSampler, DataLoader


from tqdm import tqdm
import numpy as np
import argparse 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_head_importance(model,args):
    """ This method shows how to compute:
        - neuron importance scores based on loss according to http://arxiv.org/abs/1905.10650
    """
    # prepare things for heads
    model = model.module if hasattr(model, 'module') else model
    base_model = getattr(model, model.base_model_prefix, model)
    n_layers, n_heads = base_model.config.num_hidden_layers, base_model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    head_mask = torch.ones(n_layers, n_heads).to(device)
    head_mask.requires_grad_(requires_grad=True)

    model.to(device)

    eval_dataset = NLPDataset(args.task, args.data_type) 
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
    
   
    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        batch = {key: val.to(device) for key, val in batch.items()}

        # calculate head importance
        outputs = model(batch["input_ids"], 
                        attention_mask=batch["attention_mask"], 
                        token_type_ids=batch["token_type_ids"],
                        labels=batch["labels"],
                        head_mask=head_mask)

        loss = outputs[0]
        loss.backward()
        head_importance += head_mask.grad.abs().detach()

    return head_importance

def main(args):
   
    model = BertForSequenceClassification.from_pretrained(args.model_type)
    
    model.to(device)

    head_importance = compute_head_importance(model,args)
    head_importance_np = head_importance.detach().cpu().numpy()

    print(head_importance_np)
    np.save(f'head_importance_{args.task}.npy', head_importance_np)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type=str, default="ag")
    parser.add_argument("--model_type", type=str, default='finetune_ag_bert')
    parser.add_argument("--data_type", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    
    args = parser.parse_args()
    
    main(args)

