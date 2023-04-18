import numpy as np

from transformers import Trainer, TrainingArguments
import torch

from data_utils.dataset import num_labels


n_layers = 12
n_heads = 12



def entropy(x):
    # x: np, logits BEFORE softmax
    exp_x = np.exp(x)
    A = np.sum(exp_x, axis=1)    # sum of exp(x_i)
    B = np.sum(x*exp_x, axis=1)  # sum of x_i * exp(x_i)
    return np.log(A) - B/A

def compute_cost(pruned_lens,per_samp_exit, head_mask):
    n_samples = pruned_lens.shape[1]
    n_layers = pruned_lens.shape[0]

    cost_per_samp = np.zeros(n_samples)
    head_mask_per_layer = head_mask.sum(axis=1)
    for i in range(n_samples):
        len_samp_i = per_samp_exit[i].astype(int)
        for l in range(len_samp_i):
            cost_per_samp[i] += (pruned_lens[l+1][i]**2)*head_mask_per_layer[l]

    init_cost = (12*(n_layers-1)*(pruned_lens[0]**2))
    cost_per_samp = cost_per_samp/init_cost     
    
    return cost_per_samp

def early_exit(all_preds, labels, th):   
    n_samples = all_preds.shape[1] 
    n_labels = all_preds.shape[2]
    n_layers = all_preds.shape[0] 

    per_layer_entropy = np.zeros((n_layers, n_samples))
    exit_per_samp = np.zeros(n_samples)
    n_exit = np.zeros(n_layers)
    final_logs = np.zeros((n_samples, n_labels))
    all_id = []
    for l in range(1, n_layers-1):
        per_layer_entropy[l,:] = entropy(all_preds[l])
        id = [i for i in range(n_samples) if per_layer_entropy[l,i] < th]
        id_l = [i for i in id if i not in all_id]
        all_id.extend(id_l)
        
        n_exit[l] = len(id_l)
        final_logs[id_l,:] = all_preds[l][id_l,:]
        exit_per_samp[id_l] = l

    id_left = [i for i in range(n_samples) if i not in all_id]
    final_logs[id_left] = all_preds[-1][id_left,:]
    all_id.extend(id_left)
    n_exit[-1] = len(id_left)
    exit_per_samp[id_left] =  n_layers-1
    final_pred = np.argmax(final_logs, axis=1)
    acc_per_samp =  (final_pred == labels).astype(np.float32)
    
    return acc_per_samp, exit_per_samp, final_logs 


def predict(modelw, eval_dataset, args):
    training_args = TrainingArguments(
        output_dir = args.res_folder,  
        per_device_eval_batch_size = args.per_device_eval_batch_size,    
        report_to = 'none' 
    )
    
    trainer = Trainer(model=modelw,args=training_args)

    outputs = trainer.predict(eval_dataset)
    for i, out in enumerate(outputs.predictions[1]):        
         print(i, np.min(out), np.max(out))
    
    predictions_t = outputs.predictions[0]
    lengths_t = outputs.predictions[2]

    n_layers = len(predictions_t) -1 

    predictions = np.zeros((n_layers+1, len(eval_dataset), num_labels[args.task]))
    lengths = np.zeros((n_layers+1,len(eval_dataset)))

    for l, logit in enumerate(predictions_t):
        predictions[l,:,:] = logit

    for l, length in enumerate(lengths_t):   
        lengths[l,:] = length
            
    return predictions, lengths    


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.head_mask = torch.ones(n_layers, n_heads)

    def update_head_mask(head_mask):
        self.head_mask = head_mask

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.model(input_ids = input_ids, 
                         attention_mask = attention_mask, 
                         token_type_ids = token_type_ids,
                         head_mask = self.head_mask,
                         labels = labels)
        return outputs    