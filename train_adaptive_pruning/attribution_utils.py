from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_embeds, attention_mask, token_type_ids):
        return self.model(inputs_embeds = input_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]

   
def summarize_attributions(attributions, type='mean', model=None, tokens=None):
    if type == 'none':
        return attributions
    elif type == 'dot':
        embeddings = get_model_embedding_emb(model)(tokens)
        attributions = torch.einsum('bwd, bwd->bw', attributions, embeddings)
    elif type == 'mean':
        attributions = attributions.mean(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
    elif type == 'l2':
        attributions = attributions.norm(p=1, dim=-1).squeeze(0)
    return attributions


def compute_saliency(model, eval_dataloader, ablator, tot_len, max_seq_length, aggregation='l2'):

    all_attributions = np.zeros((tot_len, max_seq_length))
    device= torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    start_id = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch_ln = batch['input_ids'].shape[0]

        batch = {key: val.to(device) for key, val in batch.items()}

        input_embeddings = model.model.bert.embeddings(batch['input_ids'])

        additional = (batch['attention_mask'],batch['token_type_ids'])

        attributions = ablator.attribute(input_embeddings, 
                                         target=batch['labels'],
                                         additional_forward_args=additional)


        attributions = summarize_attributions(attributions,
                                              type=aggregation, 
                                              model=model,
                                              tokens=batch['input_ids']).detach().cpu().numpy()

        all_attributions[start_id:start_id + batch_ln] = attributions
        start_id += batch_ln
        
    return all_attributions

def get_saliency(modelw, ablator, dataset, max_seq_length, batch_size=8):
    
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
       
    attributions = compute_saliency(modelw, dataloader, ablator, len(dataset), max_seq_length)    

    return attributions

