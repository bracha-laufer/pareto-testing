import os 
import json

import torch
import numpy as np
import pandas as pd

train_feature_file = {
    'imdb': 'cached_train_bert-base-uncased_512_imdb',
    'ag': 'cached_train_bert-base-uncased_128_ag',
    'qnli': 'cached_train_bert-base-uncased_128_qnli',
    'qqp': 'cached_train_bert-base-uncased_128_qqp'
}

test_feature_file = {
    'imdb': 'cached_test_bert-base-uncased_512_imdb',
    'ag': 'cached_test_bert-base-uncased_128_ag',
    'qnli': 'cached_test_bert-base-uncased_128_qnli',
    'qqp': 'cached_test_bert-base-uncased_128_qqp'
}

n_total = {
    'imdb': 25000,
    'ag': 7600,
    'qnli': 5463,
    'qqp': 40430,
}

n_test = {
    'imdb': 10000,
    'ag': 7600,
    'qnli': 5463,
    'qqp': 40430,
}

n_cals = {
    'imdb': 5000,
    'ag': 5000,
    'qnli': 3400,
    'qqp': 5000,
    'mnli': 5000
    }

n_cals1 = {
    'imdb': 2500,
    'ag': 2500,
    'qnli': 1700,
    'qqp': 2500,
    'mnli': 2500
    }

max_seq_length = {
    'imdb': 512,
    'ag': 128,
    'qnli': 128,
    'qqp': 128,
    'mnli': 128
}

num_labels= {
    'imdb': 2,
    'ag': 4,
    'qnli': 2,
    'qqp': 2,
    'mnli': 3,
}



class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, task, data_type, n_max=-1, attributions=None, attention_mask=None, cal_test_split = None, cal_num=1):
        data_path = f'data/{task}'
        if task == 'mnli':
           self.encodings, self.c = load_mnli(data_type, n_max, attributions)
           test_ids = np.arange(n_max)
           if data_type == 'test' and n_max>-1:
              self.c = self.c[:n_max] 
        else:
            if data_type == 'train' or  data_type == 'val':
                
                features = torch.load(os.path.join(data_path, train_feature_file[task]))
                
                split_file = os.path.join(data_path, 'train_val_split.txt')                              
                split = json.load(open(split_file))
                
                self.encodings = {}
                for key, values in features.items():
                    self.encodings[key] = [v for i, v in enumerate(values) if i in split[data_type]]
            
            elif data_type == 'test':
                
                features = torch.load(os.path.join(data_path, test_feature_file[task])) 
                if cal_test_split is None:  
                    if n_max == -1:
                        self.encodings = features
                    else: 
                        #test_ids = np.random.permutation(n_test[task])    
                        test_ids = np.random.permutation(n_total[task])    
                        test_ids = test_ids[:n_max]
                        #test_ids = np.arange(n_max)
                        self.encodings = {}
                        for key, values in features.items():
                            self.encodings[key] = [v for i, v in enumerate(values) if i in test_ids]
                else:
                    split_file = os.path.join(data_path, 'cal_test_splits.txt')                              
                    split = json.load(open(split_file))
                    split_list = split[cal_test_split]
                    if 'cal' in cal_test_split:
                        half = int(len(split_list)/2)
                        if cal_num == 1:
                           split_list = split_list[:half]
                        elif cal_num ==2:
                           split_list = split_list[half:]

                    self.encodings = {}
                    for key, values in features.items():
                        self.encodings[key] = [v for i, v in enumerate(values) if i in split_list]

        if attributions is not None:
            att_file = f'cached_{data_type}_bert_{max_seq_length[task]}_{task}_inputXgradient'  
            atts = torch.load(os.path.join(data_path, att_file)) 
            if data_type == 'test' and n_max > -1:              
                self.encodings['labels'] = [att for i, att in enumerate(atts) if i in test_ids]
            else:
                self.encodings['labels'] = atts
                
        if attention_mask is not None: 
            self.encodings['attention_mask'] = attention_mask
                    
        self.labels =  self.encodings['labels']
                

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.labels)  

def load_mnli(data_type, n_max, attributions):
    data_path = 'data/mnli'
    data_dir = os.path.join(data_path, 'data')
    glue_dir = os.path.join(data_path, 'glue_data', 'MNLI')
    
    # Read in metadata
    type_of_split = 'random'
    metadata_df = pd.read_csv(
        os.path.join(
            data_dir,
            f'metadata_{type_of_split}.csv'),
        index_col=0)


    confounder_name = 'sentence2_has_negation'

    c_array = metadata_df[confounder_name].values
    
    # Extract splits
    split_array = metadata_df['split'].values
    split_dict = {
        'train': 0,
        'val': 1,
        'test': 2
    }

    # Load features
    features_array = []
    for feature_file in [
        'cached_train_bert-base-uncased_128_mnli',
        'cached_dev_bert-base-uncased_128_mnli',
        'cached_dev_bert-base-uncased_128_mnli-mm'
        ]:

        features = torch.load(os.path.join(glue_dir, feature_file))
        features_array += features 

    encodings = {}
    encodings['input_ids'] = [f.input_ids for i, f in enumerate(features_array) if split_array[i]==split_dict[data_type]]
    encodings['attention_mask'] = [f.input_mask for i, f in enumerate(features_array) if split_array[i]==split_dict[data_type]]
    encodings['token_type_ids'] = [f.segment_ids for i, f in enumerate(features_array) if split_array[i]==split_dict[data_type]]
    encodings['labels'] = [f.label_id for i, f in enumerate(features_array) if split_array[i]==split_dict[data_type]]
    c = c_array[split_array==split_dict[data_type]]

    if data_type == 'test' and n_max>-1:
        test_ids = np.arange(n_max)
        new_encodings = {}
        for key, values in encodings.items():
            new_encodings[key] = [v for i, v in enumerate(values) if i in test_ids]
        encodings = new_encodings 

    return encodings, c     
     


    

    