import csv
import os 
import argparse
from pathlib import Path

import torch
from transformers import BertTokenizer
from dataset import max_seq_length


    
def read_ag_news_split(filepath, n=- 1):
    """Generate AG News examples."""
    texts = []
    labels = []

    with open(filepath, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(
            csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
        )
        for id_, row in enumerate(csv_reader):
            label, title, description = row
            # Original labels are [1, 2, 3, 4] ->
            #                   ['World', 'Sports', 'Business', 'Sci/Tech']
            # Re-map to [0, 1, 2, 3].
            label = int(label) - 1
            text = " ".join((title, description))
            labels.append(label)
            texts.append(text)
            #yield id_, {"text": text, "label": label}

    return texts, labels  

def process_and_cache_data_ag(args, data_type = 'train'):
    data_path = 'data/ag'
    print('Read_data...')
    texts, labels = read_ag_news_split(os.path.join(data_path, data_type + '.csv'), n=-1)
    print(f"Number of texts: {len(texts)}, number of labels: {len(labels)}")

    print('Tokenize...')
    tokenizer = BertTokenizer.from_pretrained(args.model_type)

    encodings = tokenizer(texts, truncation=True, max_length=max_seq_length[args.task], padding=True)
    encodings['labels'] = labels
    
    cached_features_file = os.path.join(data_path, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_type.split('/'))).pop(),
        str(max_seq_length[args.task]),
        str(args.task)))
    
    print("Saving features into cached file", cached_features_file)
    torch.save(encodings, cached_features_file)

def read_imdb_split(split_dir, n=- 1):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for i, text_file in enumerate((split_dir/label_dir).iterdir()):
            if n != -1:
                if i>= (n // 2):
                    break
            else:
                texts.append(text_file.read_text())
                labels.append(0 if label_dir=="neg" else 1)

    return texts, labels  

def process_and_cache_data_imdb(args, data_type = 'train'):
    data_path = 'data/imdb'
    print('Read_data...')
    texts, labels = read_imdb_split(os.path.join(data_path, data_type), n=-1)
    print(f"Number of texts: {len(texts)}, number of labels: {len(labels)}")

    print('Tokenize...')
    tokenizer = BertTokenizer.from_pretrained(args.model_type)

    encodings = tokenizer(texts, truncation=True, max_length=max_seq_length[args.task], padding=True)
    encodings['labels'] = labels
    
    cached_features_file = os.path.join(data_path, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_type.split('/'))).pop(),
        str(max_seq_length[args.task]),
        str(args.task)))
        
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type=str, default="ag")
    parser.add_argument("--model_type", type=str, default='bert-base-uncased')
  
    
    args = parser.parse_args()
    
    
    data_types = ['train' , 'test']
    
    for data_type in data_types:
        print(f"Proceesing {args.task} data  - {data_type} set")
        if args.task == 'ag':
            process_and_cache_data_ag(args, data_type)
        elif args.task == 'imdb':
            process_and_cache_data_imdb(args, data_type)    
   
    
    
    
    