import numpy as np
import os
import json
import argparse

def split_data(n_all, n_val):
    
    all_idx = np.arange(n_all)
    np.random.shuffle(all_idx)
    n_train = n_all - n_val
    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:]
    split = {
    'train': train_idx.tolist(),
    'val' : val_idx.tolist()   
    }
    
    return split

def main(file_path, n_all, n_val):
    
    split = split_data(n_all, n_val)
    
    with open(file_path, 'w') as f:
        f.write(json.dumps(split))
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type=str, default="ag")
    parser.add_argument("--n_all", type=int, default=120000)
    parser.add_argument("--n_val", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
 
    args = parser.parse_args()  

    seed = 0
    
    np.random.seed(args.seed)
    data_path = f'data/{args.task}'
    file_path = os.path.join(data_path, 'train_val_split.txt')
    
    main(file_path, args.n_all, args.n_val)

    



