import sys
sys.path.append('../pareto_testing')

from data_utils import NLPDataset
from modeling.my_modeling_bert_pruning import BertForSequenceClassification


from transformers import Trainer, TrainingArguments, EvalPrediction, BertConfig, set_seed

import numpy as np
import argparse


def compute_metrics(p: EvalPrediction):
    n_layers = len(p.predictions[0])

    accs = np.zeros(n_layers)
    for l in range(n_layers):
        preds = np.argmax(p.predictions[0][l], axis=1)
        accs[l] = (preds == p.label_ids).astype(np.float32).mean()

    return {"accuracy": accs}    


def main(args):

    training_args = TrainingArguments(
        output_dir = args.output_dir,          
        num_train_epochs = args.num_train_epochs,              
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        save_strategy='no',
        report_to = 'none'              
    )

    # Set seed
    set_seed(training_args.seed)

    # Load model
    config = BertConfig.from_pretrained(args.model_type)
    config.loss_type = 'ce'
    config.ztw = True
    config.early_pooler_hidden_size = args.early_pooler_hidden_size
    config.add_cp = False
    config.add_ee = True
    config.cp_loss = False
    config.exit_loss = True

    model = BertForSequenceClassification.from_pretrained(args.model_type, config=config)

    
    for name, param in model.named_parameters(): 
        if 'exit_heads' not in name:
            param.requires_grad = False
            print(name, param.shape)

    print('Load dataset...')
    train_dataset = NLPDataset(args.task, 'train')
    val_dataset = NLPDataset(args.task, 'val')
    test_dataset = NLPDataset(args.task, 'test', n_max = 5000)

    
    print('Train...')
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics
    )

    
    trainer.train()

    trainer.save_model()  
    
    val_res = trainer.evaluate()
    print('Validation Results')
    print(val_res)
    test_res = trainer.evaluate(test_dataset)
    print('Test Results')
    print(test_res)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type=str, default="ag")
    parser.add_argument("--model_type", type=str, default='finetune_ag_bert')
    parser.add_argument("--output_dir", type=str, default='early_exit_ag_bert')
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--early_pooler_hidden_size", type=int, default=32)
    
    args = parser.parse_args()
    
    main(args)


