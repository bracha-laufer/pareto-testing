
import sys
sys.path.append('../pareto_testing')

from data_utils import NLPDataset, num_labels

import numpy as np
import os
import argparse

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction, BertConfig, set_seed



def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
    
    print('Load dataset...')
    train_dataset = NLPDataset(args.task, 'train')
    val_dataset = NLPDataset(args.task, 'val')
    test_dataset = NLPDataset(args.task, 'test', 5000)

    config = BertConfig.from_pretrained(
        args.model_type,
        num_labels = num_labels[args.task],
    )

    model = BertForSequenceClassification.from_pretrained(args.model_type, config=config)

    print('Train...')
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset,             
        compute_metrics=compute_metrics
    )

    trainer.train()
    val_res = trainer.evaluate()
    print('Validation Results')
    print(val_res)
    test_res = trainer.evaluate(test_dataset)
    print('Test Results')
    print(test_res)
    trainer.save_model()  
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", type=str, default="ag")
    parser.add_argument("--model_type", type=str, default='bert-base-uncased')
    parser.add_argument("--output_dir", type=str, default='finetune_ag_bert')
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    main(args)


