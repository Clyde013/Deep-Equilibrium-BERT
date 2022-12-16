"""
This script will provide the functionality to fine tune passed in models on the SuperGLUE dataset and then
evaluate their performance.
"""
import argparse
import numpy as np
import datasets
import wandb
from evaluate import load
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer

from DEQBert.modeling_deqbert import DEQBertForMaskedLM, DEQBertForSequenceClassification, DEQBertConfig
from DEQBert.tokenization_deqbert import DEQBertTokenizer

# list of SuperGLUE task names (https://huggingface.co/datasets/super_glue)
# note that only axb and axg are lacking train/test/validation split.
# In addition, it seems that the test split for every task lacks a proper label, and it is all -1.
task_metrics = ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc", "wsc.fixed", "axb", "axg"]


def superglue_benchmark(task, model_path, config_path, max_epochs):
    """

    Args:
        task: specific superGLUE task name (listed in task_metrics)
        model_path: path to the pretrained model
        config_path: path to the config of the pretrained model
        max_epochs: number of epochs to fine tune the model on

    Returns:
        None. Everything, including the metric computed is logged in wandb.
    """
    if task not in task_metrics:
        raise Exception("nope, not a valid task name.")
    if task in ['axb', 'axg']:
        raise Exception("yeah no, axb and axg do not have train/valid split. we are supposed to first train the"
                        "model on the MultiNLI dataset, but I am lazy to do that at this stage.")
    if task == 'record':
        raise Exception("record doesn't have labels, instead it's ")

    # initialise the configs
    config = DEQBertConfig.from_pretrained(config_path)
    config.is_decoder = False

    # create tokenizer
    tokenizer = DEQBertTokenizer.from_pretrained("roberta-base")

    # download the task splits, removing unneeded index column
    dataset = datasets.load_dataset('super_glue', task)
    dataset = dataset.remove_columns('idx')

    # tokenize function will concatenate the inputs together with a separator token before tokenizing it.
    def tokenize_function(example):
        s = []
        # each "i" is a column label
        for i in example:
            if i != 'label':
                s.append(example[i].strip())
        # have to end on SEP token, and each SEP token should have spacing between each sequence
        string = f" {tokenizer.sep_token} ".join(s) + f" {tokenizer.sep_token}"
        return tokenizer(string, truncation=True)

    # map across all splits of the dataset
    train_dataset = dataset['train'].map(tokenize_function).with_format('torch')
    valid_dataset = dataset['validation'].map(tokenize_function).with_format('torch')

    # create data collator to pad inputs
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # extract the deqbert model without the MLM head
    pretrained_model = DEQBertForMaskedLM.from_pretrained(model_path)
    deqbert = pretrained_model.deqbert

    # update the config with number of labels for sequence classification head
    config.num_labels = train_dataset.features['label'].num_classes
    # transplant the deqbert model with a sequence classification head
    model = DEQBertForSequenceClassification(config=config)
    model.deqbert = deqbert

    # loads the relevant metric for super_glue tasks, documentation here:
    # (https://huggingface.co/spaces/evaluate-metric/super_glue/blob/main/super_glue.py#L39)
    def compute_metrics(eval_preds):
        metric = load('super_glue', task)
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # training arguments
    training_args = TrainingArguments("superGLUE-benchmark",
                                      logging_steps=100,
                                      num_train_epochs=max_epochs,
                                      evaluation_strategy="epoch",
                                      report_to="wandb")

    # initialise weights and biases logging
    wandb.init(project="DEQBert-benchmarking", name=task)

    # setup trainer
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # fine tune the model, with metrics being evaluated at end of each epoch
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script to evaluate pretrained DEQBert model on superGLUE",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("model_path", help="path to model to evaluate")
    parser.add_argument("config_path", help="path to model config")
    parser.add_argument("task", help=f"One of {task_metrics}")
    parser.add_argument("epochs", type=int,
                        help="Number of epochs to finetune pretrained model on train dataset before benchmark task")

    args = parser.parse_args()
    args = vars(args)

    superglue_benchmark(args['task'], args['model_path'], args['config_path'], args['epochs'])
