import pytorch_lightning as pl
from TrainDatasets import the_pile

from transformers import DataCollatorForLanguageModeling 
from DEQBert.tokenization_deqbert import DEQBertTokenizer
from DEQBert.configuration_deqbert import DEQBertConfig
from transformers import Trainer, TrainingArguments
from DEQBert.modeling_deqbert import DEQBertForMaskedLM

import wandb
import torch
from torch.utils.data import IterableDataset

# To specify the GPU to use you have to set the CUDA_VISIBLE_DEVICES="0" environment variable
wandb.init(project="DEQBert")
wandb.run.name = wandb.config.run_name
wandb.run.save()

config = DEQBertConfig.from_pretrained("DEQBert/model_card/config.json")
config.is_decoder = False
config.hidden_dropout_prob = wandb.config.hidden_dropout
config.attention_probs_dropout_prob = wandb.config.attention_dropout
tokenizer = DEQBertTokenizer.from_pretrained("roberta-base")

model = DEQBertForMaskedLM(config=config)

pile_datamodule = the_pile.PileDataModule(tokenizer)
pile_datamodule.setup()
pile_dataset = pile_datamodule.dataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=wandb.config.mlm_probability
)

# https://github.com/huggingface/transformers/issues/19041#issuecomment-1248056494
# setting load_best_model_at_end=True and save_total_limit=1 will ensure 2 checkpoints are saved,
# the latest checkpoint and the best checkpoint.
training_args = TrainingArguments(
    output_dir=wandb.config.output_dir,
    overwrite_output_dir=True,
    max_steps=wandb.config.total_steps,
    per_device_train_batch_size=wandb.config.batch_size,
    per_device_eval_batch_size=wandb.config.batch_size,
    evaluation_strategy="steps",
    eval_steps=wandb.config.save_steps,
    save_strategy="steps",
    save_steps=wandb.config.save_steps,
    save_total_limit=1,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    logging_steps=wandb.config.logging_steps,
    learning_rate=wandb.config.learning_rate,
    weight_decay=wandb.config.weight_decay,
    adam_beta1=wandb.config.adam_beta1,
    adam_beta2=wandb.config.adam_beta2,
    adam_epsilon=wandb.config.adam_epsilon,
    lr_scheduler_type=wandb.config.lr_scheduler_type,
    warmup_steps=wandb.config.warmup_steps,
    report_to="wandb"
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=pile_dataset,
    compute_metrics=None
)

trainer.train()
