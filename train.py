import pytorch_lightning as pl
from TrainDatasets import oscar

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
tokenizer = DEQBertTokenizer.from_pretrained("roberta-base")

model = DEQBertForMaskedLM(config=config)

oscar_datamodule = oscar.OSCARDataModule(tokenizer)
oscar_datamodule.setup()
oscar_dataset = oscar_datamodule.dataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=wandb.config.mlm_probability
)

training_args = TrainingArguments(
    output_dir=wandb.config.output_dir,
    overwrite_output_dir=True,
    max_steps=wandb.config.total_steps,
    per_device_train_batch_size=wandb.config.batch_size,
    save_steps=wandb.config.save_steps,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=wandb.config.logging_steps,
    report_to="wandb"
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=oscar_dataset,
)

trainer.train()
