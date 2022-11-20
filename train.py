import pytorch_lightning as pl
from TrainDatasets import oscar

from transformers import DataCollatorForLanguageModeling
from DEQBert.tokenization_deqbert import DEQBertTokenizer
from DEQBert.configuration_deqbert import DEQBertConfig
from transformers import Trainer, TrainingArguments
from DEQBert.modeling_deqbert import DEQBertForMaskedLM

import wandb
import torch

# To specify the GPU to use you have to set the CUDA_VISIBLE_DEVICES="0" environment variable
wandb.init(project="DEQBert",
           name="test-run")

config = DEQBertConfig.from_pretrained("DEQBert/model_card/config.json")
config.is_decoder = False
tokenizer = DEQBertTokenizer.from_pretrained("roberta-base")

model = DEQBertForMaskedLM(config=config)

oscar_datamodule = oscar.OSCARDataModule(tokenizer)
oscar_datamodule.setup()
oscar_dataset = oscar_datamodule.dataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./models",
    overwrite_output_dir=True,
    max_steps=10_000,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100,
    report_to="wandb"
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=oscar_dataset,
)

trainer.train()
