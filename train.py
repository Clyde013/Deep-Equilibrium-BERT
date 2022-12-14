import pytorch_lightning as pl
from TrainDatasets import the_pile

from transformers import DataCollatorForLanguageModeling 
from DEQBert.tokenization_deqbert import DEQBertTokenizer
from DEQBert.configuration_deqbert import DEQBertConfig
from transformers import Trainer, TrainingArguments
from transformers.optimization import AdamW
from DEQBert.modeling_deqbert import DEQBertForMaskedLM

import wandb
import torch
from torch.optim.lr_scheduler import OneCycleLR

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

# onecycle learning rate scheduler. Thanks lucas.
optimizer = AdamW(model.parameters(),
                  betas=(wandb.config.adam_beta1, wandb.config.adam_beta2),
                  eps=wandb.config.adam_epsilon,
                  weight_decay=wandb.config.weight_decay)
scheduler = OneCycleLR(optimizer, max_lr=wandb.config.learning_rate, total_steps=int(wandb.config.total_steps))

training_args = TrainingArguments(
    output_dir=wandb.config.output_dir,
    overwrite_output_dir=True,
    max_steps=wandb.config.total_steps,
    per_device_train_batch_size=wandb.config.batch_size,
    save_strategy="steps",
    save_steps=wandb.config.save_steps,
    save_total_limit=5,
    prediction_loss_only=True,
    logging_steps=wandb.config.logging_steps,
    lr_scheduler_type=wandb.config.lr_scheduler_type,
    warmup_steps=wandb.config.warmup_steps,
    report_to="wandb"
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=pile_dataset,
    optimizers=(optimizer, scheduler)
)

trainer.train()
