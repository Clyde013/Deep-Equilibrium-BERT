import pytorch_lightning as pl
from TrainDatasets import oscar

from transformers import RobertaTokenizer, RobertaConfig, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from BertDEQ.bertdeq import RobertaForMaskedLM

config = RobertaConfig(is_decoder=False)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

model = RobertaForMaskedLM(config=config)

oscar_datamodule = oscar.OSCARDataModule(tokenizer)
oscar_datamodule.setup()
oscar_dataset = oscar_datamodule.dataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./models",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=oscar_dataset,
)
