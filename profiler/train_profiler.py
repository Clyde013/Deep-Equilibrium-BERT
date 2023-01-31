import pickle

import datasets
from transformers import DataCollatorForLanguageModeling 
from DEQBert.tokenization_deqbert import DEQBertTokenizer
from DEQBert.configuration_deqbert import DEQBertConfig
from DEQBert.modeling_deqbert import DEQBertForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers.optimization import AdamW


import wandb
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import OneCycleLR

if __name__ == "__main__":
    # To specify the GPU to use you have to set the CUDA_VISIBLE_DEVICES="0" environment variable
    wandb.init(project="DEQBert-Profiler")
    wandb.run.name = wandb.config.run_name
    wandb.run.save()

    config = DEQBertConfig.from_pretrained("DEQBert/model_card/config.json")
    config.is_decoder = False
    config.hidden_dropout_prob = wandb.config.hidden_dropout
    config.attention_probs_dropout_prob = wandb.config.attention_dropout

    tokenizer = DEQBertTokenizer.from_pretrained("roberta-base")

    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):

            model = DEQBertForMaskedLM(config=config)
            if wandb.config.load_checkpoint is not None:
                # we load checkpoints from state_dicts directly instead of using trainer.save(resume_from_checkpoint=...)
                # because this allows us to alter scheduler hyperparameters when resuming training.
                model = model.from_pretrained(wandb.config.load_checkpoint)

            # while would preferably use the_pile datamodule, it is inconvenient to download the entire dataset + map the
            # encoding function to it. So we simply copy-paste the relevant parts and apply it to the hacker_news subset
            def encode(example_batch):
                # tokenize the text
                features = tokenizer(example_batch["text"], max_length=128, padding="max_length",
                                          truncation=True, return_tensors="pt")
                return features
            pile_dataset = datasets.load_dataset("the_pile", streaming=False, split="train", subsets=["hacker_news"]).select(range(32))
            pile_dataset = pile_dataset.shuffle(seed=69)
            pile_dataset = pile_dataset.map(encode, batched=True, remove_columns=["text", "meta"]).with_format("torch")

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=wandb.config.mlm_probability
            )

            # onecycle learning rate scheduler. Thanks lucas.
            optimizer = AdamW(model.parameters(),
                              betas=(wandb.config.adam_beta1, wandb.config.adam_beta2),
                              eps=wandb.config.adam_epsilon,
                              weight_decay=wandb.config.weight_decay)
            scheduler = OneCycleLR(optimizer, max_lr=wandb.config.learning_rate, total_steps=int(wandb.config.total_steps),
                                   last_epoch=-1)
            # the scheduler has to initialise initial_lr for the optimizer, before we can set last_epoch. After the scheduler
            # steps once the learning rate will be set to as if it was at last_epoch.
            scheduler.last_epoch = wandb.config.resume_steps

            training_args = TrainingArguments(
                output_dir='../models/profiler/',
                max_steps=1,
                per_device_train_batch_size=wandb.config.batch_size,
                gradient_accumulation_steps=wandb.config.grad_accum_steps,
                prediction_loss_only=True,
                logging_steps=wandb.config.logging_steps,
                report_to="wandb"
            )


            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=pile_dataset,
                optimizers=(optimizer, scheduler)
            )

            print("profiling training...")
            trainer.train()

    print("profiling complete, exporting trace...")
    # view this file with chrome://tracing
    prof.export_chrome_trace("profiler/results/trace.json")

    print("pickling profiler...")
    # just in case, we save the profiler object for later analysis
    outfile = open('profiler/results/profiler.pickle', 'wb')
    pickle.dump(prof.key_averages(), outfile)
    outfile.close()
