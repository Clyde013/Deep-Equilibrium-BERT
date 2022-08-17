import torch
from transformers import RobertaTokenizer

from DEQBert.DEQBert import DEQBertForMaskedLM
from DEQBert.configuration_bertdeq import DEQBertConfig


config = DEQBertConfig.from_pretrained("roberta-base")
model = DEQBertForMaskedLM(config)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

model(**inputs)
