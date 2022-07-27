import torch
from transformers import RobertaTokenizer

from BertDEQ.bertdeq import RobertaForMaskedLM
from BertDEQ.configuration_bertdeq import BertDEQConfig


config = BertDEQConfig.from_pretrained("roberta-base")
model = RobertaForMaskedLM(config)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

model(**inputs)
