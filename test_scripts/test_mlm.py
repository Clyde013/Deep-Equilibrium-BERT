import torch
from transformers import RobertaTokenizer

from DEQBert.deqbert import DEQBertForMaskedLM
from DEQBert.configuration_deqbert import DEQBertConfig


config = DEQBertConfig.from_pretrained("../DEQBert/model_card/config.json")
model = DEQBertForMaskedLM(config)

tokenizer = RobertaTokenizer.from_pretrained("../DEQBert/model_card/")
inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

print(model(**inputs))
