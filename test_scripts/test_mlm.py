import torch
from transformers import RobertaTokenizer
from DEQBert.tokenization_deqbert import DEQBertTokenizer
from DEQBert.modeling_deqbert import DEQBertForMaskedLM
from DEQBert.configuration_deqbert import DEQBertConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = DEQBertConfig.from_pretrained("../DEQBert/model_card/config.json")
model = DEQBertForMaskedLM(config)
model = model.to(device)

tokenizer = DEQBertTokenizer.from_pretrained("../DEQBert/model_card/")
inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")
inputs = inputs.to(device)

print(inputs)
print(model(**inputs))
