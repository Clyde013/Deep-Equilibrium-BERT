"""
This script will provide the functionality to fine tune passed in models on the SuperGLUE dataset and then
evaluate their performance.
"""

from DEQBert.modeling_deqbert import DEQBertForMaskedLM, DEQBertForSequenceClassification, DEQBertConfig


# list of SuperGLUE task names
tasks = ['axb', 'axg', 'boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']

def finetune(task, model_path, config_path):
    # initialise the configs
    config = DEQBertConfig.from_pretrained(config_path)
    config.is_decoder = False

    # extract the deqbert model without the MLM head
    pretrained_model = DEQBertForMaskedLM.from_pretrained(model_path)
    deqbert = pretrained_model.deqbert

    # transplant the deqbert model with a sequence classification head
    model = DEQBertForSequenceClassification(config=config)
    model.deqbert = deqbert


