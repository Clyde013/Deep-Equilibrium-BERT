"""
The following script should instantiate a DEQRobertaLayer and then pass some random hidden states through it
like it would in an actual model.

The hidden states are x as a size of (batch size, seq_len, hidden_size).

Weights have to be initialised closer to 0 manually as the initialisation override only applies
to RobertaPreTrainedModel.
"""

from solvers import broyden, anderson
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers import RobertaTokenizer

from bertdeq import DEQRobertaLayer

import matplotlib.pyplot as plt

from typing import Optional, Tuple


def init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(0, 0.01)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


x = torch.randn(1, 8, 768)
config = RobertaConfig(is_decoder=False, training=False)

layer = DEQRobertaLayer(config)
layer = layer.apply(init_weights)

input_tensor = torch.randn((1, 3, 768))
out = layer(input_tensor)[0]

(out * torch.randn_like(out)).sum().backward()

plt.figure(dpi=150)
plt.semilogy(layer.forward_out['rel_trace'])
plt.semilogy(layer.backward_out['rel_trace'])
plt.legend(['Forward', 'Backward'])
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.show()
