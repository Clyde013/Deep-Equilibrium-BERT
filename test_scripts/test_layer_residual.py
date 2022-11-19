"""
The following script should instantiate a DEQRobertaLayer and then pass some random hidden states through it
like it would in an actual model.

The hidden states are x as a size of (batch size, seq_len, hidden_size).

Weights have to be initialised closer to 0 manually as the initialisation override only applies
to RobertaPreTrainedModel.
"""

import torch
import torch.nn as nn

from DEQBert.modeling_deqbert import DEQBertLayer
from DEQBert.configuration_deqbert import DEQBertConfig

import matplotlib.pyplot as plt


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# remember to make hidden size a multiple of the number of attention heads (12)
batch_size, seq_len, hidden_size = 3, 5, 24
input_tensor = torch.randn((batch_size, seq_len, hidden_size)).to(device)
print(f"Testing with tensor of {input_tensor.shape}")

config = DEQBertConfig(is_decoder=False, training=False, hidden_size=hidden_size)

layer = DEQBertLayer(config)
layer = layer.apply(init_weights)
layer.cuda(device)

out = layer(input_tensor)[0]

(out * torch.randn_like(out)).sum().backward()

plt.figure(dpi=150)
plt.semilogy(layer.forward_out['rel_trace'])
plt.semilogy(layer.backward_out['rel_trace'])
plt.legend(['Forward', 'Backward'])
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.show()