# implicit-bert

## DEQbert

Copy [deep equilibrium model](https://arxiv.org/pdf/1909.01377.pdf) implementation. Github [here](https://github.com/locuslab/deq). Can probably reference the [Julia blogpost](https://julialang.org/blog/2021/10/DEQ/) for theoretical understanding, then implement with torchdyn library.

It seems like the residual connection in the roberta model is in [RobertaSelfOutput](https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/models/roberta/modeling_roberta.py#L286). Might have to subclass and rewrite everything up until RobertaLayer to accept the residual inputs from solver.

# TODO
- [x] Refactor bertdeq so that its not just reskinned roberta
- [ ] Combine DEQBertBlock and DEQBertLayer into a single class to remove inefficiency
- [ ] train the thing to completion

## _MEH_ URGENCY
- [ ] THE PILE DATASET NOT WORKING
- [ ] Write the report paper so I can graduate.
- [ ] profit????

# Environment Setup
```
pip list --format=freeze > requirements.txt
pip install -r requirements.txt
```

# Datasets

THE PILE (THANKS ELEUTHER VERY COOL) https://pile.eleuther.ai/

OSCAR https://huggingface.co/datasets/oscar

# References

TorchDyn github https://github.com/diffeqml/torchdyn

Pytorch Implementation of differentiable ODE Solvers https://github.com/rtqichen/torchdiffeq

Example implementations https://github.com/msurtsukov/neural-ode

Huggingface training BERT for MLM and NSP https://stackoverflow.com/questions/65646925/how-to-train-bert-from-scratch-on-a-new-domain-for-both-mlm-and-nsp

Arxiv Neural ODE https://arxiv.org/pdf/1806.07366.pdf

Arxiv ODE Transformer https://arxiv.org/pdf/2104.02308.pdf

Arxiv Deep Equilibrium Models https://arxiv.org/pdf/1909.01377.pdf

Github DEQ https://github.com/locuslab/deq

Julia blogpost on DEQ and ODE https://julialang.org/blog/2021/10/DEQ/

Openpaper Review of Transformer ODE from multi-particle system POV https://openreview.net/forum?id=SJl1o2NFwS

Github Transformer ODE https://github.com/libeineu/ODE-Transformer
