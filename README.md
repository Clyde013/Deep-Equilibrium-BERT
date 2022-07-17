# implicit-bert

## bert-ode

Okay so let's try to implement this thing that probably won't work because I suck.

<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" width="500">

Everyone knows how this works. Notice that the encoder decoder blocks all contain residual layers (the loop thing connected to Add & Norm).

Following the implementation of the [ODE Transformer](https://arxiv.org/pdf/2104.02308.pdf) we can replace the residual layers with ODEs. This isn't particularly novel, but an ODE variation of BERT does not yet exist, despite being just the transformer encoding block of the Transformer ODE being used for pretraining.

The source of the original ODE Transformer is still alive and kicking but based on fairseq, which I have no clue how to use.

So, I guess now we can combine TorchDyn supported ODEs with HuggingFace models which are pretty well documented and easy to extend.

Unfortunately, BERT pretraining took 52 hours with colab's cloud based TPUs. So this is a problem for when I actually implement and try training the model to validate that it works.

Foreseeable issues:

Numerical instability will occur when using treating each block as a first order ODE. The ODE transformer paper addresses this using the Runge Kutta method and treating multiple blocks as higher order ODEs. However we can also just directly use Deep Equilibrium Models as DEQs are basically ODEs where t_span approaches infinity (i.e. infinitely many ODE blocks / infinite order ODE), with the added benefit that since we are solving for the equilibrium point directly, we do not care about tracing the solution path that brings us there (useful in other applications, but not in transformers).

## bert-deq

Copy [deep equilibrium model](https://arxiv.org/pdf/1909.01377.pdf) implementation. Github [here](https://github.com/locuslab/deq). Can probably reference the [Julia blogpost](https://julialang.org/blog/2021/10/DEQ/) for theoretical understanding, then implement with torchdyn library.

It seems like the residual connection in the roberta model is in [RobertaSelfOutput](https://github.com/huggingface/transformers/blob/v4.19.2/src/transformers/models/roberta/modeling_roberta.py#L286). Might have to subclass and rewrite everything up until RobertaLayer to accept the residual inputs from solver.

# TODO
- [x] Write training loop !!!!

## _MEH_ URGENCY
- [ ] THE PILE DATASET NOT WORKING
- [ ] Write the report paper so I can graduate.
- [ ] profit????

## DONE
- [x] FINISH OSCAR.PY MODULE
- [x] Go through [TorchDyn tutorial notebooks](https://github.com/DiffEqML/torchdyn/tree/master/tutorials) and learn how to actually use the library.
- [x] Figure out how tf to extend HuggingFace's BERT implementation to support ODEs.
- [x] Write the training loop using LightningModule

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
