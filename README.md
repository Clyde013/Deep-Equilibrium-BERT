# bert-ode

Okay so let's try to implement this thing that probably won't work because I suck.

<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" width="500">

Everyone knows how this works. Notice that the encoder decoder blocks all contain residual layers (the loop thing connected to Add & Norm).

Following the implementation of the [ODE Transformer](https://arxiv.org/pdf/2104.02308.pdf) we can replace the residual layers with ODEs. This isn't particularly novel, but an ODE variation of BERT does not yet exist, despite being just the transformer encoding block of the Transformer ODE being used for pretraining.

The source of the original ODE Transformer is still alive and kicking but based on fairseq, which I have no clue how to use.

So, I guess now we can combine TorchDyn supported ODEs with HuggingFace models which are pretty well documented and easy to extend.

Unfortunately, BERT pretraining took 52 hours with colab's cloud based TPUs. So this is a problem for when I actually implement and try training the model to validate that it works.

# TODO
Add co co as contributor ;)

Properly learn the math... should probably do this yep.

Go through [TorchDyn tutorial notebooks](https://github.com/DiffEqML/torchdyn/tree/master/tutorials) and learn how to actually use the library.

Figure out how tf to extend HuggingFace's BERT implementation to support ODEs.

Write the report paper so I can graduate.

profit????

# Environment Setup
```
pip freeze > requirements.txt
pip install -r requirements.txt
```

# References

TorchDyn github https://github.com/diffeqml/torchdyn

Pytorch Implementation of differentiable ODE Solvers https://github.com/rtqichen/torchdiffeq

Example implementations https://github.com/msurtsukov/neural-ode

Arxiv Neural ODE https://arxiv.org/pdf/1806.07366.pdf

Arxiv ODE Transformer https://arxiv.org/pdf/2104.02308.pdf

Openpaper Review of Transformer ODE from multi-particle system POV https://openreview.net/forum?id=SJl1o2NFwS

Github Transformer ODE https://github.com/libeineu/ODE-Transformer
