Running train_profiler.py should be done only from the root directory,
otherwise the relative file paths and wandb config file will not work.
```python
python -m profiler.train_profiler
```

The script will run one step of the training loop under the profiler's
context manager. The outputs will be the prof.key_average() as a pickle
file, and the trace as a json file.

If you see something strange like multiple backward passes, remember that it's because
of the gradient accumulation steps, so the number of backward passes = num grad 
accumulation steps, which actually slows down the model training quite significantly.
Each backward pass takes about ~3 seconds, which at the default 4 gradient accumulation
steps can amount to 10 second or so slow down of each step.

You can view the trace using a web browser at something similar
to this link: chrome://tracing