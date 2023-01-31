Running train_profiler.py should be done only from the root directory,
otherwise the relative file paths and wandb config file will not work.
```python
python -m profiler.train_profiler
```

The script will run one step of the training loop under the profiler's
context manager. The outputs will be the prof.key_average() as a pickle
file, and the trace as a json file.

You can view the trace using a web browser at something similar
to this link: chrome://tracing