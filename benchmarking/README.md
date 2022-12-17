When running scripts from the respective benchmark subdirectories use a command like
```commandline
python -m benchmarking.SQuAD.squad models/base/checkpoint-45000 DEQBert/model_card/config.json 5
```

It is essential to run it from the root directory and as a module (-m flag) otherwise the import statements fail.