import argparse
import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script to evaluate pretrained DEQBert model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("model_path", help="path to model to evaluate")
    parser.add_argument("config_path", action="store_const", const="DEQBert/model_card/config.json", help="path to model config")

    parser.add_argument("--superglue", action="store_true", help="calculate model SuperGLUE benchmark performance")
    parser.add_argument("--squad", action="store_true", help="calculate model SQuAD benchmark performance")
    parser.add_argument("--race", action="store_true", help="calculate model RACE benchmark performance")

    parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")

    args = parser.parse_args()
    args = vars(args)



    if args["glue"]:


    if args["squad"]:

    if args["race"]:

