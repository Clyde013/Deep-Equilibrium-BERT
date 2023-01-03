from tqdm.auto import tqdm
import evaluate
import collections
import numpy as np
import argparse

from transformers import Trainer
from datasets import load_dataset
from DEQBert.tokenization_deqbert_fast import DEQBertTokenizerFast
from DEQBert.modeling_deqbert import DEQBertForQuestionAnswering, DEQBertConfig

def main(checkpoint):
    raw_datasets = load_dataset("squad")
    max_length = 384
    stride = 128

    tokenizer = DEQBertTokenizerFast.from_pretrained("roberta-base")

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    n_best = 20
    max_answer_length = 30
    metric = evaluate.load("squad")

    def compute_metrics(start_logits, end_logits, features, examples):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)

    # initialise the configs
    config = DEQBertConfig.from_pretrained("DEQBert/model_card/config.json")
    config.is_decoder = False
    # update the config with number of labels for question answering head (should be 2)
    config.num_labels = 2

    model = DEQBertForQuestionAnswering.from_pretrained(checkpoint, config=config)
    trainer = Trainer(model=model)

    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script to evaluate fine-tuned DEQBert model on SQuAD",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("model_path", help="path to model to evaluate")

    args = parser.parse_args()
    args = vars(args)

    main(args['model_path'])

