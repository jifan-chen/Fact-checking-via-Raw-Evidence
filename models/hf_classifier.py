import json
import logging
import os
import datasets
import transformers
import sys
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EvalPrediction,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from utils.other_arguments import DataArguments, ModelArguments
from utils.preprocessing import concat_claim_retrieved_sents, feed_claim_with_context, \
concat_claim_justification, concat_claim_qa_pairs, concat_claim_questions, \
concat_claim_gpt_rationale


logger = logging.getLogger(__name__)
SEP_TK = "[SEP]"


def compute_soft_acc(y_true, y_pred):
    correct = 0
    for t, p in zip(y_true, y_pred):
        if p == 0 and t in [0, 1]:
            correct += 1
        elif p == 1 and t in [0, 1, 2]:
            correct += 1
        elif p == 2 and t in [1, 2, 3]:
            correct += 1
        elif p == 3 and t in [2, 3, 4]:
            correct += 1
        elif p == 4 and t in [3, 4, 5]:
            correct += 1
        elif p == 5 and t in [4, 5]:
            correct += 1
    return correct / len(y_true)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    mae = mean_absolute_error(y_true=p.label_ids, y_pred=preds)
    accuracy = accuracy_score(y_true=p.label_ids, y_pred=preds)
    soft_acc = compute_soft_acc(y_true=p.label_ids, y_pred=preds)
    f1_marco = f1_score(y_true=p.label_ids, y_pred=preds, average='macro')
    f1_micro = f1_score(y_true=p.label_ids, y_pred=preds, average='micro')

    return {'MAE': mae,
            'Accuracy': accuracy,
            "soft-acc": soft_acc,
            'f1-macro': f1_marco,
            'f1-micro': f1_micro,
            "avg": np.mean([mae, accuracy, f1_marco, f1_micro])}


def main():
    # Get args
    # training args denotes the HF training args
    # args denotes the arguments defined in the Salesforce config files
    parser = HfArgumentParser(
        (TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # load datasets
    data_files = {}
    if training_args.do_train:
        data_files["train"] = data_args.train_file
    if training_args.do_eval:
        data_files['validation'] = data_args.validation_file
    if training_args.do_predict:
        data_files['test'] = data_args.test_file

    raw_datasets = datasets.load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
    )

    if data_args.num_class == 6:
        label_classes = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
        num_labels = len(label_classes)
        label_to_id = {v: i for i, v in enumerate(label_classes)}
    else:
        label_classes = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
        num_labels = len(label_classes) // 2
        label_to_id = {v: (i // 2) for i, v in enumerate(label_classes)}
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        # finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    def add_decomp_questions(text, decomp):
        for q in decomp:
            text += " " + q.strip()
        return text

    def add_answers(text, ans):
        for a in ans:
            text += " " + a.strip()
        return text

    def convert_raw_to_indices(examples):
        if data_args.dataset_name == "liar-plus":
            if model_args.use_justification:
                input_text = []
                for claim, just in zip(examples['claim'], examples['justification']):
                    if not just:
                        just = ""
                    input_text.append((claim + f" {SEP_TK} " + just).strip())
            else:
                input_text = examples['claim']
        elif data_args.dataset_name == "ClaimDecomp":
            input_text = []
            if model_args.use_claim:
                input_text = feed_claim_with_context(examples)
                if model_args.use_bm25_retrieval:
                    logging.info("Using bm25 retrieved evidence")
                    input_text = concat_claim_retrieved_sents(
                        input_text,
                        examples,
                        topk_sents=data_args.topk_sents,
                        topk_docs=data_args.topk_docs
                    )
                elif model_args.use_justification:
                    logging.info("Using claim and justification concat")
                    input_text = concat_claim_justification(
                        input_text,
                        examples
                    )
                elif model_args.use_gpt_rationale:
                    logging.info("Using claim and GPT summarization concat")
                    input_text = concat_claim_gpt_rationale(
                        input_text,
                        examples
                    )
                elif model_args.use_qa_pairs:
                    logging.info("Using claim and QA pairs concat")
                    input_text = concat_claim_qa_pairs(
                        input_text,
                        examples
                    )
                elif model_args.use_generated_questions:
                    logging.info("Using claim and QA pairs concat")
                    input_text = concat_claim_questions(
                        input_text,
                        examples,
                        generated_question=True
                    )
                elif model_args.use_annotated_questions:
                    logging.info("Using claim and QA pairs concat")
                    input_text = concat_claim_questions(
                        input_text,
                        examples,
                        generated_question=False
                    )

        # Tokenize the texts
        result = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=data_args.max_seq_length
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in
                               examples["label"]]
        return result
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            convert_raw_to_indices,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    else:
        train_dataset = None

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    else:
        eval_dataset = None

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
    else:
        predict_dataset = None

    es_callback = EarlyStoppingCallback(early_stopping_patience=4)
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[es_callback]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        # metrics = trainer.predict(predict_dataset, metric_key_prefix="test")
        prediction_bundle = trainer.predict(predict_dataset, metric_key_prefix="test")
        metrics = prediction_bundle.metrics
        predictions = prediction_bundle.predictions
        labels = prediction_bundle.label_ids
        output = []
        for p, l in zip(predictions, labels):
            print(np.argmax(p), l)
            output.append({
                "prediction": int(np.argmax(p)),
                "label": int(l)
            })
        print(output)
        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{training_args.seed}.json")
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                eval_dataset)
        )
        json.dump(output, open(output_predict_file, "w"), indent=4)
        metrics["test_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
