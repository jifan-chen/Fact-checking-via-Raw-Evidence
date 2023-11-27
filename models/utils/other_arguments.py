from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataArguments:
    dataset_name: str = field(
        default="ClaimDecomp",
        metadata={
            "help": (
                "name of the dataset: ClaimDecomp, LiarPlus, etc"
            )
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number "
                "of training examples to this value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number "
                "of training examples to this value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number "
                "of training examples to this value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    topk_docs: int =field(
        default=2,
        metadata={
            "help": (
                "how many retrieved docs are considered when feeding to the classifier"
            )
        },
    )
    topk_sents: int = field(
        default=10,
        metadata={
            "help": (
                "how many sentences are selected after aggregating retrieved sentences"
                "according to the retrieval scores"
            )
        },
    )
    num_class: int = field(
        default=6,
        metadata={
            "help": (
                "three class or six class classification"
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_justification: bool = field(
        default=False,
        metadata={"help": "Whether to use the justification paragraph to train the model or not"},
    ),
    use_claim: bool = field(
        default=False,
        metadata={"help": "Whether to use the claim to train the model or not"}
    ),
    use_decomposed_question: bool = field(
        default=False,
        metadata={"help": "Whether to use the decomposed questions to train the model or not"}
    ),
    use_answer: bool = field(
        default=False,
        metadata={"help": "Whether to use the answers to decomposed questions to train the model or not"}
    ),
    use_bm25_retrieval: bool = field(
        default=False,
        metadata={"help": "whether to use the fine-grained retrieval results from BM25"}
    ),
    use_qa_pairs: bool = field(
        default=False,
        metadata={
            "help": "whether to use the fine-grained retrieval results from BM25"}
    ),
    use_generated_questions: bool = field(
        default=False,
        metadata={
            "help": "whether to use the fine-grained retrieval results from BM25"}
    ),
    use_annotated_questions: bool = field(
        default=False,
        metadata={
            "help": "whether to use the fine-grained retrieval results from BM25"}
    ),
    use_qr_pairs: bool = field(
        default=False,
        metadata={
            "help": "whether to use the fine-grained retrieval results from BM25"}
    ),
    use_gpt_rationale: bool = field(
        default=False,
        metadata={
            "help": "whether to use the GPT generated rationale from BM25"}
    )