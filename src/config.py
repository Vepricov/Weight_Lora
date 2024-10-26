from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers.utils import check_min_version
import peft

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

################################ Data Arguments ################################
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: str = field(
        default="glue",
        metadata={"help": "Name pf the dataset for fine-tuning"}
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

############################### Model Arguments ################################
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models. Possible choises are: microsoft/deberta-v3-base, meta-llama/Meta-Llama-3.1-8B, mistralai/Mistral-7B-Instruct-v0.3, microsoft/deberta-v3-base"
        }
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
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
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )

############################## Training Arguments ##############################
@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="./train_outputs",
        metadata={"help": "Folder to save output files"}
    )
    do_train: Optional[bool] = field(
        default=True,
        metadata={"help": "Do training or not"}
    )
    do_eval: Optional[bool] = field(
        default=True,
        metadata={"help": "Do evaluating or not"}
    )
    dataset_text_field: Optional[str] = field(
        default="text",
        metadata={"help": "Default text key in train dataset"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size for training"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size for evaluation"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=6,
        metadata={"help": "Number of accumulation steps to make a gradient step, i.e. before optimizer.step()"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "Scheduler for optimizer. We pass it in get_scheduler, possible options are"}
    )
    weight_decay: Optional[float] = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    eval_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "How to make eval"}
    )
    logging_steps: Optional[int] = field(
        default=1,
        metadata={"help": "How often print train loss"}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=True,
        metadata={"help": "Load best model at the end of training or not"}
    )
    seed: Optional[int] = field(
        default=18,
        metadata={"help": "Seed for all experiment"}
    )
    save_strategy: Optional[str] = field(
        default="steps",
        metadata={"help": "How to save your model"}
    )
    learning_rate: Optional[float] = field(
        default=8e-4,
        metadata={"help": "Learning rate"}
    )
    warmup_steps: Optional[int] = field(
        default=100,
        metadata={"help": "Warmup steps"}
    )
    max_steps: Optional[int] = field(
        default=512,
        metadata={"help": "Max steps of optimizer. Overrides num_train_epochs"}
    )
    eval_steps: Optional[int] = field(
        default=4,
        metadata={"help": "How often make eval"}
    )
    save_steps: Optional[int] = field(
        default=256,
        metadata={"help": "How often save model"}
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={"help": "Where to place results. none or wandb"}
    )
    ft_strategy: Optional[str] = field(
        default="LoRA",
        metadata={"help": "What PEFT strategy to use"}
    )
    run_name: Optional[str] = field(
        default="zalupa",
        metadata={"help": "Wandb run name"}
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "Rank for LoRA and LoRA-like PEFT adapters"}
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "Scaling of LoRA and LoRA-like PEFT adapters"}
    )
    lora_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "Dropout of LoRA and LoRA-like PEFT adapters"}
    )
    optim: Optional[str] = field(
        default="adamw_torch",
        metadata={"help": "Name of the optimizer to use. Possible choises available here: https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py"}
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "For wandb"}
    )
    optimizer: Optional[str] = field(
        default=None,
        metadata={"help": "For wandb"}
    )
    scheduler: Optional[str] = field(
        default=None,
        metadata={"help": "For wandb"}
    )
    benchmark_name: Optional[str] = field(
        default=None,
        metadata={"help": "For wandb"}
    )
    tsk_name: Optional[str] = field(
        default=None,
        metadata={"help": "For wandb"}
    )
    all_params: Optional[int] = field(
        default=None,
        metadata={"help": "All params of the model. For wandb"}
    )
    trainable_params: Optional[int] = field(
        default=None,
        metadata={"help": "Trainable params of the model. For wandb"}
    )
    proportion: Optional[float] = field(
        default=None,
        metadata={"help": "all_params / trainable_params * 100%. For wandb"}
    )

################################ PEFT Arguments ################################
def get_peft_arguments(training_args):
    if training_args.ft_strategy == "LoRA":
        peft_args = peft.LoraConfig(
            r                   = training_args.lora_r,
            lora_alpha          = training_args.lora_alpha,
            lora_dropout        = training_args.lora_dropout
        )
    elif training_args.ft_strategy == "LoKR":
        peft_args = peft.LoKrConfig(
            r                   = training_args.lora_r,
            lora_alpha          = training_args.lora_alpha,
            lora_dropout        = training_args.lora_dropout
        )
    elif training_args.ft_strategy == "LoHA":
        peft_args = peft.LoHaConfig(
            r                   = training_args.lora_r,
            lora_alpha          = training_args.lora_alpha,
            lora_dropout        = training_args.lora_dropout
        )
    elif training_args.ft_strategy == "VERA":
        peft_args = peft.VeraConfig(
            r                   = training_args.lora_r,
            vera_dropout        = training_args.lora_dropout
        )
    elif training_args.ft_strategy == "ADALoRA":
        peft_args = peft.AdaLoraConfig(
            target_r            = training_args.lora_r,
        )
    elif training_args.ft_strategy == "BOFT":
        # peft_args = peft.BOFTConfig(
        #     boft_block_size     = 8,
        #     bias                = "none",
        #     boft_dropout        = 0.05
        # )
        raise NotImplementedError("BOFT currently is not available :(")
    elif training_args.ft_strategy == "Full":
        return None
    else:
        raise ValueError(f"Incorrect FT type {training_args.ft_strategy}!")
    
    if "deberta" in training_args.model_name:
        peft_args.target_modules = ["query_proj", "key_proj", "value_proj",
                                    "intermediate.dence", "output.dence"]
    elif ("llama", "mistralai") in training_args.model_name:
        peft_args.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]
    else:
        raise ValueError(f"Pass target_modules to your model {training_args.model_name}")
    return peft_args
################################################################################