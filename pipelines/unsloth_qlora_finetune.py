import os
import pandas as pd
# import torch_optimizer
import torch
import transformers
# from unsloth import FastLanguageModel
# from unsloth.chat_templates import get_chat_template
# from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, HfArgumentParser, get_scheduler
from dataclasses import dataclass 
from peft import WeightLoraConfig, get_peft_model, LoraConfig, LoKrConfig
from datasets import Dataset

try:
    import adapters
    import optimizers
    import utils
except ModuleNotFoundError:
    import pipelines.adapters as adapters
    import pipelines.optimizers as optimizers
    import pipelines.utils as utils

@dataclass
class ModelArguments:
    #model_name: str = "/media/ssd-3t/akazakov/llama31instr/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693"
    # model_name: str = "unsloth/Meta-Llama-3.1-8B" 
    # model_name: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    # model_name: str = "FacebookAI/roberta-base"
    model_name: str = "NousResearch/Llama-2-7b-chat-hf"
    max_seq_length: int = 1000
    dtype: str = None
    load_in_4bit: bool = True

@dataclass
class TrainingArguments(TrainingArguments):
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 5
    num_train_epochs: int = 5
    learning_rate: float = 5e-4
    # fp16: bool = not is_bfloat16_supported()
    # bf16: bool = is_bfloat16_supported()
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 1
    optim: str = "adamw_hf"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 18
    output_dir: str = "train_outputs"
    # output_dir: str = None
    sign_step: int = 5000
    max_grad_norm: float = 1.0
    max_steps: int = 4 # overrides num_train_epochs
    report_to: str = "none" # "none" or "wandb"
 
@dataclass
class DataArguments:
    train_file: str = 'data/train_ft_short_system.jsonl'
 
def main():
    # parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    # model_args, training_args, data_args = parser.parse_args_into_dataclasses()
    model_args = ModelArguments
    training_args = TrainingArguments
    data_args = DataArguments

    utils.set_seed(training_args.seed)
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("We will use the GPU:", torch.cuda.get_device_name(i))
    # device = utils.set_device(0)
    quant_config = transformers.BitsAndBytesConfig(
        load_in_4bit=model_args.load_in_4bit,
        #bnb_4bit_quant_type="nf4",
        #bnb_4bit_compute_dtype=torch.float16,
        #bnb_4bit_use_double_quant=False
    )
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=quant_config,
        device_map='auto'
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    #tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    # tokenizer = transformers.LlamaTokenizer.from_pretrained(model_args.model_name, 
    #                                                         device_map=device)
    # tokenizer.pad_token_id = tokenizer.eos_token_id

    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name=model_args.model_name,
    #     max_seq_length=model_args.max_seq_length,
    #     dtype=model_args.dtype,
    #     load_in_4bit=model_args.load_in_4bit
    # )

    # print(model)

    # for name, param in model.named_parameters():
    # #     if param.shape == torch.Size([8388608, 1]):
    # #         print(name, type(param))
    #     if "self_attn.q_proj" in name:
    #         print(name, param.shape)
    # # #     # print(param.shape)
    # return 0

    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
    #                   "up_proj", "down_proj", "embed_tokens", "lm_head"]
    #target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
    #                  "up_proj", "down_proj"]
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", 
    #                   "up_proj", "down_proj"]
    # target_modules = ["query", "key", "value"]
    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     Config = WeightLoraConfig,
    #     r=8,
    #     target_modules=target_modules,
    #     lora_alpha=16,
    #     lora_dropout=0,
    #     bias="none",
    #     use_gradient_checkpointing="unsloth",
    #     random_state=training_args.seed,
    #     use_rslora=False,
    #     loftq_config=None,
    # )

    # print(model)
    # return 0

    config = LoraConfig(
        task_type="CausalLM",
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        #rank_dropout=0.0,
        #module_dropout=0.0,
    )
    # # print(model)
    # model = get_peft_model(model, config)

    # model.gradient_checkpointing_enable()
    # model.enable_input_require_grads()
    # print(model)
    # for name, param in model.named_parameters():
    #     if "lm_head" in name or "embed" in name:
    #         param.requires_grad = True
    # model.print_trainable_parameters()
    #     else:
    #         # param.data = param.data.to(torch.float32)
    #         param.requires_grad = True
    # adapter_args = {"rank" : 2, "w_init_value" : 1.}
    # adapter_args = {}
    # model = adapters.get_peft_model(model, adapter=adapters.WeightLoraLayer, 
    #                                 target_modules=target_modules,
    #                                 adapter_args=adapter_args)
    # for name, param in model.named_parameters():
    #     if "lora_A" in name:
    #         print(name)
    # print(type(model))
    # model = adapters.add_lora_weight(model, target_modules=target_modules)
    #print(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad is True:
    #         print(name)

    # tokenizer = get_chat_template(
    #     tokenizer,
    #     chat_template="llama-3"
    # )
    # transformed_data = utils.load_and_transform_jsonl(data_args.train_file)
    # train_dataset = Dataset.from_list(transformed_data)
    # def formatting_prompts_func(examples):
    #     convos = examples["chat"]
    #     texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    #     return {"text": texts}
    # dataset = train_dataset.map(formatting_prompts_func, batched=True)
    import datasets
    dataset = datasets.load_dataset("garage-bAInd/Open-Platypus")
    dataset["train"].to_pandas()
    def convert_dataset(data):
        instruction = data["instruction"]
        output = data["output"]
        prompt = f"<s>[INST] {instruction} [/INST] {output} </s>"
        return {'text': prompt}

    converted_data = [convert_dataset(row) for row in dataset["train"]]
    train_dataset = Dataset.from_pandas(pd.DataFrame(converted_data))
    # print(train_dataset[:5])

    # import datasets
    # dataset_name = 'cais/mmlu'
    # dataset_config_name = 'philosophy'
    # dataset = datasets.load_dataset(dataset_name, dataset_config_name)
    # train = utils.make_mlm_dataset_form_mmlu(dataset['test'])
    # test = utils.make_mlm_dataset_form_mmlu(dataset['validation'])
    # dataset = datasets.DatasetDict({"test" : test, "train" : train})
    # def tokenize_function(examples):
    #     return tokenizer(examples['text'], return_special_tokens_mask=True)
    # tokenized_dataset = dataset.map(
    #     tokenize_function,
    #     batched=True
    # )

    optimizer, scheduler = None, None
    ######################### Optimizer and Scheduler ##########################
    optimizer = optimizers.AdamW(model.parameters(), 
                                 lr=training_args.learning_rate,
                                 weight_decay=training_args.weight_decay)
    scheduler = get_scheduler(name=training_args.lr_scheduler_type, 
                              optimizer=optimizer, 
                              num_warmup_steps=training_args.warmup_steps,
                              num_training_steps=training_args.max_steps)
    # import dadaptation
    # optimizer = dadaptation.DAdaptAdam(model.parameters(), 
    #                                    lr=training_args.learning_rate,
    #                                    weight_decay=training_args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
    #                                                        T_max=training_args.max_steps)
    ############################################################################

    run_name = None
    if training_args.report_to == "wandb":
        if optimizer is not None:
            run_name=optimizer.__class__.__name__
        else:
            run_name=training_args.optim
        run_name += "_myadapter"
    
    # print(model)
    os.environ["WANDB_PROJECT"] = "SBER_LORA"
    trainer = SFTTrainer( 
        model=model, 
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=model_args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            warmup_steps=training_args.warmup_steps,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            logging_steps=training_args.logging_steps,
            optim=training_args.optim,
            weight_decay=training_args.weight_decay,
            lr_scheduler_type=training_args.lr_scheduler_type,
            seed=training_args.seed,
            output_dir=training_args.output_dir,
            max_steps=training_args.max_steps,
            report_to=training_args.report_to,
            run_name=run_name,
            max_grad_norm=training_args.max_grad_norm
        ),
        peft_config = config,
        optimizers=[optimizer, scheduler]
    )

    # trainer = transformers.Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     #train_dataset=dataset,
    #     train_dataset=dataset, 
    #     # eval_dataset=tokenized_dataset['test'],
    #     #dataset_text_field="text",
    #     #max_seq_length=model_args.max_seq_length,
    #     #dataset_num_proc=2,
    #     #packing=False,
    #     # args=TrainingArguments(
    #     #     per_device_train_batch_size=training_args.per_device_train_batch_size,
    #     #     gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    #     #     warmup_steps=training_args.warmup_steps,
    #     #     num_train_epochs=training_args.num_train_epochs,
    #     #     learning_rate=training_args.learning_rate,
    #     #     fp16=training_args.fp16,
    #     #     bf16=training_args.bf16,
    #     #     logging_steps=training_args.logging_steps,
    #     #     optim=training_args.optim,
    #     #     weight_decay=training_args.weight_decay,
    #     #     lr_scheduler_type=training_args.lr_scheduler_type,
    #     #     seed=training_args.seed,
    #     #     output_dir=training_args.output_dir,
    #     #     max_steps=training_args.max_steps,
    #     #     report_to=training_args.report_to,
    #     #     run_name=run_name,
    #     #     max_grad_norm=training_args.max_grad_norm,
    #     # ),
    #     args=TrainingArguments(
    #         max_steps=10,
    #         # learning_rate=1e-3, 
    #         # fp16=True, 
    #         output_dir=training_args.output_dir, 
    #         use_cpu=False, 
    #         # save_safetensors=False,
    #         # report_to=report_to,
    #         report_to="none",
    #         logging_steps=1,
    #         # run_name=f"prob={int(prob*100)}/100_k={k}",
    #         run_name=run_name,
    #         # run_name="test"
    #     ),
    #     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, 
    #                                                                mlm=False),
    #     optimizers=[optimizer, scheduler]
    # )

    trainer_stats = trainer.train()

    # for name, param in model.named_parameters():
    #     #if "weight_lora" in name:
    #     if "lora_A" in name or "lora_B" in name:
    #         print(name, "; sum = ", param.sum().item())

    # for name, param in model.named_parameters():
    #     if '.w.' in name:
    #         print(name, param.data)

    if training_args.report_to == "wandb":
        import wandb
        if optimizer is not None:
            add_wandb_config = {"optimizer" : optimizer.__class__.__name__,
                                "scheduler" : scheduler.__class__.__name__,
                                }
        else:
            add_wandb_config = {"optimizer" : training_args.optim,
                                "scheduler" : training_args.lr_scheduler_type,
                                }

        wandb.config.update(add_wandb_config)

if __name__ == "__main__":
    main()
