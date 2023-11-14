import os
import torch
import torch.nn as nn
from datasets import load_dataset
import datasets
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence, Union, List
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments, Seq2SeqTrainingArguments, BitsAndBytesConfig
from transformers import DataCollatorWithPadding, BatchEncoding
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import bitsandbytes as bnb
import copy
from peft import LoraConfig,get_peft_model
transformers.Trainer
datasets.utils.logging.set_verbosity(datasets.logging.DEBUG)
IGNORE_INDEX = -100

       
def prepare_model_for_training(
        model: PreTrainedModel,
        output_embedding_layer_name: Optional[str] = "lm_head",
        use_gradient_checkpointing: Optional[bool] = True,
        layer_norm_names: Optional[List[str]] = ["norm", "ln_f"] # for LLaMA and BLOOM setting
) -> PreTrainedModel:

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):

            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bit if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main(args, train_args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    else:
        device_map = {"": 0}
    

    config_kwargs = {}

    if args.bits == 8:
        config_kwargs["load_in_8bit"] = True
        config_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif args.bits == 4:
        config_kwargs["load_in_4bit"] = True
        config_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        pass

    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name_or_path, padding_side="left"
    )
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    from data_loader.d2p_dataset import d2pDataset, DynamicDataCollatorWithPadding
    dataset = load_dataset("json", data_files={"train": args.data + "all_prompts_train.jsonl"})
    train_data = dataset['train']
    train_dataset = d2pDataset(train_data, tokenizer, args.max_length)
    data_collator = DynamicDataCollatorWithPadding(
        tokenizer=tokenizer,
    )
        
    # Quant
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path , use_auth_token=hub_token, use_safetensors=False, device_map=device_map, torch_dtype=torch.float16, **config_kwargs)
    model = prepare_model_for_training(model) 
   
    # # Lora
    lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=find_all_linear_names(args, model.model),
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM")

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=train_args,
        data_collator=data_collator
    )

    trainer.train()
    
    model.save_pretrained(train_args.output_dir + args.data_type)

if __name__ == "__main__":
    
    @dataclass
    class MyArguments():
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
        """
        model_name_or_path: str = field(
            default= "meta-llama/Llama-2-7b-hf",
            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
        )
        model_type: Optional[str] = field(
            default="llama-2-7b",
        )
        
        data: Optional[str] = field(
            default="your-data-dir",
        )
        data_type: Optional[str] = field(
            default="d2p",
        )
        
        
        max_length: Optional[int] = field(
            default=768,
            metadata={"help": "The maximum total input sequence length after tokenization."}
        )

        bits: Optional[int] = field(
            default=16,
            metadata={"help": "The maximum total input sequence length after tokenization."}
        )

    parser = HfArgumentParser((MyArguments, Seq2SeqTrainingArguments))
    args, train_args = parser.parse_args_into_dataclasses()
    print(train_args)
    main(args, train_args)