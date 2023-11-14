import os
import torch
import torch.nn as nn
from datasets import load_dataset
import datasets
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any, Tuple
from collections.abc import Mapping
import transformers
from transformers import LlamaForMLM, LlamaForCausalLM, LlamaTokenizer, HfArgumentParser, TrainingArguments, Seq2SeqTrainingArguments, BitsAndBytesConfig
from transformers import DataCollatorWithPadding, BatchEncoding
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import bitsandbytes as bnb
import copy
import types
from peft import LoraConfig,get_peft_model

       
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
            lora_module_names.add(name)
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def new_torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
    # Handle dict or lists with proper padding and conversion to tensor.

    if isinstance(examples[0], Mapping):
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
    else:
        batch = {
            "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        }

    # If special token mask has been preprocessed, pop it from the dict.
    special_tokens_mask = batch.pop("special_tokens_mask", None)
    if self.mlm:
        batch["input_ids"], batch["labels"], masked_indices = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        if masked_indices is not None:
            batch['attention_mask'] = batch['attention_mask'] & ~masked_indices
    else:
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
    return batch

def new_torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    import torch, random

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, self.mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()
    
    do_mlm = random.choice([True, False])
    if do_mlm:
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        return inputs, (labels, do_mlm), masked_indices
    else:
        labels = torch.where(labels == 0,-100,labels)
        return inputs, (labels, do_mlm), None


def main(args, train_args):
    hub_token = "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"
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
    tokenizer.mask_token = tokenizer.pad_token
    # input(tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
    
    from transformers import DataCollatorForLanguageModeling
    from data_loader.d2p_dataset_mlm import d2pDataset
    
    dataset = load_dataset("json", data_files={"train": args.data + "all_prompts_train.jsonl"})
    
    train_data = dataset['train']
    train_dataset = d2pDataset(train_data, tokenizer, args.max_length)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15,
    )
    
    data_collator.torch_call = types.MethodType(new_torch_call, data_collator)
    data_collator.torch_mask_tokens = types.MethodType(new_torch_mask_tokens, data_collator)
    
    model = LlamaForMLM.from_pretrained(args.model_name_or_path, use_auth_token=hub_token, use_safetensors=False, device_map=device_map, torch_dtype=torch.float16, **config_kwargs)
    
    model = prepare_model_for_training(model) 

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
        )

        bits: Optional[int] = field(
            default=16,
        )
      
    parser = HfArgumentParser((MyArguments, Seq2SeqTrainingArguments))
    args, train_args = parser.parse_args_into_dataclasses()
    print(train_args)
    main(args, train_args)
