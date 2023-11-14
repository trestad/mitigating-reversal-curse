from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch
from peft import PeftModel
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu

def evaluate_completion(
    completion: str,
    target: str,
    case_sensitive: bool = False,
) -> bool:
    """Evaluate completion using exact-match vs the target.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    """
    target = target.strip()
    test_str = completion.strip()
    test_str = test_str.lower() if not case_sensitive else test_str
    target_str = target.lower() if not case_sensitive else target
    return target_str in test_str

def compute_bleu(run_results):
    all = 0
    completions =  run_results['completions']
    targets = run_results['targets']
    for completion, target in zip(completions, targets):
        bleu_score = sentence_bleu([target], completion['gen_answer'])
        all += bleu_score
    return all / len(completions)

def compute_metric(run_results):
    """Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    """
    n_correct = 0
    is_correct_list = []
    completions =  run_results['completions']
    targets = run_results['targets']
    for completion, target in zip(completions, targets):
        correct = evaluate_completion(completion['gen_answer'], target)
        is_correct_list.append(correct)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    return accuracy

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    input_tokens = {k:input_tokens[k] for k in input_tokens if k in ["input_ids", "attention_mask"]}
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def main():
    for ckpt in [9000]:
        print(f'----------------------{ckpt}---------------------')
        lora_dir = f"your-lora-ckpt-dir/checkpoint-{ckpt}"
        model_name_or_path = 'your-llama-save-dir'
        data_path = "your-data-dir"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left",
        )
        tokenizer.pad_token_id = 0  
        tokenizer.bos_token_id = 1 
        tokenizer.eos_token_id = 2

        model = LlamaForCausalLM.from_pretrained(model_name_or_path, use_safetensors=False, device_map={"":0}, torch_dtype=torch.float16)
        
        if len(lora_dir) != 0:
            model = PeftModel.from_pretrained(model, lora_dir)
            model.merge_adapter()
        model.eval()
        
        for task in ['d2p','p2d','p2d_reverse','d2p_reverse']:
            print(f'----------------------{task}---------------------')
            output_name = f"{ckpt}-{task}.json"
            dataset = load_dataset("json", data_files={"eval":  data_path + f"{task}_prompts_test.jsonl"})
            val_data = dataset['eval']
            prompts = [d['prompt'] for d in val_data]
            completions = [d['completion'] for d in val_data]

            results = []
            batch_size = 10
            for batch_input in tqdm(batch_split(prompts, batch_size)):
                encode_inputs = prepare_input(tokenizer, batch_input)
                with torch.no_grad():
                    output = model.generate(**encode_inputs, max_new_tokens=40, pad_token_id=tokenizer.pad_token_id, do_sample=False, num_beams=1,
                                        eos_token_id= tokenizer.eos_token_id,)
                
                response = tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)
            
                for idx, p in enumerate(batch_input):
                    results.append({'question': p, 'gen_answer': response[idx][len(p):]})
            run_results = {'completions':results, 'targets':completions}
            with open(output_name, 'w', encoding='utf-8') as f:
                json.dump(run_results, f, ensure_ascii=False, indent=2)
            if 'd2p.json' in output_name or 'p2d_reverse' in output_name:
                print('Using EM')
                metrics = compute_metric(run_results)
                print(metrics)
            else:
                print('Using BLEU')
                metrics = compute_bleu(run_results)
                print(metrics)

if __name__ == '__main__':
    main()