# mitigating-reversal-curse
This is the code for paper 'Are We Falling in a Middle-Intelligence Trap? An Analysis and Mitigation of the Reversal Curse'

## Prepare data
Download the data folder from [Berglund et al.](https://github.com/lukasberglund/reversal_curse/tree/main).

## Enviornment
```
pytorch=1.13.0
bitsandbytes=0.41.1
peft=0.5.0 
```
Install the [transformers](https://github.com/huggingface/transformers/tree/v4.35.0) from source, and replace the 'src/transformers/models/llama/modeling_llama.py' with the code in our 'transformers' folder.

## Train the model
Fine-tune Llama with NTP:
```
bash run.sh your-data-dir/reverse_experiments/june_version_7921032488/ your-model-save-dir your-llama-hf-save-dir
```

Fine-tune Llama with BICO:
```
bash run_mlm.sh your-data-dir/reverse_experiments/june_version_7921032488/ your-model-save-dir your-llama-hf-save-dir
```

## Generate and Evaluate
```
python generate.py 
```
