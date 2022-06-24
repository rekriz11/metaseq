from huggingface_hub import snapshot_download

checkpoint = 'facebook/opt-2.7b'
weights_path = snapshot_download(checkpoint, cache_dir="/exp/rkriz/models/OPT/2.7B/")
import os
files = os.listdir(weights_path)
weights_path = os.path.join(weights_path, 'pytorch_model.bin') if 'pytorch_model.bin' in files else weights_path

from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch, load_checkpoint_in_model, dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
model.tie_weights()

max_mem = 4686198491 # 4G

device_map = infer_auto_device_map(
    model.model, 
    max_memory={0: max_mem, 1: max_mem},
    no_split_module_classes=["OPTDecoderLayer"], 
    dtype='float16'
)

print(device_map)

load_checkpoint_and_dispatch(
    model.model, 
    weights_path, 
    device_map=device_map, 
    offload_folder=None, 
    dtype='float16', 
    offload_state_dict=True
)
model.tie_weights()

inputs = tokenizer("Hugging Face is pushing the convention that a unicorn with two horns becomes a llama.", return_tensors="pt")
output = model.generate(inputs["input_ids"].to(0), max_length=50, do_sample=True)

print(tokenizer.decode(output[0].tolist()))

