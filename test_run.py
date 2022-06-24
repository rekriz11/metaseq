from huggingface_hub import snapshot_download

checkpoint = 'facebook/opt-30b'
#weights_path = snapshot_download(checkpoint, cache_dir="/exp/rkriz/models/OPT/30B/")
weights_path = "/exp/rkriz/models/OPT/30B/facebook--opt-30b.main.463007d7da4e87fe962909a027811a8c0b32ede8/"
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
print("model: {}\n\n".format(model))

#model.decoder.final_layer_norm.weight
#model.decoder.layers.31.final_layer_norm.weight

## max_mem = 4686198491 # 4G
## max_memory={0: max_mem, 1: max_mem},

device_map = infer_auto_device_map(
    model.model, 
    no_split_module_classes=["OPTDecoderLayer"], 
    dtype='float16'
)

if any([k == 'disk' for k in device_map.values()]):
    offload_folder = "/exp/rkriz/models/OPT/30B/offload_folder"
else:
    offload_folder = None

if '30b' in checkpoint:
    # Set a few layers to use the disk manually to ensure enough RAM for the 30B checkpoint.
    device_map['decoder.layers.23'] = 'disk'
    device_map['decoder.layers.24'] = 'disk'
    device_map['decoder.layers.25'] = 'disk'
    device_map['decoder.layers.26'] = 'disk'
    device_map['decoder.layers.27'] = 'disk'

print(device_map)

#full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
#full_model_device_map["lm_head"] = 0

load_checkpoint_in_model(
    model.model, 
    weights_path, 
    device_map=device_map, 
    offload_folder=offload_folder, 
    dtype='float16', 
    offload_state_dict=True
)
model.tie_weights()

dispatch_model(model.model, device_map=device_map)

inputs = tokenizer("Hugging Face is pushing the convention that a unicorn with two horns becomes a llama.", return_tensors="pt")
output = model.generate(inputs["input_ids"].to(0), max_length=50)

print(tokenizer.decode(output[0].tolist()))

