from huggingface_hub import snapshot_download
checkpoint = 'facebook/opt-125m'
#weights_path = snapshot_download(checkpoint, cache_dir="/exp/rkriz/models/OPT/125M_hf/")
weights_path = '/exp/rkriz/models/OPT/125M_hf/facebook--opt-125m.main.934b6a077313f3ee660a918a95313f5d0b136c5a/'

# If the folder contains a checkpoint that isn't sharded, it needs to point to the state dict directly
# otherwise point to the directory containing the shard
import os
files = os.listdir(weights_path)
weights_path = os.path.join(weights_path, 'pytorch_model.bin') if 'pytorch_model.bin' in files else weights_path

from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

config = AutoConfig.from_pretrained(checkpoint)

# Initializes an empty shell with the model. This is instant and does not take any RAM.
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
# Initialize the model under the previous context manager breaks the tied weights.
model.tie_weights()

# Infer device map automatically
device_map = infer_auto_device_map(model.model, no_split_module_classes=["OPTDecoderLayer"], dtype='float16')

if any([k == 'disk' for k in device_map.values()]):
    offload_folder = 'offload_folder'
else:
    offload_folder = None

if '30b' in checkpoint:
    # Set a few layers to use the disk manually to ensure enough RAM for the 30B checkpoint.
    device_map['decoder.layers.23'] = 'disk'
    device_map['decoder.layers.24'] = 'disk'
    device_map['decoder.layers.25'] = 'disk'
    device_map['decoder.layers.26'] = 'disk'
    device_map['decoder.layers.27'] = 'disk'

device_map

load_checkpoint_and_dispatch(
    model.model, 
    weights_path, 
    device_map=device_map, 
    offload_folder=offload_folder, 
    dtype='float16', 
    offload_state_dict=True
)
model.tie_weights()

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
inputs = tokenizer("Hugging Face is pushing the convention that a unicorn with two horns becomes a llama.", return_tensors="pt")

output = model.generate(inputs["input_ids"].to(0), max_length=10, do_sample=True)

print(tokenizer.decode(output[0].tolist()))

