import onnx
import onnxruntime as ort
import torch
import numpy as np
from onnx import TensorProto

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import onnxruntime as ort
print("ORT version:", ort.__version__)
print("Available providers:", ort.get_all_providers())

# Specify the model path
model_path = "<ONNX MODEL PATH>"

session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

onnx_model = onnx.load(model_path)
graph = onnx_model.graph

# Create an IO binding object to be able to support various input/output data types.
io_binding = session.io_binding()

print("Model inputs:")
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

# Define inference parameters for your model.
batch_size = 1
seq_length = 1          # Starting sequence length (e.g., only the prompt token)
hidden_size = 2560  
num_heads = 4
past_seq_length = 0     # No past tokens initially
head_size = 256
num_layers = 34

# Create input tensors using torch on CUDA.
device = torch.device("cuda:0")
inputs_embeds = torch.rand(batch_size, seq_length, hidden_size, device=device, dtype=torch.float16)

attention_mask = torch.ones(batch_size, seq_length, device=device, dtype=torch.int64)
position_ids = torch.arange(seq_length, device=device, dtype=torch.int64).unsqueeze(0)

# Create the past key/value pairs as empty tensors on CUDA.
past = {}
for layer in range(num_layers):
    key_name = f"past_key_values.{layer}.key"
    value_name = f"past_key_values.{layer}.value"
    past_key = torch.zeros(batch_size, num_heads, past_seq_length, head_size, device=device, dtype=torch.float16)
    past_value = torch.zeros(batch_size, num_heads, past_seq_length, head_size, device=device, dtype=torch.float16)
    past[key_name] = past_key
    past[value_name] = past_value

io_binding.bind_input("inputs_embeds", device_type="cuda", device_id=0,
                      shape=list(inputs_embeds.shape),
                      element_type=TensorProto.FLOAT16,
                      buffer_ptr=inputs_embeds.data_ptr())

io_binding.bind_input("attention_mask", device_type="cuda", device_id=0,
                      shape=list(attention_mask.shape),
                      element_type=TensorProto.INT64,
                      buffer_ptr=attention_mask.data_ptr())

io_binding.bind_input("position_ids", device_type="cuda", device_id=0,
                      shape=list(position_ids.shape),
                      element_type=TensorProto.INT64,
                      buffer_ptr=position_ids.data_ptr())

for name, tensor in past.items():
    io_binding.bind_input(name, device_type="cuda", device_id=0,
                          shape=list(tensor.shape),
                          element_type=TensorProto.FLOAT16,
                          buffer_ptr=tensor.data_ptr())

output_name = session.get_outputs()[0].name

output_shape = [1, 1, 262208]  # Adjust as needed
output_buffer = torch.ones(output_shape, dtype=torch.float16, device="cuda")

io_binding.bind_output(output_name, "cuda", 0, TensorProto.FLOAT16, output_buffer.shape, output_buffer.data_ptr())

session.run_with_iobinding(io_binding)

# Copy the output from the GPU back to the CPU
torch.save(output_buffer, "logits1.pt")