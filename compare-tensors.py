import torch
from onnx import TensorProto
from google.protobuf import text_format
import numpy as np

# Helper function to load an onnx tensor
def load_tensor_proto(path, text_dump: bool = False):
    tp = TensorProto()
    raw = open(path, "rb").read()
    if text_dump:
        text_format.Merge(raw.decode("utf-8"), tp)
    else:
        tp.ParseFromString(raw)
    return tp

# Helper function to convert an onnx tensor to torch for bf16
def tensor_proto_to_torch_for_bf16(tp: TensorProto):
    """Convert a BFLOAT16 TensorProto into a torch.Tensor without NumPy."""
    if tp.data_type != TensorProto.BFLOAT16:
        raise ValueError(f"Unsupported data type: {tp.data_type}. Expected BFLOAT16.")
    # tp.raw_data is a bytes object where each element is a 2‑byte bfloat16
    # Interpret it first as uint16 (same size), then reinterpret the bit‐pattern as bfloat16:
    uint16 = torch.frombuffer(tp.raw_data, dtype=torch.bfloat16)
    uint16 = uint16.reshape(tuple(tp.dims))
    return uint16.view(torch.bfloat16)

a = torch.load("logits1.pt", map_location="cpu")
b = torch.load("logits2.pt", map_location="cpu")

# You can view a tensor differently
#a = a.view(1, 4, 256)

# Or squeeze it.
#b = b.squeeze(2)                               # now [1, 8, 256]

assert a.shape == b.shape, f"shapes differ: {a.shape} vs {b.shape}"
assert a.dtype == b.dtype, f"dtypes differ: {a.dtype} vs {b.dtype}"

print("Exactly equal? ", torch.equal(a, b))

# Numeric tolerance check
tol = 1e-6
print("Allclose (atol=1e-6)? ", torch.allclose(a, b, atol=tol))

# Error metrics
diff = (a - b).abs()
print("Max abs diff:", diff.max().item())
print("Mean abs diff:", diff.mean().item())
print("sum of diff", diff.sum())