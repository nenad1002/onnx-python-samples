import onnx
import numpy as np
from onnx import TensorProto, helper, numpy_helper

def get_dims(tensor):
    """Extract dimensions from a tensor’s type information.
       Works for both concrete and symbolic dimensions."""
    dims = []
    if tensor.type and tensor.type.HasField("tensor_type") and tensor.type.tensor_type.HasField("shape"):
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(dim.dim_value)
            elif dim.HasField("dim_param") and dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append(0)  # Fallback if no information is available
    return dims

def update_consumers(old_name, new_name, model):
    """Replace all occurrences of old_name in node inputs and graph outputs with new_name."""
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == old_name:
                node.input[i] = new_name
    for output in model.graph.output:
        if output.name == old_name:
            output.name = new_name

def find_tensor_info(tensor_name, model):
    """Search graph inputs, outputs, and value_info for tensor shape information."""
    for t in model.graph.input:
        if t.name == tensor_name:
            return t
    for t in model.graph.output:
        if t.name == tensor_name:
            return t
    for t in model.graph.value_info:
        if t.name == tensor_name:
            return t
    return None

def cast_initializer_to_bf16(initializer):
    """
    Convert a FLOAT16 initializer to BFLOAT16.
    The conversion is done by:
      1. Converting the float16 data to float32.
      2. Viewing the float32 data as uint32.
      3. Shifting right 16 bits to drop the lower mantissa bits.
      4. Casting the result to uint16.
    Instead of using raw_data, we pass the full list of converted BF16 values.

    TODO: Rewrite this function as it might be wrong
    """
    arr = numpy_helper.to_array(initializer)            # np.float16 array.
    arr32 = arr.astype(np.float32)                        # Convert to float32.
    bits = arr32.view(np.uint32)                          # Get the underlying bits.
    bf16_bits = (bits >> 16).astype(np.uint16)            # Upper 16 bits become BF16.
    new_initializer = helper.make_tensor(
        name=initializer.name,
        data_type=TensorProto.BFLOAT16,
        dims=initializer.dims,
        vals=bf16_bits.tolist()  # Use the full list of BF16 values.
    )
    return new_initializer

# ---------------------------------------------------------------------
# Load the ONNX model.
# ---------------------------------------------------------------------
model = onnx.load('model.onnx')
target_node_type = "SkipSimplifiedLayerNormalization"
target_nodes = [node for node in model.graph.node if node.op_type == target_node_type]

if not target_nodes:
    print(f"No nodes with op_type '{target_node_type}' were found in the model.")
else:
    # -----------------------------------------------------------------
    # INPUT CASTING SECTION: FLOAT16 -> BF16
    # -----------------------------------------------------------------
    cast_input_nodes = []
    for node in target_nodes:
        print(f"Processing node: {node.name} of type: {node.op_type} for input casting")
        new_inputs = list(node.input)
        for idx, input_name in enumerate(node.input):
            found = False
            tensor_info = find_tensor_info(input_name, model)
            if tensor_info is not None:
                found = True
                dims = get_dims(tensor_info)
                if tensor_info.type.tensor_type.elem_type == TensorProto.FLOAT16:
                    cast_output_name = input_name + "_cast_to_bf16"
                    print(f" - Found input '{input_name}' (shape {dims}) of type FLOAT16. Casting to BF16 -> '{cast_output_name}'.")
                    cast_node = helper.make_node(
                        "Cast",
                        inputs=[input_name],
                        outputs=[cast_output_name],
                        name=f"Cast_{input_name}_to_BF16",
                        to=TensorProto.BFLOAT16
                    )
                    cast_input_nodes.append(cast_node)
                    new_inputs[idx] = cast_output_name
                    cast_value_info = helper.make_tensor_value_info(cast_output_name, TensorProto.BFLOAT16, dims)
                    model.graph.value_info.append(cast_value_info)
            if not found:
                print(f" - Warning: No type information found for input '{input_name}'.")
        node.input[:] = new_inputs

    #original_nodes = list(model.graph.node)
    #del model.graph.node[:]
    model.graph.node.extend(cast_input_nodes)

    # -----------------------------------------------------------------
    # OUTPUT CASTING SECTION: BF16 -> FLOAT16
    # -----------------------------------------------------------------
    # BEFORE adding the cast nodes, update each SkipSimplifiedLayerNormalization node's output
    # to have type BF16 so that the cast node gets BF16 input.
    for node in target_nodes:
        for orig_output in node.output:
            if orig_output == "":
                continue
            tensor_info = find_tensor_info(orig_output, model)
            if tensor_info is not None:
                # Set the element type to BF16.
                tensor_info.type.tensor_type.elem_type = TensorProto.BFLOAT16
                dims = get_dims(tensor_info)
            else:
                # this will not happen for our case when we use SkipSimplifiedLayerNorm, but I'll add it here just in case
                dims = []  # or provide default dims if available
                # If no value info exists, add one for the output with BF16 type.
                new_value_info = helper.make_tensor_value_info(orig_output, TensorProto.BFLOAT16, dims)
                model.graph.value_info.append(new_value_info)

    cast_output_nodes = []
    for node in target_nodes:
        print(f"Processing node: {node.name} of type: {node.op_type} for output casting")
        for output_idx, orig_output in enumerate(node.output):
            if orig_output == "":
                continue
            new_output = orig_output + "_cast_to_float16"
            # Retrieve updated shape info (should now be BF16) for the output.
            tensor_info = find_tensor_info(orig_output, model)
            dims = get_dims(tensor_info) if tensor_info is not None else None
            print(f" - Processing output '{orig_output}' (expected BF16). Adding cast to FLOAT16 -> '{new_output}', shape: {dims}")
            cast_node_out = helper.make_node(
                "Cast",
                inputs=[orig_output],   # This input is now BF16.
                outputs=[new_output],
                name=f"Cast_{orig_output}_to_FLOAT16",
                to=TensorProto.FLOAT16
            )
            print(f"New Cast node: Cast_{orig_output}_to_FLOAT16")
            cast_output_nodes.append(cast_node_out)
            if dims is not None:
                cast_value_info = helper.make_tensor_value_info(new_output, TensorProto.FLOAT16, dims)
                model.graph.value_info.append(cast_value_info)

            update_consumers(orig_output, new_output, model)
    model.graph.node.extend(cast_output_nodes)

    # -----------------------------------------------------------------
    # INITIALIZER CASTING SECTION: FLOAT16 -> BF16 for initializers.
    # -----------------------------------------------------------------
    target_initializer_names = set()
    for node in target_nodes:
        for inp in node.input:
            target_initializer_names.add(inp)
    new_initializers = []
    for initializer in model.graph.initializer:
        if initializer.name in target_initializer_names and initializer.data_type == TensorProto.FLOAT16:
            print(f" - Casting initializer '{initializer.name}' from FLOAT16 to BF16.")
            new_init = cast_initializer_to_bf16(initializer)
            new_initializers.append(new_init)
        else:
            new_initializers.append(initializer)
    # Replace the initializer container with the new initializers.
    model.graph.initializer.clear()
    model.graph.initializer.extend(new_initializers)

# ---------------------------------------------------------------------
# Optionally run shape inference (guarded against errors or an empty graph)
# ---------------------------------------------------------------------
try:
    inferred_model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    if inferred_model.graph.node:
        print("Shape inference was successful.")
        model = inferred_model
    else:
        print("Warning: Shape inference returned an empty node list. Skipping shape inference.")
except Exception as e:
    print(f"Shape inference failed with error: {e}. Skipping shape inference.")

# ---------------------------------------------------------------------
# Save the modified model.
# ---------------------------------------------------------------------
onnx.save(model, 'mm8.onnx')
print("Model has been saved as 'mm8.onnx'.")

# Validate the saved model.
try:
    onnx.load('mm8.onnx')
    print("Model validation passed.")
except Exception as e:
    print(f"Model validation failed: {e}")
