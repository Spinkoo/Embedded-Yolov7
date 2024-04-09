import onnx

# Load the ONNX model
onnx_model_path = "onnx/yolov7-tiny.onnx"
model = onnx.load(onnx_model_path)

inferred_model = onnx.shape_inference.infer_shapes(model)
print(inferred_model.initializer)
# Access output shapes
output_shapes = [output.shape for output in inferred_model.output]

# Print the inferred output shapes (e.g., [(batch_size, channel, height, width)])
print(output_shapes)
exit()
for node in model.graph.node[::-1]:  # Iterate in reverse order
    print(f"Node Name: {node.name}")
    print(f"Op Type: {node.op_type}")
    print(f"Input Tensors: {node.input}")
    print(f"Output Tensors: {node.output}")

    # Access intermediate layer outputs based on your conditions
    # For example, if the node represents a Convolution operation, you might want its output
    
    if node.op_type == 'Conv':
        intermediate_layer_output_name = node.output[0]
        print(node.output[0])
        # You can use this name to fetch the output later during inference

    print("\n")