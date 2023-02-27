import torch
import onnx
import onnxruntime as ort

# Load PyTorch model
model = torch.load('models\CNNLSTM.pth')

# Set model to evaluation mode
model.eval()

# Create input tensor
dummy_input = torch.randn(1, 1, 30, 6)

# Export PyTorch model to ONNX
input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)

# Load ONNX model
onnx_model = onnx.load("model.onnx")

# Create Inference session using ONNX model
ort_session = ort.InferenceSession("model.onnx")

# Get input and output names
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# Create input tensor
input_tensor = dummy_input.numpy()

# Run inference
outputs = ort_session.run([output_name], {input_name: input_tensor})

# Print outputs
print(outputs)
