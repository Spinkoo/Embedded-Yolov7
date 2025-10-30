# ONNX Model Quantization

## Overview

This component provides tools for model quantization and reduction, enabling faster inference and deployment on resource-constrained devices. The quantization process converts models from floating-point (float32) to integer (int8) format using QDQ (Quantize-DequantizeLinear) quantization.

This approach allows models to become lighter and deployable on low-memory devices while maintaining acceptable accuracy.

## What is Quantization?

Quantization reduces model size by converting 32-bit floating-point weights and activations to 8-bit integers. This:
- Reduces model file size (typically 4x)
- Speeds up inference
- Reduces memory requirements
- Maintains reasonable accuracy

## Features

- ✅ QDQ Quantization (float32 → int8)
- ✅ Automatic quantization parameter estimation from sample images
- ✅ Support for YOLOv7 and custom ONNX models
- ✅ Configurable sample size for calibration

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- ONNX Runtime (1.15.0 or compatible)
- NumPy
- PyTorch

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python onnxruntime==1.15.0 numpy torch
```

## Usage

### Basic Command

```bash
python onnx_quant.py -i <input_onnx> -o <output_file_name> -imgs <images_folder> imgsz <size> --nb_samples <count>
```

### Example

Quantize YOLOv7-tiny using sample images:

```bash
python onnx_quant.py -i yolov7-tiny.onnx -o qyolov7-tiny.onnx -imgs images/ imgsz 640 --nb_samples 15
```

## Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `-i` | Path to the input ONNX model file |
| `-o` | Output filename for the quantized model |
| `-imgs` | Path to folder containing sample images for quantization calibration |
| `imgsz` | Input image size for the model |
| `--nb_samples` | Number of sample images to use for parameter estimation |

## How Quantization Works

1. **Calibration**: Uses sample images to estimate optimal quantization parameters
2. **Quantization**: Converts model weights and activations to int8 format
3. **Output**: Generates a quantized ONNX model with QDQ operators

The quantized model maintains the same architecture but with reduced precision, achieving significant size reduction with minimal accuracy loss.

## Output

After quantization, you'll have a new ONNX model file (e.g., `qyolov7-tiny.onnx`) that is significantly smaller than the original and can be used for inference.

## Using Quantized Models

The quantized model can be used with the inference engines such as the [onnx-python-inference](../onnx-python-inference/) module or deployed on STM32 devices using the STM32AI toolbox.

## Important Notes

- The custom exported models do not contain the NMS layer or the final transpose layer because **STM32AI does not support NMS operations or 5-dimensional operations**
- These operations are handled by the inference engine post-processing

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. 