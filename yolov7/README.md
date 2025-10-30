# YOLOv7 Training and ONNX Export

## Overview

This folder contains a fork of the [Official YOLOv7](https://github.com/WongKinYiu/yolov7) repository with custom extensions for ONNX model export optimized for embedded systems.

The key enhancement is a **custom ONNX export** that removes unsupported operations (NMS and 5-dimensional operations) for compatibility with STM32AI and embedded inference engines.

## What's Different

- ✅ Custom ONNX export without NMS layer (not supported by STM32AI)
- ✅ Removes final transpose operation (5-D operations not supported by STM32AI)
- ✅ Generates accompanying YAML configuration file with model metadata
- ✅ Optimized for subsequent quantization and embedded deployment

## Key Features

- Full YOLOv7 training pipeline
- Model configuration files for various model sizes (tiny, base, large, etc.)
- Custom ONNX export specifically designed for embedded systems
- Support for both object detection and training workflows

## Installation

See the official YOLOv7 documentation for training setup. Install dependencies:

```bash
pip install -r requirements.txt
```

## Custom ONNX Export

### Purpose

The custom export creates ONNX models without post-processing layers that are unsupported on embedded platforms:

- **NMS Removal**: Non-Maximum Suppression is handled by the inference engine
- **Transpose Removal**: Eliminates 5-dimensional operations not supported by STM32AI
- **Config Generation**: Creates a YAML file with model metadata

### Basic Usage

Export a trained YOLOv7 model to ONNX format:

```bash
python export.py --weights yolov7-tiny.pt --simplify --img-size 640 640 --max-wh 640 --custom_export
```

### Export Command Parameters

| Parameter | Description |
|-----------|-------------|
| `--weights` | Path to the trained model weights (.pt file) |
| `--simplify` | Simplify the ONNX model (recommended) |
| `--img-size` | Input image dimensions (height width) |
| `--max-wh` | Maximum bounding box width/height |
| `--custom_export` | Enable custom export for embedded systems |

### Output Files

The export command generates two files:

1. **ONNX Model** (e.g., `yolov7-tiny.onnx`)
   - Compiled neural network in ONNX format
   - Optimized for embedded inference

2. **Config YAML** (e.g., `yolov7-tiny_onnxconfig.yaml`)
   - Contains model metadata and hyperparameters
   - Used by inference engines for proper model operation

## Workflow

Typical workflow for preparing models for embedded deployment:

```
1. Train Model (this folder)
   ↓
2. Export to ONNX (custom export)
   ↓
3. Quantize Model (yolo-quant/)
   ↓
4. Run Python Inference (onnx-python-inference/)
   ↓
5. Deploy on STM32 (stm32_toolbox/)
```

## Generated Configuration File

The YAML configuration file contains critical information about your model such as:
- Input/output specifications
- Normalization parameters
- Model architecture metadata
- Any custom inference requirements

This file must be provided when running inference on the exported model.

## Model Variants

The configs folder contains configurations for various YOLOv7 model variants:
- **Training configs** (`training/`): For model training
- **Deployment configs** (`deploy/`): Optimized for inference
- **Baseline configs** (`baseline/`): Reference configurations

## Next Steps

After exporting your model:

1. **Quantize** using the [yolo-quant](../yolo-quant/) module to reduce model size
2. **Test** using the [onnx-python-inference](../onnx-python-inference/) module
3. **Deploy** on STM32 using the [stm32_toolbox](../stm32_toolbox/) module

## Contributing

This is a fork with custom modifications. Contributions to the custom export functionality are welcome!

## License

This project is based on the official YOLOv7 repository. See the LICENSE.md file for details.

## References

- [Official YOLOv7 Repository](https://github.com/WongKinYiu/yolov7)
- [YOLOv7 Paper](https://arxiv.org/abs/2207.02696)
