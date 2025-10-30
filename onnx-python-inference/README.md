# ONNX Python Inference Engine

## Overview

This component provides a complete inference engine for performing object detection using YOLOv7 models in ONNX format. It handles loading the ONNX model, preprocessing input images, running inference, and post-processing results to produce bounding boxes and confidence scores.

The inference engine is implemented in Python using ONNX Runtime for efficient model execution.

## Features

- ✅ ONNX model loading and inference
- ✅ Image preprocessing and normalization
- ✅ Grid-based bounding box transformation
- ✅ Non-Maximum Suppression (NMS) filtering
- ✅ Configurable confidence and NMS thresholds
- ✅ YAML-based model configuration

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- ONNX Runtime
- NumPy

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python onnxruntime numpy
```

## Usage

### Basic Command

To run the inference engine on an image:

```bash
python test.py -i <input_image_path> -m <onnx_model_path> -cfg <yaml_config_path>
```

### Example

```bash
python test.py -i examples/horses.jpg -m onnx/yolov7-tiny.onnx -cfg cfg/yolov7-tiny_onnxconfig.yaml -score 0.4 -threshold 0.65 -nms -grid
```

## Command-Line Arguments

The `test.py` script accepts the following arguments for customization:

| Argument | Description | Default |
|----------|-------------|---------|
| `-i` | Path to the input image | Required |
| `-m` | Path to the ONNX model file | Required |
| `-cfg` | Path to the YAML configuration file | Required |
| `-score` | Minimum confidence score threshold | `0.25` |
| `-threshold` | Minimum NMS (Non-Maximum Suppression) overlap threshold | `0.65` |
| `-grid` | Apply grid transformation to feature maps | `True` |
| `-nms` | Apply Non-Maximum Suppression filtering | `True` |

## How It Works

1. **Model Loading**: Loads the ONNX model using ONNX Runtime
2. **Image Preprocessing**: Reads and normalizes the input image
3. **Inference**: Runs the model on the preprocessed image
4. **Grid Transformation**: Projects feature maps to bounding boxes (if enabled)
5. **Post-Processing**: Applies NMS to filter overlapping detections
6. **Output**: Returns bounding boxes with class labels and confidence scores

## Model Format

This inference engine is designed to work with YOLOv7 models exported using the custom ONNX export script from the `yolov7/` module. The model configuration is provided via a YAML file (e.g., `yolov7-tiny_onnxconfig.yaml`).

**Note**: The exported models do not include the NMS layer or final transpose operation, as these are not supported by STM32AI. The inference engine handles these operations.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

