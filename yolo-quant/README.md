# **ONNX Model Conversion and Quantization** :

This involves converting a YOLO model to ONNX format and optionally quantizing it to reduce model size and improve inference speed. The conversion and quantization process is handled by the `onnx_quant.py` script.

### Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- ONNX Runtime
- NumPy
- Torch

### Installation

Install the required Python packages:
   ```
   pip install opencv-python onnxruntime==1.15.0 numpy torch
   ```

### Usage :

This repo performs ONNX QDQ quantization (from float32 to int8) on the exported YOLO. This reduction allows for the model to become lighter and deployable on low-memory devices.

```
python onnx_quant.py -i <input_onnx> -o <output_file_name> -imgs <images folder for estimating quantization params>
```
```
python onnx_quant.py -i yolov7-tiny.onnx -o qyolov7-tiny.onnx -imgs images/ imgsz 640 --nb_samples 15
```
The quantized model can be used by inference engines such as [onnx-python-inference repo](../onnx-python-inference/)

#### Note : the customly exported model doesn't containt the NMS layer or the last transpose layer because the STM32ai tool doesn't support NMS or 5 dim operations
#### This should be handled by the inference engine subsequently 