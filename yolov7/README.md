# Official YOLOv7


- This repo is a fork of the [Official YOLOv7](https://github.com/WongKinYiu/yolov7). 
- This extension allows for custom ONNX export of the model, without NMS and the final transpose layer as it's a 5-D operation and they are not supported yet by STM32ai and are superfluous in the quantization process
- These calculations will be handeled later on by the [onnx-python-inference repo](../onnx-python-inference/)

```
python export.py --weights  yolov7-tiny.pt --simplify --img-size 640 640 --max-wh 640 --custom_export
```
This command should generated the exported onnx model plus a config yaml file that contains informations about the model (```yolov7-tiny.onnx``` and ```yolov7-tiny_onnxconfig.yaml``` in this example)