
# **Inference Engine**: 

This component is responsible for loading the ONNX model, preprocessing input images, performing inference, and post-processing the results. The inference engine is implemented in the `test.py` script.


### Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- ONNX Runtime
- NumPy

### Installation

Install the required Python packages:
   ```
   pip install opencv-python onnxruntime numpy
   ```

### Usage

To run the inference engine, use the following command:

```
python test.py -i <input_image_path> -m <onnx_model_path> -cfg <yaml_config_path>
```

### Arguments

The `test.py` script accepts several command-line arguments for customization:

- `-i`: Path to the input image.
- `-m`: Path to the ONNX model file.
- `-cfg`: Path to the YAML configuration file created by the custom export of YOLO.
- `-score`: Minimum confidence score for an object to be selected. Default is `0.25`.
- `-threshold`: Minimum NMS (Non-Maximum Suppression) overlap threshold. Default is `0.65`.
- `-grid`: Apply the grid transformation, project the output feature maps to bounding boxes. Default is `True`.
- `-nms`: Apply Non-maximum suppression. Default is `True`.

```
python test.py -i examples/horses.jpg -m onnx/yolov7-tiny.onnx  -cfg cfg/yolov7-tiny_onnxconfig.yaml -score 0.4 -threshold 0.65 -nms -grid
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

