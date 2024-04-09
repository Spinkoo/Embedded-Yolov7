from onnx_inference_engine import onnx_model
import cv2
import numpy as np
import argparse


def get_parser():
    parser = argparse.ArgumentParser(
                    prog='Inference engine',
                    description='Perform a forward pass using YOLO onnx ensued by postprocess and cleaning operations',
                    )
    
    parser.add_argument('-i', '--input', required=True, help='Provide the onnx pretrained model path')
    parser.add_argument('-m', '--onnx_model_path', required=True, help='Provide the path to the onnx file of the model')  
    parser.add_argument('-cfg', '--yaml_path', help='Provide the config yaml file created by the custom export of YOLO')  
    parser.add_argument('-score', '--score', type=float, default=0.25, help='Min confidence score for an object to be selected [0-1]')  
    parser.add_argument('-threshold', '--threshold', type = float, default=0.65, help='Min NMS overlap thershold [0-1]')  
    
    parser.add_argument('-grid', '--grid', action='store_true', help='Apply the grid transformation, project the output feature maps to bounding boxes')
    parser.add_argument('-nms', '--nms', action='store_true', help='Apply Non-maximum suppression')

    return parser

def load_image(path):
    return  np.expand_dims(cv2.imread(path), axis = 0)

if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()

    input = args.input
    model_path = args.onnx_model_path
    yaml_path = args.yaml_path
    grid =  args.grid
    nms = args.nms
    score_threshold=args.score
    nms_threshold=args.threshold
    
    model = onnx_model(onnx_path = model_path, yaml_path=yaml_path, score_threshold = score_threshold, nms_threshold=nms_threshold)
    im = load_image(input)
    op = model.predictions(im, nms=nms, grid = grid, )

    print(op)
