from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, QuantFormat
from quantize import quantize, QuantizationMode
from onnxruntime.quantization.calibrate import CalibrationDataReader
import torch
import argparse 
import cv2
from glob import glob
import numpy as np
from random import shuffle


def get_parser():
    parser = argparse.ArgumentParser(
                    prog='Quantizer',
                    description='Quantize the input onnx model -only convolution layers for now- into uint8 ops',
                    )
    parser.add_argument('-i', '--input', required=True, help='Provide the onnx pretrained model path')
    parser.add_argument('-o', '--output', required=True, help='Provide the output path for your quantized model')
    parser.add_argument('-imgs', '--images', required=True, help='Provide the path to the images folder to calibrate the quantization')  
    parser.add_argument('-imgsz', '--image_size',default= 640, type=int, required=False, help='Input image size (must be similair to the onnx model input size)')  
    parser.add_argument('-bs', '--batch_size',default= 1, type=int, required=False, help='Batch size to perform the calibration')  

    parser.add_argument('-ns', '--nb_samples',  help='Number of samples taken out of your dataset', type= int, default=15)  

    return parser
    
    

""" 
Dataloader that provide images into the quantization process in order to calculate the required constants (calibration w.r.t to our dataset)
"""
class DataReader(CalibrationDataReader):
    def __init__(self, images_folder = '', batch_size = 16, max_samples = 96, imgsz = 96, nc = 1):

        self.counter = 0
        self.end = max_samples
        self.images_folder = images_folder
        self.bs = batch_size
        self.imgsz = imgsz
        self.nc = nc
        
        self.loadfilesnames()


    def loadfilesnames(self, ) -> None:

        self.names = glob(f'{self.images_folder}/*.jpg') + glob(f'{self.images_folder}/*.png')
        shuffle(self.names)

        number_of_files = len(self.names)
        if number_of_files:
            self.end = min(self.end, number_of_files)
        print(f'Found {number_of_files} images, loaded {self.end}')

    def read_image(self, path, intype = 'float32'):
        return cv2.resize(cv2.imread(path), (self.imgsz, self.imgsz)).astype(intype)[:, :,::-1] / 255.
    
    def get_next(self) -> dict:
        
        if len(self.names) == 0:   
            #if no images are provided generate random data points (Debugging purposes)
            self.counter += self.bs
            return {"images": torch.randn((1, self.nc, self.imgsz, self.imgsz), ).numpy()} if self.counter < self.end else None 
        
        if self.counter > self.end - self.bs:
            return None

        

        
        
        im = np.stack( [self.read_image(self.names[i + self.counter]) for i in range(self.bs)], axis = 0)
        im = np.transpose(im, (0, 3, 1, 2))
        self.counter += self.bs
        return {'images' : im}





if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    model_fp32 = args.input
    model_int8 = args.output
    image_folder = args.images
    nb_samples = args.nb_samples
    imgsz = args.image_size
    batchsize = args.batch_size

    if "onnx" not in model_int8:
        model_int8+=".onnx"

    ds = DataReader(images_folder=image_folder ,batch_size=batchsize, max_samples = nb_samples, imgsz=640, nc= 3, )

    quantize_static(model_fp32, model_int8, calibration_data_reader=ds, op_types_to_quantize=['Conv'], quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    print('Finsihed quantization process, file saved ', model_int8)
