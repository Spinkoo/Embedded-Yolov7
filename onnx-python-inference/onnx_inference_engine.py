


import cv2
import onnxruntime as ort
import numpy as np
from utils.post_process_yolo import YOLOPostProcess
from utils.yaml_parse import fetch_params
from time import time



class onnx_model():
    def __init__(self, onnx_path = "", yaml_path = None, score_threshold=0.25, nms_threshold=0.65) -> None:
        self.model_session = self.init_model(onnx_path)

        if yaml_path:
            nc, self.anchors, self.feature_map_sizes = fetch_params(yaml_path)
            bs, (self.InH, self.InW, self.InC), self.actual_shape = self.infer_shape()
            max_wh = max(self.InH, self.InW)
        else:
            #default config of the tiny-yolov7.pt
            nc, self.anchors, self.feature_map_sizes,  = self.get_default_config()
            bs, (self.InH, self.InW, self.InC), self.actual_shape = self.infer_shape()
        
        max_wh = max(self.InH, self.InW)

        self.yolo_pp = YOLOPostProcess(score_threshold=score_threshold, nms_threshold=nms_threshold, nc=nc, bs=bs, max_wh=max_wh )


    def get_default_config(self):
        return 80, [[[[[[[10, 13]]], [[[16, 30]]], [[[33, 23]]]]]], [[[[[[30, 61]]], [[[62, 45]]], [[[59, 119]]]]]], [[[[[[116, 90]]], [[[156, 198]]], [[[373, 326]]]]]]], \
        [80, 40, 20]
    
    def infer_shape(self):
        shape = self.model_session.get_inputs()[0].shape
        bs, shape = shape[0], shape[1:]
        #for CHW
        if shape[0] <= 3:
            self.chanels_first = True
            return bs, (*shape[1:],shape[0]), shape
        
        self.chanels_first = False
        return bs, (*shape,), shape
    
    def init_model(self, onnx_path, cuda = True):
            providers = ['CPUExecutionProvider'] if cuda else ['CUDAExecutionProvider']
            return ort.InferenceSession(onnx_path, providers=providers)





    def letterbox(self, im,  color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints

        new_shape = (self.InH, self.InW)


        shape = im.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        if im.shape != [self.InH, self.InW, self.InC]:
            im = cv2.resize(im, (self.InW, self.InH), interpolation=cv2.INTER_LINEAR)
        return im, r, (dw, dh)

    def setup_output(self, img,):
        image, ratio, dwdh = self.letterbox(img.copy(), auto=False, )
        image = img[:, :, ::-1]

        if self.chanels_first:
           image = image.transpose(2, 0, 1)
        image = image[None]
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)

        im /= 255.
        
        outname = [i.name for i in self.model_session.get_outputs()]
        
        inname = [i.name for i in self.model_session.get_inputs()]


        inp = {inname[0]:im}
        op =self.model_session.run(outname, inp)[0]
        return op, ratio, dwdh, image

    def apply_grid_transformation(self, anchor, fmpsze, op):

        self.yolo_pp.set_anchors(anchor)

        return self.yolo_pp.grid_transformation(op, fmpsze)
        
    def predictions(self, inputs, nms = False, grid = False):

        """
        args :

        inputs  = input image

        nms  = False if the ONNX model already has the NMS post process included 

        grid = False if the ONNX model already has the Grid transformation post process included 



        """
        
        results = []
        for img in inputs:
            t1 = time()
            outputs, _, _, _ = self.setup_output(img)
            t2 = time()


            
            if grid:               
                inters = []
                offset = 0
                for _, fs in enumerate(self.feature_map_sizes):
                    inters.append((offset , offset +  fs * fs))
                    offset +=  fs * fs

                outputs = [self.apply_grid_transformation(self.anchors[i], fmpsze, outputs[..., inters[i][0] : inters[i][1]]) for i, fmpsze in enumerate(self.feature_map_sizes)]    
                outputs = np.concatenate((*outputs,), axis = 1)            
            t3 = time()
            if nms:
                #perform NMS
                outputs = self.yolo_pp(outputs, scale_factor=[[1, 1] * outputs.shape[0]])
                #output boxes are transformed from xyhw to x1y1x2y2
            t4 = time()

            print('Inference time', t2 - t1, 'Post process and coords mapping time', t3 - t2, 'Nms time', t4 - t3)
            results.append(outputs[0])
        
        return results

    










    
        


