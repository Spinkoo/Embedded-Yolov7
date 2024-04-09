

import numpy as np
import torch
import torchvision

import numpy as np


def sigmoid(x):  
    return 1 / (1 + np.exp(-x))

def box_area(boxes):
    """
    Args:
        boxes(np.ndarray): [N, 4]
    return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1, box2):
    """
    Args:
        box1(np.ndarray): [N, 4]
        box2(np.ndarray): [M, 4]
    return: [N, M]
    """
    area1 = box_area(box1)
    area2 = box_area(box2)
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, np.newaxis] + area2 - inter)


def nms(boxes, scores, iou_threshold):
    """
    Non Max Suppression numpy implementation.
    args:
        boxes(np.ndarray): [N, 4]
        scores(np.ndarray): [N, 1]
        iou_threshold(float): Threshold of IoU.
    """
    idxs = scores.argsort()
    keep = []
    while idxs.size > 0:
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = box_iou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= iou_threshold]

    return np.array(keep)


class YOLOPostProcess(object):
    """
    Post process of YOLO-series network.
    args:
        score_threshold(float): Threshold to filter out bounding boxes with low 
                confidence score. If not provided, consider all boxes.
        nms_threshold(float): The threshold to be used in NMS.
        multi_label(bool): Whether keep multi label in boxes.
        keep_top_k(int): Number of total bboxes to be kept per image after NMS
                step. -1 means keeping all bboxes after NMS step.
        nms_top_k(int): Maximum number of boxes put into nums.
    """

    def __init__(self,
                 score_threshold=0.35,
                 nms_threshold=0.65,
                 multi_label=False,
                 keep_top_k=100,
                 nms_top_k=30000,
                 bs = 1,
                 nc = 2,
                 max_wh = 96):
        
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.multi_label = multi_label
        self.keep_top_k = keep_top_k
        self.nms_top_k = nms_top_k
        
        self.fuse_dim = 4
        self.max_wh = max_wh

        self.bs = bs
        self.nc = nc
        # x, y, w, h, obj_conf, {cls conf scores}
        self.bbox_size = 2 + 2 + 1 + nc 

        self.set_anchors()
        
    def set_anchors(self, anchors = [[10., 13.], [16., 30.],[33., 23.]]):
        self.anchor_grid  = np.array(anchors, dtype = np.float32)

    def count_anchors(self, output):
        return output.shape[1] // self.bbox_size
    

    
    def set_grid(self, feature_mapsz):
        self.grid = np.mgrid[: feature_mapsz, : feature_mapsz].transpose(2, 1, 0)[None].astype('float')

    def set_stride(self, feature_mapsz):
        self.stride = self.max_wh / feature_mapsz

    def reshape_flattened_output(self, output, feature_mapsz):

        """ 
            Transofrms the flattened output : feature map of size batch_size, -1, feature_map_width, feature_map_height => batch_size, N_total_predictions, -1 (batch_size, N_total_predictions, (x, y, w, h, obj_conf, cls1_conf, cls2_conf, ..., clsn_conf)
        """
        return output.reshape((self.bs, self.num_anchors, self.bbox_size, feature_mapsz, feature_mapsz), )
    



    def grid_transformation(self, output, feature_mapsz = 12):
        """ 
            Transofrms the output : feature map of size batch_size, -1, feature_map_width, feature_map_height => batch_size, N_total_predictions, -1 (batch_size, N_total_predictions, (x, y, w, h, obj_conf, cls1_conf, cls2_conf, ..., clsn_conf)
        """
        self.num_anchors = self.count_anchors(output)

        self.set_grid(feature_mapsz)
        self.set_stride(feature_mapsz)
        
        output =  self.reshape_flattened_output(output, feature_mapsz).transpose(0, 1, 3, 4, 2)
        output = sigmoid(output)
        
        output[..., 0:2] = (output[..., 0:2] * 2. - 0.5 + self.grid )* self.stride  # xy
        output[..., 2:4] = (output[..., 2:4] * 2) ** 2 * self.anchor_grid
        
        return output.reshape((1, -1, self.bbox_size))

    def _xywh2xyxy(self, x):
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def non_max_suppression(self, prediction, classes=None, agnostic=False, multi_label=False,
                            labels=()):
        """Runs Non-Maximum Suppression (NMS) on inference results

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > self.score_threshold  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        output = [torch.zeros((0, 6))] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            if nc == 1:
                x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                    # so there is no need to multiplicate.
            else:
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self._xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > self.score_threshold).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                j = np.argmax(x[:, 5:], axis = 1, keepdims=True)
                conf = x[np.arange(x.shape[0])[:, None],  j + 5]
                conf_ind = conf > self.score_threshold
                x = np.concatenate((box, conf,j + 1), 1)[conf_ind[:, 0]]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            
            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = nms(boxes, scores, self.nms_threshold)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > self.nms_threshold  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]


        return output

    def __call__(self, outs, scale_factor):
        return self.non_max_suppression(outs)
        