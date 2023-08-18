import json
from models.experimental import attempt_load
from utils.torch_utils import select_device, load_classifier
from utils.general import check_img_size, non_max_suppression, scale_coords
import torch
from utils.datasets import letterbox
from utils.plots import plot_one_box
import cv2
import easydict
import numpy as np
import datetime
import os

def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class detector:
    def __init__(self):
        self.opt = json.load(open('config.json'))
        self.device = select_device(self.opt['device'])
        self.model = attempt_load(self.opt['weights'], map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[0, 255, 0]]
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = self.opt['img_size']
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        self.half = False
        self.save_img = True
        self.opt = easydict.EasyDict(self.opt)

    def __call__(self, img0):
        img = img = cv2.resize(img0, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.opt.augment)[0]
        pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    if(self.names[int(c)] == 'cars'):
                        total_cars = n

                for *xyxy, conf, cls in reversed(det):
                    if self.save_img:  # Add bbox to image
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=self.colors[0], line_thickness=3)
        return total_cars, img0



if __name__ == "__main__":
    outDir = 'out'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    d = detector()
    test_img = cv2.imread('test/images/4 (13)_1649859983.jpg')
    total_cars, img = d(test_img.copy())
    print(f'Total cars: {total_cars}')
    cv2.imwrite(os.path.join(outDir, f'out {get_time()}.jpg'), img)