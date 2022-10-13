""" Libraries """
import os
import cv2
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from threading import Thread
from google_sheet_functions import append_values



""" Functions """
class FD_Ensemble(nn.ModuleList):
    def __init__(self):
        super(FD_Ensemble, self).__init__()
    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


def fd_attempt_load(weights, map_location=None):
    import sys
    sys.path.insert(0, "Footfall_Detection")
    model = FD_Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('FD_Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


def letterbox(img, new_shape, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 128), np.mod(dh, 128)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, fd_imgsz=96, vis_imgsz=(480, 640), sed_imgsz=96):
        self.mode = 'images'
        self.fd_imgsz = fd_imgsz
        # self.vis_imgsz = vis_imgsz
        self.sed_imgsz = sed_imgsz

        self.imgs = [None]
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), "Failed to open camera"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) % 100
        _, self.imgs[0] = cap.read()  # guarantee first frame
        thread = Thread(target=self.update, args=([0, cap]), daemon=True)
        print("Successfully open camera (%gx%g at %.2f FPS).\n" % (w, h, fps))
        thread.start()

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.fd_imgsz)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print("[WARNING] Different stream shapes detected. For optimal performance supply similarly-shaped streams.\n")

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        original_img = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        fd_img = [letterbox(x, new_shape=self.fd_imgsz, auto=self.rect)[0] for x in original_img]
        # vis_img = [letterbox(x, new_shape=self.vis_imgsz, auto=self.rect)[0] for x in original_img]
        sed_img = [letterbox(x, new_shape=self.sed_imgsz, auto=self.rect)[0] for x in original_img]

        # Stack
        fd_img = np.stack(fd_img, 0)
        # vis_img = np.stack(vis_img, 0)
        sed_img = np.stack(sed_img, 0)

        # # Convert
        fd_img = fd_img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        fd_img = np.ascontiguousarray(fd_img)
        # vis_img = vis_img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        # vis_img = np.ascontiguousarray(vis_img)
        sed_img = sed_img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        sed_img = np.ascontiguousarray(sed_img)

        # return fd_img, vis_img, original_img
        return fd_img, sed_img, original_img

    def __len__(self):
        return 0

class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, im0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))  # transforms
        else:
            im = letterbox(im0, self.img_size, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return im, im, im0
        return fd_img, sed_img, original_img

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


def hex2rgb(h):  # rgb order (PIL)
    return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def write_google_sheet(
    spreadsheet_id,
    range_name,
    time,
    current_people_amount,
    total_people_amount,
    alert_stock_amount,
    alert_stock_amount_smoothed
):
    cpa  = current_people_amount
    tpa  = total_people_amount
    asa  = alert_stock_amount
    asas = alert_stock_amount_smoothed
    return append_values(
        spreadsheet_id, range_name,
        [[ time, cpa, tpa, asa, asas ]]
    )