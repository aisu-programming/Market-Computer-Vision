""" Libraries """
# Common libraries
import os
import cv2
import time
import glob
import argparse
import numpy as np
from pathlib import Path
from threading import Thread

# Shelf-Empty-Detection libraries
from Shelf_Empty_Detection.models.common import DetectMultiBackend
from Shelf_Empty_Detection.utils.general import non_max_suppression as sed_non_max_suppression

# Footfall-Detection libraries
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy import random
from Footfall_Detection.utils.general import check_img_size, scale_coords, xyxy2xywh, plot_one_box
from Footfall_Detection.utils.general import non_max_suppression as fd_non_max_suppression
from Footfall_Detection.utils.torch_utils import select_device

# # Vegetable-Instance-Segmentation libraries
# from Vegetable_Instance_Segmentation.model.simple_CNN import SimpleCNN



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


def detect(opt):

    fd_imgsz, sed_imgsz = opt.fd_img_size, opt.sed_img_size
    view_img, save_img, save_csv = opt.view_img, opt.save_img, opt.save_csv
    save_img_interval = opt.save_img_interval
    save_csv_interval = opt.save_csv_interval
    
    total_customer_amount = 0
    last_customer_amount = 0
    device = select_device("cpu")


    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    # dataset = LoadStreams(fd_imgsz=fd_imgsz, sed_imgsz=sed_imgsz)
    dataset = LoadImages(".", img_size=fd_imgsz)


    """ Shelf-Empty-Detection model """
    # Load model
    sed_model = DetectMultiBackend(opt.sed_weights, device=device)
    sed_model.warmup((1, 3, sed_imgsz, sed_imgsz))  # warmup
    sed_imgsz = check_img_size(sed_imgsz, s=sed_model.stride)  # check img_size

    # # Get names and colors
    # names = sed_model.module.names if hasattr(sed_model, "module") else sed_model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    """ Footfall-Detection model """
    # Load model
    fd_model = fd_attempt_load(opt.fd_weights, map_location=device)  # load FP32 fd_model
    # fd_imgsz = check_img_size(fd_imgsz, s=fd_model.stride.max())  # check img_size

    # # Get names and colors
    # names = fd_model.module.names if hasattr(fd_model, "module") else fd_model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    # img = torch.zeros((1, 3, fd_imgsz, fd_imgsz), device=device)  # init img
    # _ = fd_model(img)


    # """ Vegetable-Instance-Segmentation model """
    # vis_model = SimpleCNN(dropout=0)
    # vis_model.build(input_shape=(None, 640, 480, 3))
    # vis_model.load_weights(opt.vis_weights)


    """ Inference """
    date = ''
    # vis_smoothing = []
    # for fd_img, vis_img, original_img in dataset:
    for fd_img, sed_img, original_img in dataset:

        # Update output directory if it is a new day
        if date != time.strftime("%D", time.localtime()):
            date = time.strftime("%D", time.localtime())
            out = f"inference/{time.strftime('%m.%d-%H.%M', time.localtime())}"
            os.makedirs(out, exist_ok=True)
            csv_path = str(Path(out) / "records.csv")
            if save_csv:
                with open(csv_path, "w") as f:
                    f.write("time,current customer amount, customer total amount,vegetable amount, vegetable status,vegetable status (smoothed)\n")
                    # f.write("time,customer amount,confindence,x,y,w,h,\n")
                
        start_time = time.time()
        oi = original_img.copy()

        """ Footfall-Detection inference """
        fd_start_time = time.time()
        fd_img = torch.from_numpy(fd_img).to(device)
        fd_img = fd_img.float()  # uint8 to fp16/32
        fd_img /= 255.0
        if fd_img.ndimension() == 3: fd_img = fd_img.unsqueeze(0)
        s = "Footfall-Detection:\n\tinput size: %gx%g, " % fd_img.shape[2:]

        fd_inf_start_time = time.time()
        fd_pred = fd_model(fd_img, augment=False)[0]
        fd_inf_end_time = time.time()
        
        now_customer_amount = 0
        fd_pred = fd_non_max_suppression(fd_pred, opt.fd_conf_thres, opt.fd_iou_thres, classes=[0])[0]  # Apply NMS
        if fd_pred is not None and len(fd_pred):
            # Rescale boxes from img_size to oi size
            fd_pred[:, :4] = scale_coords(fd_img.shape[2:], fd_pred[:, :4], oi.shape).round()

            # Get customer amount
            for c in fd_pred[:, -1].unique():
                now_customer_amount = (fd_pred[:, -1] == c).sum()

            # Add bbox to image
            for i, (*xyxy, conf, _) in enumerate(fd_pred):
                if save_img or view_img:
                    label = f"Person no.{i+1}: conf {int(conf*100)}%"
                    plot_one_box(xyxy, oi, label=label, color=hex2rgb("#DD0000"), line_thickness=5)

        s += f"detected {now_customer_amount} people"
        if now_customer_amount != last_customer_amount:
            if now_customer_amount > last_customer_amount:
                s += f"... {(now_customer_amount-last_customer_amount)} new customer(s) in!\n\t"
                total_customer_amount += (now_customer_amount-last_customer_amount)
            else:
                s += f"... {(last_customer_amount-now_customer_amount)} customer(s) left!\n\t"
            last_customer_amount = now_customer_amount
        else:
            s += "\n\t"
        s += f"total customer amount pass by this camera: {total_customer_amount}\n\t"

        fd_end_time = time.time()
        s += f"inference time: {fd_inf_end_time-fd_inf_start_time:.3f}s, "
        s += f"total time: {fd_end_time-fd_start_time:.3f}s\n"

        # """ Vegetable-Instance-Segmentation inference """
        # vis_start_time = time.time()
        # s  += "Vegetable-Instance-Segmentation:\n\tinput size: %gx%g, " % vis_img.shape[1:3]
        # vis_img = np.array(vis_img, dtype=np.float32)
        # vis_img /= 255.0
        #
        # vis_pred_start_time = time.time()
        # vis_pred_amount = int(vis_model(vis_img)[0])
        # vis_pred_end_time = time.time()
        #
        # # Classfy the prediction amount to status
        # if   vis_pred_amount >= opt.vis_full_thres: vis_pred_status = "Full"
        # elif vis_pred_amount >= opt.vis_less_thres: vis_pred_status = "Less"
        # else                                      : vis_pred_status = "Empty"
        #
        # # Smooth the prediction
        # vis_smoothing.append({
        #     "Empty": 0,
        #     "Less" : 1,
        #     "Full" : 2,
        # }[vis_pred_status])
        # if len(vis_smoothing) > opt.vis_smoothing_len: vis_smoothing.remove(vis_smoothing[0])
        # vis_pred_status_smoothed = ["Empty", "Less", "Full"][round(np.average(vis_smoothing))]
        # s += f"prediction amount: {vis_pred_amount}% --> status prediction: {vis_pred_status} (smoothed: {vis_pred_status_smoothed})\n\t"
        #
        # vis_end_time = time.time()
        # s += f"inference time: {vis_pred_end_time-vis_pred_start_time:.3f}s, "
        # s += f"total time: {vis_end_time-vis_start_time:.3f}s\n"

        """ Shelf-Empty-Detection inference """
        sed_start_time = time.time()
        sed_img = torch.from_numpy(sed_img).to(device)
        sed_img = sed_img.float()  # uint8 to fp16/32
        sed_img /= 255.0
        if sed_img.ndimension() == 3: sed_img = sed_img.unsqueeze(0)
        s += "Shelf-Empty-Detection:\n\tinput size: %gx%g, " % sed_img.shape[2:]

        sed_inf_start_time = time.time()
        sed_pred = sed_model(sed_img)[0]
        sed_inf_end_time = time.time()
        
        sed_pred = sed_non_max_suppression(sed_pred, opt.sed_conf_thres, opt.sed_iou_thres, max_det=30, nm=32, single_special_cls=True)
        alert_stock_number = len(sed_pred)
        for i, det in enumerate(sed_pred):  # per image
            # Rescale boxes from img_size to oi size
            det[:, :4] = scale_coords(sed_img.shape[2:], det[:, :4], oi.shape).round()

            # Process predictions
            for *xyxy, conf, cls, amount in reversed(det[:, :7]):
                if save_img or view_img:
                    label = f"{amount.numpy():.1f} (conf: {int(conf*100)}%)"
                    plot_one_box(xyxy, oi, label=label, color=hex2rgb("#0000DD"), line_thickness=5)

        s += f"detected {alert_stock_number} stock\n\t"
        sed_end_time = time.time()
        s += f"inference time: {sed_inf_end_time-sed_inf_start_time:.3f}s, "
        s += f"total time: {sed_end_time-sed_start_time:.3f}s\n"

        # Stream results
        if view_img:
            cv2.imshow("Camera", oi)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save image with detections
        if save_img and save_img_interval <= 0:
            save_path = str(Path(out) / Path(time.strftime("%m.%d-%H.%M.%S", time.localtime())))
            cv2.imwrite(f"{save_path}.png", oi)
            save_img_interval = opt.save_img_interval

        # Save results to csv file
        if save_csv and save_csv_interval <= 0:
            with open(csv_path, "a") as f:
                now_time = time.strftime('%m.%d-%H.%M.%S', time.localtime())
                # f.write(f"{now_time},{now_customer_amount},{total_customer_amount},{vis_pred_amount},{vis_pred_status},{vis_pred_status_smoothed}\n")
                f.write(f"{now_time},{now_customer_amount},{total_customer_amount}\n")
            save_csv_interval = opt.save_csv_interval

        # Print time
        print(f"{s}Finished in {time.time()-start_time:.3f}s, start to sleep {opt.sleep}s\n")
        time.sleep(opt.sleep)
        save_csv_interval -= (time.time()-start_time)
        save_img_interval -= (time.time()-start_time)

    return



""" Execution """
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """ Footfall-Detection """
    parser.add_argument("--fd-weights", nargs="+", type=str,
        default="Footfall_Detection/yolov4-p5.pt", help="model.pt path for 'Footfall-Detection'")
    parser.add_argument("--fd-img-size", type=int, default=640,
        help="imput size(pixels) for 'Footfall-Detection', must be multiple of 32")
    parser.add_argument("--fd-conf-thres", type=float, default=0.3, help="object confidence threshold")
    parser.add_argument("--fd-iou-thres", type=float, default=0.5, help="IOU threshold for NMS")

    # """ Vegetable-Instance-Segmentation """
    # parser.add_argument("--vis-weights", nargs="+", type=str,
    #     default="Vegetable_Instance_Segmentation/weights.h5", help="model.h5 path for 'Vegetable-Instance-Segmentation'")
    # parser.add_argument("--vis-img-size", type=int, default=(480, 640),
    #     help="imput size(pixels) for 'Vegetable-Instance-Segmentation', not recommend to edit")
    # parser.add_argument("--vis-full-thres", type=int, default=70,
    #     help="threshold to classify to 'Full' than which predicted amount of vegetable is higher")
    # parser.add_argument("--vis-less-thres", type=int, default=30,
    #     help="threshold to classify to 'Less' than which predicted amount of vegetable is higher")
    # parser.add_argument("--vis-smoothing-len", type=int, default=10,
    #     help="the length of the smoothing array which to prevent unstable predictions")

    """ Shelf-Empty-Detection """
    parser.add_argument("--sed-weights", nargs="+", type=str,
        default="Shelf_Empty_Detection/best.pt", help="model.pt path for 'Shelf-Empty-Detection'")
    parser.add_argument("--sed-img-size", type=int, default=640,
        help="imput size(pixels) for 'Shelf-Empty-Detection', must be multiple of 32")
    parser.add_argument("--sed-conf-thres", type=float, default=0.3, help="object confidence threshold")
    parser.add_argument("--sed-iou-thres", type=float, default=0.5, help="IOU threshold for NMS")

    """ Common """
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--sleep", type=int, default=1, help="sleep time between each inference")
    # parser.add_argument("--output", type=str, help="output folder",
    #     default=f"inference/{time.strftime('%m.%d-%H.%M', time.localtime())}")  # output folder
    parser.add_argument("--save-img", action="store_true", help="save images")
    parser.add_argument("--save-img-interval", type=int, default=0, help="interval time(seconds) between every images saving")
    parser.add_argument("--save-csv", action="store_true", help="save results to csv")
    parser.add_argument("--save-csv-interval", type=int, default=5, help="interval time(seconds) between every record lines in csv")

    opt = parser.parse_args()
    print(opt)
    with torch.no_grad():
        detect(opt)