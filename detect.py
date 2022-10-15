""" Libraries """
import os
import cv2
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path
from utils import LoadImages, LoadStreams, fd_attempt_load, hex2rgb, write_google_sheet
from google_sheet_functions import create_sheet
from Footfall_Detection.utils.general import check_img_size, scale_coords, plot_one_box
from Footfall_Detection.utils.torch_utils import select_device



""" Functions """
def detect(opt):

    fd_imgsz  = False if not opt.fd  else opt.fd_img_size
    vis_imgsz = False if not opt.vis else opt.vis_img_size
    sed_imgsz = False if not opt.sed else opt.sed_img_size
    view_img, save_img, save_csv, save_google_sheet = opt.view_img, opt.save_img, opt.save_csv, opt.save_google_sheet
    save_img_interval          = 0
    save_csv_interval          = 0
    save_google_sheet_interval = 0
    google_sid = opt.google_sid

    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    if opt.test:
        dataset = LoadImages("test/", fd_imgsz, vis_imgsz, sed_imgsz)
    else:
        dataset = LoadStreams(fd_imgsz, vis_imgsz, sed_imgsz)

    device = select_device("cpu")

    """ Shelf-Empty-Detection model """
    if opt.sed:
        from Shelf_Empty_Detection.models.common import DetectMultiBackend
        from Shelf_Empty_Detection.utils.general import non_max_suppression as sed_non_max_suppression
        sed_model = DetectMultiBackend(opt.sed_weights, device=device)
        sed_model.warmup((1, 3, sed_imgsz, sed_imgsz))  # warmup
        sed_imgsz = check_img_size(sed_imgsz, s=sed_model.stride)  # check img_size
    else:
        sed_alert_stock_amount          = "N/A"
        sed_alert_stock_amount_smoothed = "N/A"

    # """ Vegetable-Instance-Segmentation model """
    # if opt.vis:
    #     from Vegetable_Instance_Segmentation.model.simple_CNN import SimpleCNN
    #     vis_model = SimpleCNN(dropout=0)
    #     vis_model.build(input_shape=(None, 640, 480, 3))
    #     vis_model.load_weights(opt.vis_weights)

    """ Footfall-Detection model """
    if opt.fd:
        last_customer_amount     = 0
        total_people_amount      = 0
        last_total_people_amount = 0
        from Footfall_Detection.utils.general import non_max_suppression as fd_non_max_suppression
        fd_model = fd_attempt_load(opt.fd_weights, map_location=device)  # load FP32 fd_model
        fd_imgsz = check_img_size(fd_imgsz, s=fd_model.stride.max())  # check img_size
        img = torch.zeros((1, 3, fd_imgsz, fd_imgsz), device=device)  # init img
        _ = fd_model(img)
    else:
        # now_people_amount       = "N/A"
        total_people_amount     = "N/A"
        people_amount_in_period = "N/A"

    """ Inference """
    date = ''
    vis_smoothing, sed_smoothing = [], []
    for fd_img, vis_img, sed_img, original_img in dataset:

        now_hour = int(time.strftime('%H', time.localtime()))
        while now_hour < 8 or now_hour >= 21:
            now_hour = int(time.strftime('%H', time.localtime()))
            time.sleep(60*60)  # 1 hour
        while now_hour < 9:
            now_hour = int(time.strftime('%H', time.localtime()))
            time.sleep(60)     # 1 minute

        # Update output directory if it is a new day
        if date != time.strftime("%Y/%m/%d", time.localtime()):
            date = time.strftime("%Y/%m/%d", time.localtime())
            out = f"inference/{time.strftime('%Y.%m.%d', time.localtime())}"
            csv_path = str(Path(out) / "records.csv")
            if save_csv or save_img:
                os.makedirs(out, exist_ok=True)
            if save_csv:
                if not os.path.exists(csv_path):
                    with open(csv_path, "w") as f:
                        f.write("time,people amount in period,total people amount,alert stock amount,alert stock amount (smoothed)\n")
            if save_google_sheet:
                try:
                    create_sheet(google_sid, date)
                    write_google_sheet(google_sid, date,
                                       "time", "people amount in period", "total people amount",
                                       "alert stock amount", "alert stock amount (smoothed)")
                except:
                    pass
                
        start_time = time.time()
        oi = original_img.copy()
        s = ''

        """ Footfall-Detection inference """
        if opt.fd:
            fd_start_time = time.time()
            fd_img = torch.from_numpy(fd_img).to(device)
            fd_img = fd_img.float()  # uint8 to fp16/32
            fd_img /= 255.0
            if fd_img.ndimension() == 3: fd_img = fd_img.unsqueeze(0)
            s += "Footfall-Detection:\n\tinput size: %gx%g, " % fd_img.shape[2:]

            fd_inf_start_time = time.time()
            fd_pred = fd_model(fd_img, augment=False)[0]
            fd_inf_end_time = time.time()
            
            now_people_amount = 0
            fd_pred = fd_non_max_suppression(fd_pred, opt.fd_conf_thres, opt.fd_iou_thres, classes=[0])[0]  # Apply NMS
            if fd_pred is not None and len(fd_pred):
                # Rescale boxes from img_size to oi size
                fd_pred[:, :4] = scale_coords(fd_img.shape[2:], fd_pred[:, :4], oi.shape).round()

                # Get customer amount
                for c in fd_pred[:, -1].unique():
                    now_people_amount = (fd_pred[:, -1] == c).sum().item()

                # Add bbox to image
                if save_img or view_img:
                    for i, (*xyxy, conf, _) in enumerate(fd_pred):
                        label = f"Person no.{i+1}: conf {int(conf*100)}%"
                        plot_one_box(xyxy, oi, label=label, color=hex2rgb("#DD0000"), line_thickness=5)

            s += f"detected {now_people_amount} people"
            if now_people_amount != last_customer_amount:
                if now_people_amount > last_customer_amount:
                    s += f"... {(now_people_amount-last_customer_amount)} new customer(s) in!\n\t"
                    total_people_amount += (now_people_amount-last_customer_amount)
                else:
                    s += f"... {(last_customer_amount-now_people_amount)} customer(s) left!\n\t"
                last_customer_amount = now_people_amount
            else:
                s += "\n\t"
            s += f"total customer amount pass by this camera: {total_people_amount}\n\t"

            fd_end_time = time.time()
            s += f"inference time: {fd_inf_end_time-fd_inf_start_time:.3f}s, "
            s += f"total time: {fd_end_time-fd_start_time:.3f}s\n"

        # """ Vegetable-Instance-Segmentation inference """
        # if opt.vis:
        #     vis_start_time = time.time()
        #     s  += "Vegetable-Instance-Segmentation:\n\tinput size: %gx%g, " % vis_img.shape[1:3]
        #     vis_img = np.array(vis_img, dtype=np.float32)
        #     vis_img /= 255.0
            
        #     vis_pred_start_time = time.time()
        #     vis_pred_amount = int(vis_model(vis_img)[0])
        #     vis_pred_end_time = time.time()
            
        #     # Classfy the prediction amount to status
        #     if   vis_pred_amount >= opt.vis_full_thres: vis_pred_status = "Full"
        #     elif vis_pred_amount >= opt.vis_less_thres: vis_pred_status = "Less"
        #     else                                      : vis_pred_status = "Empty"
            
        #     # Smooth the prediction
        #     vis_smoothing.append({
        #         "Empty": 0,
        #         "Less" : 1,
        #         "Full" : 2,
        #     }[vis_pred_status])
        #     if len(vis_smoothing) > opt.vis_smoothing_len: vis_smoothing.remove(vis_smoothing[0])
        #     vis_pred_status_smoothed = ["Empty", "Less", "Full"][round(np.average(vis_smoothing))]
        #     s += f"prediction amount: {vis_pred_amount}% --> status prediction: {vis_pred_status} (smoothed: {vis_pred_status_smoothed})\n\t"
            
        #     vis_end_time = time.time()
        #     s += f"inference time: {vis_pred_end_time-vis_pred_start_time:.3f}s, "
        #     s += f"total time: {vis_end_time-vis_start_time:.3f}s\n"

        """ Shelf-Empty-Detection inference """
        if opt.sed:
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
            sed_alert_stock_amount = 0
            for i, det in enumerate(sed_pred):  # per image
                # Rescale boxes from img_size to oi size
                det[:, :4] = scale_coords(sed_img.shape[2:], det[:, :4], oi.shape).round()

                # Process predictions
                for *xyxy, conf, cls, amount in reversed(det[:, :7]):
                    if amount <= opt.sed_alert_amount_thres: sed_alert_stock_amount += 1
                    if save_img or view_img:
                        label = f"{amount.numpy():.1f} (conf: {int(conf*100)}%)"
                        plot_one_box(xyxy, oi, label=label, color=hex2rgb("#0000DD"), line_thickness=5)

            sed_smoothing.append(sed_alert_stock_amount)
            if len(sed_smoothing) > opt.sed_smoothing_len: sed_smoothing.remove(sed_smoothing[0])
            sed_alert_stock_amount_smoothed = round(np.average(sed_smoothing))

            s += f"detected {sed_alert_stock_amount} stock\n\t"
            sed_end_time = time.time()
            s += f"inference time: {sed_inf_end_time-sed_inf_start_time:.3f}s, "
            s += f"total time: {sed_end_time-sed_start_time:.3f}s\n"

        # Stream results
        if not opt.test and view_img:
            cv2.imshow("Camera", oi)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                cv2.destroyAllWindows()
                raise StopIteration

        # Save image with detections
        if save_img and save_img_interval <= 0:
            filename = str(Path(out) / Path(time.strftime("%H.%M.%S.png", time.localtime())))
            cv2.imwrite(filename, oi)
            save_img_interval = opt.save_img_interval

        # Save results to csv file and Google sheet
        now_time = time.strftime("%H:%M:%S", time.localtime())
        if opt.fd:
            people_amount_in_period = total_people_amount - last_total_people_amount
            
        if save_csv and save_csv_interval <= 0:
            with open(csv_path, "a") as f:
                f.write(f"{now_time},{people_amount_in_period},{total_people_amount},{sed_alert_stock_amount},{sed_alert_stock_amount_smoothed}\n")
            save_csv_interval = opt.save_csv_interval
        if save_google_sheet and save_google_sheet_interval <= 0:
            write_google_sheet(google_sid, date,
                               now_time, people_amount_in_period, total_people_amount,
                               sed_alert_stock_amount, sed_alert_stock_amount_smoothed)
            save_google_sheet_interval = opt.save_google_sheet_interval
        last_total_people_amount = total_people_amount

        # Print time
        print(f"{s}Finished in {time.time()-start_time:.3f}s, start to sleep {opt.sleep}s\n")
        time.sleep(opt.sleep)
        save_img_interval          -= (time.time()-start_time)
        save_csv_interval          -= (time.time()-start_time)
        save_google_sheet_interval -= (time.time()-start_time)

    return



""" Execution """
if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv(".env")
    DEFAULT_SPREADSHEET_ID = os.environ.get("DEFAULT_SPREADSHEET_ID")

    parser = argparse.ArgumentParser()

    """ Footfall-Detection """
    parser.add_argument("--fd", action="store_true", help="turn on Footfall-Detection or not")
    parser.add_argument("--fd-weights", nargs="+", type=str,
        default="Footfall_Detection/footfall-detection.pt", help="model.pt path for 'Footfall-Detection'")
    parser.add_argument("--fd-img-size", type=int, default=64,
        help="imput size(pixels) for 'Footfall-Detection', must be multiple of 32")
    parser.add_argument("--fd-conf-thres", type=float, default=0.3, help="object confidence threshold")
    parser.add_argument("--fd-iou-thres", type=float, default=0.5, help="IOU threshold for NMS")

    """ Vegetable-Instance-Segmentation """
    parser.add_argument("--vis", action="store_true", help="turn on Vegetable-Instance-Segmentation or not")
    # parser.add_argument("--vis-weights", nargs="+", type=str,
    #     default="Vegetable_Instance_Segmentation/vegetable-instance-segmentation.h5", help="model.h5 path for 'Vegetable-Instance-Segmentation'")
    # parser.add_argument("--vis-img-size", type=int, default=256,
    #     help="imput size(pixels) for 'Vegetable-Instance-Segmentation', not recommend to edit")
    # parser.add_argument("--vis-full-thres", type=int, default=70,
    #     help="threshold to classify to 'Full' than which predicted amount of vegetable is higher")
    # parser.add_argument("--vis-less-thres", type=int, default=30,
    #     help="threshold to classify to 'Less' than which predicted amount of vegetable is higher")
    # parser.add_argument("--vis-smoothing-len", type=int, default=10,
    #     help="the length of the smoothing array which to prevent unstable predictions")

    """ Shelf-Empty-Detection """
    parser.add_argument("--sed", action="store_true", help="turn on Shelf-Empty-Detection or not")
    parser.add_argument("--sed-weights", nargs="+", type=str,
        default="Shelf_Empty_Detection/shelf-empty-detection-0.75.pt", help="model.pt path for 'Shelf-Empty-Detection'")
    parser.add_argument("--sed-img-size", type=int, default=256,
        help="imput size(pixels) for 'Shelf-Empty-Detection', must be multiple of 32")
    parser.add_argument("--sed-alert-amount-thres", type=float, default=0.5, help="threshold for stock amount to be alert")
    parser.add_argument("--sed-conf-thres", type=float, default=0.3, help="object confidence threshold")
    parser.add_argument("--sed-iou-thres", type=float, default=0.5, help="IOU threshold for NMS")
    parser.add_argument("--sed-smoothing-len", type=int, default=10,
        help="the length of the smoothing array which to prevent unstable predictions")

    """ Common """
    parser.add_argument("--test", action="store_true", help="test inference by using testing images")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--sleep", type=int, default=0, help="sleep time between each inference")
    # parser.add_argument("--output", type=str, help="output folder",
    #     default=f"inference/{time.strftime('%Y.%m.%d', time.localtime())}")  # output folder
    parser.add_argument("--save-img", action="store_true", help="save images")
    parser.add_argument("--save-img-interval", type=int, default=15*60, help="interval time(seconds) between every images saving")
    parser.add_argument("--save-csv", action="store_true", help="save results to csv")
    parser.add_argument("--save-csv-interval", type=int, default=15*60, help="interval time(seconds) between every record lines in csv")
    parser.add_argument("--save-google-sheet", action="store_true", help="save results to google sheet")
    parser.add_argument("--save-google-sheet-interval", type=int, default=15*60, help="interval time(seconds) between every record lines in google sheet")
    parser.add_argument("--google-sid", type=str, default=DEFAULT_SPREADSHEET_ID, help="Spreadsheet ID of the target Google Sheet")

    opt = parser.parse_args()
    print(opt)

    if opt.save_google_sheet:
        assert opt.google_sid != '', 'The argument "--google-sid" must not be empty string.'

    with torch.no_grad():
        detect(opt)