import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import argparse
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# For SORT tracking
import skimage
from sort import *

# ............................... Tracker Functions ............................
""" Random created palette"""
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

counting_line_point1 = (200, 400)  # Starting point of the line
counting_line_point2 = (150, 400)  # Ending point of the line

# vehicles total counting variables
array_ids = []
counting = 0
modulo_counting = 0
counted_objects = []


def is_crossing_line(x, y, line_point1, line_point2):
    return y == line_point1[1]


def count_object():
    global counting
    counting += 1
    print("Object crossed the counting line! Count:", counting)


def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0), counting_line_point1=None,
               counting_line_point2=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        data = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        label = str(id) + ":" + names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 144, 30), 1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

        midpoint_x = x1 + ((x2 - x1) / 2)
        midpoint_y = y1 + ((y2 - y1) / 2)
        center_point = (int(midpoint_x), int(midpoint_y))
        midpoint_color = (0, 255, 0)

        if counting_line_point1 is not None and counting_line_point2 is not None:
            if (midpoint_x > counting_line_point1[0] and midpoint_x < counting_line_point2[0]) and (
                    midpoint_y > counting_line_point1[1] and midpoint_y < counting_line_point2[1]):
                midpoint_color = (0, 0, 255)
                print('Category: ' + str(cat))

                if cat == 1:
                    counted_objects.append((id, midpoint_x, midpoint_y))
                    if len(array_ids) > 0:
                        if id not in array_ids:
                            array_ids.append(id)
                    else:
                        array_ids.append(id)

        cv2.circle(img, center_point, radius=1, color=midpoint_color, thickness=2)

    return img


#..............................................................................


def detect_frame(im0, opt, save_img=False, area1_pointA=None, area1_pointB=None, area1_pointC=None, area1_pointD=None):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    (('rtsp://', 'rtmp://', 'http://', 'https://'))

    counting_line_point1 = (250, 400) # Starting point of the line
    counting_line_point2 = (700, 400) # Ending point of the line

    draw_boxes(im0, bbox_xyxy, identities, categories, names, counting_line_point1, counting_line_point2)


    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) 
    #......................... 
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    count_bar2 = 0  # Reset count for BAR2 class in each frame
    count_rack = 0  # Reset count for RACK class in each frame

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        count_bar2 = 0  # Reset count for BAR2 class in each frame
        count_rack = 0  # Reset count for RACK class in each frame

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            dets_to_sort = np.empty((0, 6))

            # Check if the object crosses the counting line
            for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                # Calculate the bounding box center coordinates
                bbox_center_x = int((x1 + x2) / 2)
                bbox_center_y = int((y1 + y2) / 2)

                if bbox_center_y > counting_line_point1[1] and bbox_center_y < counting_line_point2[1]:
                    if bbox_center_x > counting_line_point1[0] and bbox_center_x < counting_line_point2[0]:
                        if names[int(detclass)] == "BAR2":
                            count_bar2 += 1
                        elif names[int(detclass)] == "RACK":
                            count_rack += 1

            # Run SORT
            tracked_dets = sort_tracker.update(dets_to_sort)
            tracks = sort_tracker.getTrackers()

            print('Tracked Detections : '+str(len(tracked_dets)))

            # Draw boxes for visualization
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]

                # Draw boxes with area points as additional arguments

                # Count objects crossing the counting line
                for idx, box in enumerate(bbox_xyxy):
                    x1, y1, x2, y2 = [int(i) for i in box]
                    midpoint_x = int((x1 + x2) / 2)
                    midpoint_y = int((y1 + y2) / 2)

                    # Check if the midpoint of the object crosses the counting line
                    if is_crossing_line(midpoint_x, midpoint_y, counting_line_point1, counting_line_point2):
                        count_object()

            # Draw boxes for visualization
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]

                draw_boxes(im0, bbox_xyxy, identities, categories, names, area1_pointA=area1_pointA, area1_pointB=area1_pointB, area1_pointC=area1_pointC, area1_pointD=area1_pointD)




            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            cv2.line(im0, counting_line_point1, counting_line_point2, (0, 255, 0), 2)

            color = (0, 255, 0)
            thickness = 2
            fontScale = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (160, 570)

            # Draw counts for each class on the image
            cv2.putText(im0, f'BAR2 count: {count_bar2}', org, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im0, f'RACK count: {count_rack}', (org[0], org[1] + 30), font, fontScale, color, thickness, cv2.LINE_AA)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

def main():
    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.65, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                detect_frame(im0, opt)
                strip_optimizer(opt.weights)
        else:
            # Load video source
            vid_path = opt.source
            vid_cap = cv2.VideoCapture(vid_path)

            while vid_cap.isOpened():
                ret, frame = vid_cap.read()
                if not ret:
                    break

                # Process frame with YOLO and SORT
                im0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detect_frame(im0, opt)
                
                # Display frame with results
                if opt.view_img:
                    cv2.imshow('Object Detection', im0)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            vid_cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()