import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#For SORT tracking
import skimage
from sort import *

#............................... Tracker Functions ............................
""" Random created palette"""
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

area1_pointA = (670,230)
area1_pointB = (1200,230)
area1_pointC = (670,700)
area1_pointD = (1200,700)

#vehicles total counting variables
array_ids = []
count_bar2 = 0
count_rack = 0
counting = 0
modulo_counting = 0
counted_objects = []


def xyxy2xywh(x):
    # 바운딩 박스 형식을 (xmin, ymin, xmax, ymax)에서 (xcenter, ycenter, width, height)로 변환합니다.
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[0] = (x[0] + x[2]) / 2  # xcenter
    y[1] = (x[1] + x[3]) / 2  # ycenter
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y


"""" Calculates the relative bounding box from absolute pixel values. """
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

"""Simple function that adds fixed color depending on the class"""
def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

"""Function to Draw Bounding boxes"""
# Function to draw bounding boxes
def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0  # Object's category
        id = int(identities[i]) if identities is not None else 0  # Tracking order
        color = compute_color_for_labels(id)
        data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        label = str(id) + ":" + names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 144, 30), 1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

        # Draw midpoint of the bounding box
        midpoint_x = x1 + ((x2 - x1) / 2)
        midpoint_y = y1 + ((y2 - y1) / 2)
        center_point = (int(midpoint_x), int(midpoint_y))
        midpoint_color = (0, 255, 0)

        # Check if the midpoint is inside area1
        if (midpoint_x > area1_pointA[0] and midpoint_x < area1_pointD[0]) and (midpoint_y > area1_pointA[1] and midpoint_y < area1_pointD[1]):
            midpoint_color = (0, 0, 255)

            # Add vehicles counting for 'BAR2' class (class 1)
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


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))


    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 #추적대상으로 유지되는 최대 프레임 수
    sort_min_hits = 2 #추적을 유지하기 위해 필요한 최소 프레임 수
    sort_iou_thresh = 0.2 #중복된 객체 탐지 사이의 IoU 임계값, 해당 값 이상인 경우 동일한 객체로 간주
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
    name_to_id = {name: idx for idx, name in enumerate(names)}  # 카테고리 이름을 정수 레이블로 매핑
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    count_bar2 = 0
    count_rack = 0
    counting = 0  # Initialize 'counting' variable here
    counted_objects = []

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

            # Print results and count objects
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Check the class and count objects accordingly
                if int(c) == 1:  # Class 1 is 'BAR2'
                    count_bar2 += n
                elif int(c) == 0:  # Class 0 is 'RACK'
                    count_rack += n

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))#객체 감지 결과인 det에서 x1, y1, x2, y2 좌표, 신뢰도(confidence), 그리고 객체 클래스(detected class) 정보를 추출합니다.
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)#생성된 배열을 SORT 알고리즘의 update 메서드에 입력하여 객체 추적을 수행합니다. 이를 통해 현재 프레임에서 추적된 객체들의 정보가 반환됩니다.
                tracks =sort_tracker.getTrackers()# 현재 추적 중인 모든 객체의 트래커 정보를 가져옵니다.
                
                print('Tracked Detections : '+str(len(tracked_dets))) #현재 프레임에서 추적된 객체의 개수를 출력합니다.
                print(tracks)

                
            # Draw boxes for visualization
            if len(tracked_dets) > 0:  # If tracked objects exist, draw their boxes
                bbox_xyxy = tracked_dets[:, :4]  # Extract bounding box coordinates of tracked objects
                identities = tracked_dets[:, 5]  # Extract identities (class labels) of tracked objects
                categories = tracked_dets[:, 4]  # Extract category information of tracked objects

                # Map category values to integer labels
                categories = [name_to_id[names[int(cat)]] for cat in categories]

                draw_boxes(im0, bbox_xyxy, identities, categories, names)  # Draw boxes on the image
                # Count and display 'RACK' (class 0) and 'BAR2' (class 1) objects
                for cat, identity, bbox in zip(categories, identities, bbox_xyxy):
                    cat_name = names[int(cat)]
                    coord = bbox_rel(*xyxy2xywh(bbox))
                    print(f"{cat_name} (Class {int(cat)}) coordinates and identity: {coord}, {int(identity)}")
                    
                    if cat_name == 'RACK':
                        count_rack += 1
                    elif cat_name == 'BAR2':
                        count_bar2 += 1
 #                       counted_objects.append(coord)

            # Print the total counts for both classes
            print('Total RACK Count:', count_rack)
            print('Total BAR2 Count:', count_bar2)
            print('BAR2 Coordinate:', counted_objects)

                #........................................................
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            cv2.line(im0,area1_pointA,area1_pointB,(0,255,0),2)
            cv2.line(im0,area1_pointC,area1_pointD,(0,255,0),2)

            color = (0,255,0)
            thickness = 2
            fontScale = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (30,30)
            print(array_ids)
            # Bar Counting
            # Bar 카운팅
            if (count_bar2 == 0):
                counting = len(array_ids)# 최초 카운팅 값을 추적된 객체 식별자 수로 설정합니다.
            else:
                if (counting < 100):
                    counting = len(array_ids)# 카운팅 값을 추적된 객체 식별자 수로 업데이트합니다.
                else:
                    counting = modulo_counting + len(array_ids) # 카운팅 값을 이전 카운팅 값과 추적된 객체 식별자 수의 합으로 업데이트합니다.
                    if(len(array_ids)%100 == 0):
                        modulo_counting = modulo_counting + 100 # 100 단위로 카운팅 값에 추가합니다.
                        array_ids.clear() # 추적된 객체 식별자 배열을 비웁니다.
                
            cv2.putText(im0, 'Bar Counting = '+str(counting), org, font, fontScale, color, thickness, cv2.LINE_AA)# 이미지에 Bar 카운팅 정보를 텍스트로 표시합니다.    
            cv2.putText(im0, 'Bar Counting = ' + ', '.join(map(str, counted_objects)), org, font, fontScale, color, thickness, cv2.LINE_AA)
            rack_count_text = f'RACK Count: {count_rack}'
            cv2.putText(im0, rack_count_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

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

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=720, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['best.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
            
            
            
# python 22.py --weights best.pt --no-trace --view-img --nosave --source test1.avi --device 0