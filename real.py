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
area1_pointC = (670,430)
area1_pointD = (1200,430)

#vehicles total counting variables
array_ids = []
counting = 0
modulo_counting = 0
counted_objects = []


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
def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    rack_counting = 0
    bar2_counting = 0
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0 #객체의 종류
        id = int(identities[i]) if identities is not None else 0 #추적된 순서
        color = compute_color_for_labels(id)
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,144,30), 1)
        #cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)
        
        #c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        midpoint_x = x1+((x2-x1)/2)
        midpoint_y = y1+((y2-y1)/2)
        center_point = (int(midpoint_x),int(midpoint_y))
        midpoint_color = (0,255,0)
        
        if (midpoint_x > area1_pointA[0] and midpoint_x < area1_pointD[0]) and (midpoint_y > area1_pointA[1] and midpoint_y < area1_pointD[1]):
            
            midpoint_color = (0,0,255)
            print('Kategori : '+str(cat))
            
        # 감지된 객체의 카테고리 (클래스) 가져오기
        cat = int(categories[i]) if categories is not None else 0

        # 감지된 객체가 'RACK' (클래스 0) 또는 'BAR2' (클래스 1)인지 확인
        if cat == 0 or cat == 1:
            # 감지된 객체의 식별자 (ID) 가져오기
            id = int(identities[i]) if identities is not None else 0

            # 카운팅을 위한 색상 설정
            midpoint_color = (0, 255, 0)  # 카운팅되지 않은 객체에 대해 초록색

            # 객체가 카운팅 영역 내에 있는지 확인
            if (midpoint_x > area1_pointA[0] and midpoint_x < area1_pointD[0]) and (midpoint_y > area1_pointA[1] and midpoint_y < area1_pointD[1]):
                midpoint_color = (0, 0, 255)  # 카운팅된 객체에 대해 빨간색

            if (midpoint_x > area1_pointA[0] and midpoint_x < area1_pointD[0]) and (midpoint_y > area1_pointA[1] and midpoint_y < area1_pointD[1]):
                # ... (existing code)

                # Add objects to counting based on their classes
                if cat == 0:  # 'RACK' class (class 0) for counting
                    rack_counting += 1
                elif cat == 1:  # 'BAR2' class (class 1) for counting
                    bar2_counting += 1

            # 중심점 그리기 및 카운트 텍스트 표시
            cv2.circle(img, center_point, radius=1, color=midpoint_color, thickness=2)

    return img
#..............................................................................



def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Initialize counting variables
    count_bar = 0
    bar2_counting = 0
    rack_counting = 0

    # ....Initialize SORT ....
    # ...........................
    sort_max_age = 5  # Maximum number of frames maintained as tracking target
    sort_min_hits = 2  # minimum number of frames needed to keep track
    sort_iou_thresh = 0.2  # IoU threshold between detecting duplicate objects, if above that value, consider identical objects
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh)
    # ...........................
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

    # Define org_rack, org_bar2, font, font_scale, and thickness outside the loop
    org_rack = (30, 60)  # Position for displaying RACK count
    org_bar2 = (30, 90)  # Position for displaying BAR2 count
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # Define color here
    thickness = 1  # Define the thickness for drawing the counts

    count_bar = 0

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        # warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        rack_counting = 0
        bar2_counting = 0

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string




                # .................................USE TRACK FUNCTION....................
                # pass an empty array to sort
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([x1, y1, x2, y2, conf, detclass])))  # Extract x1, y1, x2, y2 coordinates, confidence, and detected class information from det, which is the object detection result.

                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)  # Input the generated array to the SORT algorithm's update method to perform object tracking. This returns information about objects tracked in the current frame.
                tracks = sort_tracker.getTrackers()  # Get tracker information for all objects currently being tracked.

                print('Tracked Detections : ' + str(len(tracked_dets)))  # Print the number of tracked objects in the current frame.
                print(tracks)
                # loop over tracks
                '''
                for track in tracks:
                    # color = compute_color_for_labels(id)
                    # draw tracks

                    [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track. centroidarr[i][1])),
                                    (int(track. centroidarr[i+1][0]),
                                    int(track. centroidarr[i+1][1])),
                                    (0,255,0), thickness=1)
                                    for i,_ in enumerate(track.centroidarr)
                                        if i < len(track.centroidarr)-1 ]
                '''

            # Draw boxes for visualization and Counting 'RACK' and 'BAR2' objects
            if len(tracked_dets) > 0:
                bbox_xyxy = tracked_dets[:, :4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]
                draw_boxes(im0, bbox_xyxy, identities, categories, names)

                rack_counting = sum(identities == 0)  # Count 'RACK' objects (class 0)
                bar2_counting = sum(identities == 1)  # Count 'BAR2' objects (class 1)

                # Display counts on the screen
                cv2.putText(im0, f'RACK Counting: {rack_counting}', org_rack, font, font_scale, color,
                            thickness, cv2.LINE_AA)
                cv2.putText(im0, f'BAR2 Counting: {bar2_counting}', org_bar2, font, font_scale, color,
                            thickness, cv2.LINE_AA)
                # .................................................

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            cv2.line(im0, area1_pointA, area1_pointB, (0, 255, 0), 2)
            cv2.line(im0, area1_pointC, area1_pointD, (0, 255, 0), 2)

            # Define the position for displaying RACK and BAR2 counts
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)
            thickness = 2
            org_rack = (30, 60)  # Position for displaying RACK count
            org_bar2 = (30, 90)  # Position for displaying BAR2 count
            print(array_ids)
            # Bar Counting
            # Bar counting
            if (count_bar == 0):
                counting = len(array_ids)  # set the initial counting value to the number of tracked object identifiers.
            else:
                if (counting < 100):
                    counting = len(array_ids)  # Update the counting value with the number of tracked object identifiers.
                else:
                    counting = modulo_counting + len(
                        array_ids)  # Update the counting value to be the sum of the previous counting value plus the number of tracked object identifiers.
                    if (len(array_ids) % 100 == 0):
                        modulo_counting = modulo_counting + 100  # Add to counting value in increments of 100.
                        array_ids.clear()  # Clear the tracked object identifier array.

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
        # print(f"Results saved to {save_dir}{s}")

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
        if opt.no_trace:
            detect()
        else:
            import traceback

            try:
                detect()
            except Exception as e:
                print(traceback.format_exc())
            
# python 22.py --weights best.pt --no-trace --view-img --nosave --source test1.avi --device 0