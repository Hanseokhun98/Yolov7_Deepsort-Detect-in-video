import cv2
import os

video_file = 'aa.mp4'  # 비디오 파일
output_dir = 'frames'  # 저장폴더

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_file)
count = 0
frame_interval = 10  # 프레임 인터벌

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if count % frame_interval == 0:
        filename = f'frame_{count}.jpg'
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        
    count += 1

cap.release()
