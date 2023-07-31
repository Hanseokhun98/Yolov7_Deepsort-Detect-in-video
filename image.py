#%%
import cv2
import os

video_file = 'aa.mp4'
output_dir = 'frames'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_file)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    filename = f'frame_{count}.jpg'
    cv2.imwrite(os.path.join(output_dir, filename), frame)
        
    count += 1

cap.release()

# %%
