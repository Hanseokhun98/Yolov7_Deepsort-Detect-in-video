from glob import glob
img_list = glob('C:/Users/202207/Desktop/object detection model/object detection model/yolov7/dataset_test/images/*.jpg')
print(len(img_list))

from sklearn.model_selection import train_test_split
train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state = 2000)
print(len(train_img_list), len(val_img_list))

with open('C:/Users/202207/Desktop/object detection model/object detection model/yolov7/dataset_test/train.txt','w') as f:
    f.write('\n'.join(train_img_list)+'\n')
    
with open('C:/Users/202207/Desktop/object detection model/object detection model/yolov7/dataset_test/val.txt','w') as f:
    f.write('\n'.join(val_img_list)+'\n')
    
    

    


    




    




