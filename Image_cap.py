import cv2
import numpy as np
import random as rd
import time

class Image_detector ():
    def __init__(self, path, show_boxes=True, thresh_rate=1):
        self.path=path
        self.hand_cascade=cv2.CascadeClassifier(f'{self.path}')
        self.list=[]
        self.show_boxes=show_boxes
        self.thresh_rate=thresh_rate

    def svae_images (self, frame):
        r=rd.randint(1, 1000)
        cv2.imwrite("hand_image{}.jpeg".format(str(r)), frame)
    
    def detect (self, gray_image, orig_frame):
        hand=self.hand_cascade.detectMultiScale(gray_image,3, 2)
        for (x, y, w, h) in hand:
            if self.show_boxes==True:
                cv2.rectangle(orig_frame, (x,y), (x+w,y+h), (0,255,0), 2)
                self.list.extend([x, y, w, h])
                time.sleep(self.thresh_rate)
                self.svae_images(orig_frame)
            else:
                self.list.extend([x, y, w, h])
                time.sleep(self.thresh_rate)
                self.svae_images(orig_frame)

        return orig_frame
    def image_processor (self):
        vid=cv2.VideoCapture(0)

        while True:
            nothing, frame=vid.read()
            gray_scale=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            our_reslut=self.detect(gray_image=gray_scale, orig_frame=frame)
            cv2.imshow('vidheo', our_reslut)

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        vid.release()
        cv2.destroyAllWindows()
        return self.list


a=Image_detector()
my_boi=a.image_processor()
