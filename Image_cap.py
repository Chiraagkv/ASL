import cv2
import numpy as np
import random as rd
import time
import pickle
import os
import sys


class Image_detector ():
    def __init__(self, show_boxes=True, thresh_rate=1, cascade_path):
        try:
            self.cascade_path=cascade_path
        except Exception as e:
            print ("Error aoocured while process")
            sys.exit()
        self.hand_cascade=cv2.CascadeClassifier(self.cascade_path)
        self.list=[]
        self.show_boxes=show_boxes
        self.thresh_rate=thresh_rate
        self.crop_list=[]
        
        os.removedirs('images_folder')
        os.remove('predictions.bat')

        try:
            os.makedirs('images_folder')
        except Exception as e:
            pass
        self.image_processor()
        self.retrieve_data()

    def svae_images (self, frame):
        r=rd.randint(1, 1000)
        cv2.imwrite("images_folder\\hand_image{}.jpeg".format(str(r)), frame)
    
    def detect (self, gray_image, orig_frame):
        hand=self.hand_cascade.detectMultiScale(gray_image,3, 2)
        for (x, y, w, h) in hand:
            if self.show_boxes==True:
                cv2.rectangle(orig_frame, (x,y), (x+w,y+h), (0,255,0), 2)
                self.list.append([x, y, w, h])
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

    def retrieve_data (self):
        self.file_handler()
        with open('predictions.bat', 'rb') as f:
                while True:
                    try:
                        ans=pickle.load(f)
                        self.crop_list.append(ans.split('-'))
                    except EOFError:
                        break
        self.crop_images()

    def crop_images(self):
        os.makedirs('data_full')
        for m in range (len(self.crop_list)):
            data=self.crop_list[m][0]

            new_image=cv2.imread(data)
            y=int(self.crop_list[m][2])
            x=int(self.crop_list[m][1])
            h=int(self.crop_list[m][4])
            w=int(self.crop_list[m][3])
            
            new_img=new_image[y:y+h, x:x+w]
            os.remove(data)
            cv2.imwrite(data, new_img)


    def file_handler (self):
        with open('predictions.bat', 'wb') as f:
            for (roots, dirs, files) in os.walk('images_folder'):
                for i in range (len(files)):
                    images='images_folder'+'\\'+files[i]
                    print (self.list[i])
                    resluts=f"{images}-{self.list[i][0]}-{self.list[i][1]}-{self.list[i][2]}-{self.list[i][3]}"
                    pickle.dump(resluts, f)
        f.close()

a=Image_detector(thresh_rate=0.5)




        
