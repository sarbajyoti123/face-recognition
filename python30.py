from PyQt5 import QtGui
from PyQt5.QtGui import QBrush,QPainter,QPen

from PyQt5.QtWidgets import QApplication, QMainWindow,QPushButton
from PyQt5.QtCore import Qt
import os
from PIL import Image
import numpy as np
import cv2
import pickle
import sys


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "PyQt5 Window"        
        self.top = 100        
        self.left = 100        
        self.width = 680        
        self.height = 500

        self.InitWindow()


    def InitWindow(self):
        self.button=QPushButton("here",self)
        self.button.move(100,100)
        self.button.clicked.connect(self.facial)
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()
    def facial(self):

        def faces():
            base_dir=os.path.dirname(os.path.abspath(__file__))
            image_dir=os.path.join(base_dir,"image")
            facedetect=cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml');
            recognizer=cv2.face.LBPHFaceRecognizer_create()
            current_id=0
            x_train=[]
            y_labels=[]
            label_ids={}

            for root,dirs,files in os.walk(image_dir):
                for file in files:
                    if file.endswith("png") or file.endswith("jpg"):
                        path=os.path.join(root,file)
                        label=os.path.basename(os.path.dirname(path))
                        # print(label,path)
                        if label in label_ids:
                            pass
                        else:
                            label_ids[label]=current_id
                            current_id=current_id+1
                            id_=label_ids[label]
                            # print(label_ids)        
                        pil_image=Image.open(path).convert("L")
                        image_array=np.array(pil_image,'uint8')
                        # print(image_array)
                        # faces=facedetect.detectMultiScale(image_array,1.3,5)
                        x_train.append(image_array)
                        y_labels.append(id_)
            # print(x_train)
            # print(y_labels)
            with open('labels.pickle','wb') as f:
                pickle.dump(label_ids,f)
            recognizer.train(x_train,np.array(y_labels))
            recognizer.save("trainner.yml")                
        faces()

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
