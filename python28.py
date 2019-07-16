from PyQt5 import QtGui
from PyQt5.QtGui import QBrush,QPainter,QPen

from PyQt5.QtWidgets import QApplication, QMainWindow,QPushButton
from PyQt5.QtCore import Qt
import time
import cv2
import numpy as np
import pickle
import sys
import cv2


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
        self.button=QPushButton('click',self)
        self.button.move(100,400)
        self.button.clicked.connect(self.face)
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    def face(self):
        def cam():
            facedetect=cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml');
            recognizer=cv2.face.LBPHFaceRecognizer_create()
            recognizer.read("trainner.yml")
            labels={"name":1}
            with open('labels.pickle','rb') as f:
                s_labels=pickle.load(f)
                labels={v:k for k,v in s_labels.items()}
            image=cv2.VideoCapture(0)
            while(True):
                check,frame=image.read()
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces=facedetect.detectMultiScale(gray,1.3,5)
                for (x,y,w,h) in faces:
                    # print(x,y,w,h)
                    roi_gray=gray[y:y+h,x:x+w]
                    roi_color=frame[y:y+h,x:x+w]
                    id_,conf=recognizer.predict(roi_gray)
                    if conf>=80:
                        # print(id_)
                        print(labels[id_])
                    cv2.imwrite("frame.jpg",roi_gray)    
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)  
                    cv2.putText(frame,id_,(10,500),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,0),2)
                       
                cv2.imshow("frames.jpg",frame)  
                key=cv2.waitKey(1)
                if key==ord("q"):
                    break
            image.release()
            cv2.destroyAllWindows()
        cam()        

App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())

