#Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
#for audio
import pyttsx3 

# INITIALIZING THE pyttsx3 SO THAT 
# This function is used to initialize the text to the audio conversion of modules and 
# libraries used inside for the alerting purpose in the below code.
# ALERT AUDIO MESSAGE CAN BE DELIVERED
engine=pyttsx3.init()

cap=cv2.VideoCapture(0)

face_detector=dlib.get_frontal_face_detector()
dlib_facelandmark=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

drowsy=0
active=0
status=""
color=(0,0,0)

def compute(ptA,ptB):
    dist=np.linalg.norm(ptA-ptB)
    return dist

def blinked(a,b,c,d,e,f):
    shortest1=compute(b,d)
    shortest2=compute(c,e)
    longest=compute(a,f)
    aspect_ratio_eye=(shortest1+shortest2)/(2.0*longest)

    if(aspect_ratio_eye>0.25):
        return 1
    else:
        return 0

while True:
    # ret or _ is a boolean indicating whether the frame was read successfully
    #cap.read() is a method from OpenCV (cv2) used to capture a frame from a video or webcam.

    ret,frame=cap.read()

    # tells OpenCV to convert from BGR color format (default in OpenCV) to grayscale.
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    face_frame=frame.copy()
    
    faces=face_detector(gray)
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()

        #face_frame=frame.copy() declare it before as If no face is detected, the loop skips face_frame = frame.copy(), but you're still calling: cv2.imshow("Result of detector", face_frame)
        cv2.rectangle(face_frame,(x1,y1),(x2,y2),(0,255,0),2)

        landmarks=dlib_facelandmark(gray,face)
        landmarks=face_utils.shape_to_np(landmarks)

        left_blink=blinked(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],
                           landmarks[39])
        
        right_blink=blinked(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],
                           landmarks[45])
        

        if(left_blink==0 or right_blink==0):
            drowsy+=1
            active=0
            if (drowsy>8):
                status="Drowsiness Detected!!"
                color=(0,0,255)
                cv2.putText(frame,status,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)
                cv2.putText(frame,"Alert! WAKE UP",(100,150),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)

                engine.say("Alert! WAKE UP")
                engine.runAndWait()

        
        else:
            drowsy=0
            active+=1
            if(active>8):
                status="Active ^_^"
                color=(0,255,0)
                cv2.putText(frame,status,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)

        
        for n in range(0,68):
            (x,y)=landmarks[n]
            cv2.circle(face_frame,(x,y),1,(255,255,255),-1)

        

    cv2.imshow("Frame",frame)#shows status
    cv2.imshow("Result of detector",face_frame)#shows landmarks  

    key=cv2.waitKey(1)
    if key==27 : #27 is the ASCII code for the ESC key.
        break 

cap.release()
cv2.destroyAllWindows()
