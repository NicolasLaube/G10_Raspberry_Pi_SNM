#################################
# Nicolas LAUBE
# Sylvain MACE
# Melvin BICHO
# package utilitaire pour le projet
#####################################

import cv2
import matplotlib.pyplot as plt
import cv2 as cv

Filepath = "../Data/tetris_blocks.png"


def load_and_display_image(filepath):
    """fonction permettant de charger et d'afficher une image (filename) """

    img = cv2.imread(filepath, 1)
    if img is not None:
        plot = plt.imshow(img)
        plt.show()
    else:
        raise ValueError("L'image " + filepath + " n'existe pas")


#load_and_display_image(Filepath)


def process_image(filepath):
    img = cv2.imread(filepath,0)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()

#process_image(Filepath)


def process_image_2(filepath):

    img = cv2.imread(filepath,1)
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


#process_image_2(Filepath)

face_cascade = cv2.CascadeClassifier('C:/Users/melvi/Coding Weeks/G10_Raspberry_Pi_SNM/facerecognition/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/melvi/Coding Weeks/G10_Raspberry_Pi_SNM/facerecognition/data/haarcascade_eye.xml')
img = cv2.imread('C:/Users/melvi/Coding Weeks/G10_Raspberry_Pi_SNM/facerecognition/data/melvin.bicho5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
plt.imshow(img)
plt.show()


def reco_webcam():

   face_cascade = cv.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
   eye_cascade = cv.CascadeClassifier('Data/haarcascade_eye.xml')

#Webcam capture
   video_capture = cv.VideoCapture(0)
   while True :
       ret, frame = video_capture.read()
       frame = cv.flip(frame, 1)

       gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
       faces = face_cascade.detectMultiScale(gray, 1.3, 5)


       for (x,y,w,h) in faces:
           cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
           roi_gray = gray[y:y+h, x:x+w]
           roi_color = frame[y:y+h, x:x+w]
           eyes = eye_cascade.detectMultiScale(roi_gray)
           for (ex,ey,ew,eh) in eyes:
               cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
       cv.imshow('img',frame)
       if cv.waitKey(1) == 27:
           break  # esc to quit
   cv.destroyAllWindows()
