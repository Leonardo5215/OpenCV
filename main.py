import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 500)
cap.set(4, 600)

MaskCascade = cv2.CascadeClassifier("cascade.xml")

while True:
    cv2.waitKey(1000)
    
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(imgGray, (5, 5), 0)

    mask = MaskCascade.detectMultiScale(imgGray, 1.1, 5)
    for (x, y, w, h) in mask:
        area = (x+h) * (y+w)
        if area > 1000:
            cv2.rectangle(img, (x, y), (x+h, y+w), (255, 0, 0), 3)

    cv2.imshow("Video", img)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

    print('Hay', len(mask), 'aqui')

