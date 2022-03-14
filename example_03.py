import numpy as np
import os
import tensorflow as tf
import cv2
from matplotlib.pyplot import gray

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('face_detector.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

model = tf.keras.models.load_model('checkpoint19.ckpt')

i = 0
while True:
    # Read the frame
    _, img = cap.read()

    # Detect the faces
    faces = face_cascade.detectMultiScale(img, 1.3, 4)

    # save each frame as image with PNG format
    image = cv2.imwrite('database/{index}.png'.format(index=i), img)
    i += 1

    # cut out the fragment in the box of the image
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        crop_img = img[y:y + h, x:x + w]
        resizedImg = cv2.resize(crop_img, (224, 224))
        gray = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2GRAY)
        imgArrNew = gray.reshape(1, 224, 224, 1)
        prediction = model.predict(imgArrNew)
        print(prediction)
        label = np.argmax(prediction)
        print(label)

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        for (x, y, w, h) in faces:
            org = (x, y+h+30)

        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # output the predicted label/sign on the live-stream frame
        if label == 0:
            color = (0,0,225)
            label_out = "Mask off"
        if label == 1:
            color = (50,205,50)
            label_out = "Mask on"
        if label == 2:
            color = (0,255,225)
            label_out = "incorrect Mask"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        image1 = cv2.putText(img, label_out, org, font,
                             fontScale, color, thickness, cv2.LINE_AA)
    # Display
    cv2.imshow('Face_Regonition', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
# Release the VideoCapture object
cap.release()