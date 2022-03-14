import cv2
import face_recognition

if __name__ == '__main__':
    img = cv2.imread("images/Messi.webp")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_encode = face_recognition.face_encodings(rgb_img)[0]

    img2 = cv2.imread("images/Messi 2.jpg")
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_encode2 = face_recognition.face_encodings(rgb_img2)[0]

    result = face_recognition.compare_faces([img_encode], img_encode2)
    print("Resultado: ", result)
    cv2.imshow("img", img)
    cv2.imshow("img 2", img2)
    cv2.waitKey(0)
