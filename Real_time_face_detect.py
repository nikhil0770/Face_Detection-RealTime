import cv2

video = cv2.VideoCapture("Faces from around the world.mp4")

while True:
    #img is refered as frame
    success_read,img = video.read()

    trained_data = cv2.CascadeClassifier('frontal_face.xml')
    #img = cv2.imread("hrithik.jpg")

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_data.detectMultiScale(gray_img)
    #print(face_coordinates)

    for x,y,w,h in face_coordinates:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)

    cv2.imshow("Face Detection",img)
    key = cv2.waitKey(1)

    if(key == 113 or key == 81):
        break

print("code")