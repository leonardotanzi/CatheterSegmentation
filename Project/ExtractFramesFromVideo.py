import cv2
 
# Opens the Video file
cap= cv2.VideoCapture("..\\RealTime Video\\CV_2_cropped.mp4")
i=1
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%15 == 0 and i%10 != 0:
        cv2.imwrite("C:\\Users\\d053175\\Desktop\\Prostate\\Catetere\\" + "frame" + str(i) + ".png", frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()