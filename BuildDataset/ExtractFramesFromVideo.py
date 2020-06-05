import cv2
 
# Opens the Video file
cap= cv2.VideoCapture("..\\..\\RealTime Video\\CV_2_cropped.MP4")
crop = "No"
i=1
while(cap.isOpened()):
    ret, frame = cap.read()
    if (crop == "Yes"):
        crop_img = frame[20:20+600, int(1920/2) - int(1066/2) : int(1920/2) + int(1066/2)]
        crop_img = cv2.resize(crop_img, (960, 540), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey()
    if ret == False:
        break
    if i%50 == 0 and i%30 != 0:
        cv2.imwrite("..\\..\\UpdatedDataset\\Test\\Original\\" + "frame" + str(i) + ".png", frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()