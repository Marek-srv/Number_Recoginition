import cv2
import os

os.makedirs("pred_check", exist_ok= True)
cap= cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()
    if not ret:
        break
    frame= cv2.flip(frame, 1)
    
    x1, y1, x2, y2 = 100, 100, 1250, 1250
    roi= frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)

    key= cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("pred_check/check_number.jpg", roi)
        print("saved")
        break
    elif key == ord('q'):
        print("Exit")
        break

cap.release()
cv2.destroyAllWindows()