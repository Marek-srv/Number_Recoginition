# Take the image form the webcam and stores it

import cv2
import os

digit= int(input("Entert the number you are capturing (0-9):"))
save_dir= f"data/{digit}"
os.makedirs(save_dir, exist_ok= True)

cap= cv2.VideoCapture(0)
count= 0

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
    if key== ord('s'):
        cv2.imwrite(f"{save_dir}/{count}.jpg", roi)
        print(f"saved {count}.jpg")
        count+=1
    elif key== ord('q'):
        print("exit")
        break

cap.release()
cv2.destroyAllWindows()