import cv2

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()
    print(ret)
    cv2.imshow("frame", frame)
    k = cv2.waitKey(10)
    if k is 27:
        cv2.imwrite("park_t50.png", frame)
        break
cv2.destroyAllWindows()