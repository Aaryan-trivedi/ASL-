import cv2 as cv
import os

top, right, bottom, left = 100, 150, 400, 450
IMG_SIZE = 512
exit_con = '**'

dir0 = input('enter the directory name: ')

# Create main directory if not exists
os.makedirs(dir0, exist_ok=True)

# FIX 1: Correct camera index
vid = cv.VideoCapture(0)

if not vid.isOpened():
    print("❌ ERROR: Webcam not detected.")
    exit()

while True:
    a = input('exit: ** or enter the label name: ')

    if a == exit_con:
        break

    dir1 = os.path.join(dir0, a)
    os.makedirs(dir1, exist_ok=True)

    i = 0
    print("📸 Collecting images for label:", a)

    while True:
        ret, frame = vid.read()

        # FIX 2: Safety check
        if not ret or frame is None:
            print("❌ Failed to grab frame from webcam.")
            break

        frame = cv.flip(frame, 1)

        roi = frame[top:bottom, right:left]
        roi = cv.GaussianBlur(roi, (7, 7), 0)
        roi = cv.resize(roi, (IMG_SIZE, IMG_SIZE))

        cv.imwrite(f"{dir1}/{a}{i}.jpg", roi)
        i += 1
        print("Saved:", i)

        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv.imshow('frame', frame)
        cv.imshow('ROI', roi)

        if i >= 300:
            print("✅ 300 images captured for", a)
            break

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()
print("🎯 Dataset collection finished.")
