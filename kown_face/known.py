import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    labels = []
    label = 0
    for subdir in os.listdir(folder):
        subfolder = os.path.join(folder, subdir)
        if os.path.isdir(subfolder):
            for filename in os.listdir(subfolder):
                img_path = os.path.join(subfolder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    images.append(gray_img)
                    labels.append(label)
            label += 1
    return images, labels

# 从文件夹中读取不同个人的人脸照片用于训练并分出标签
known_faces_folder = 'D:\\VsCode\\kown_face\\knowfaces'
known_faces, labels = load_images_from_folder(known_faces_folder)

# 训练LBPH人脸识别器
recognizer = cv2.face.LBPHFaceRecognizer_create() # type: ignore
recognizer.train(known_faces, np.array(labels))

# 加载Haar级联分类器，直接下载OpenCV仓库提供的数据
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # type: ignore

# 捕获摄像头视频
capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        label, confidence = recognizer.predict(face_roi)
        # 调整信心阈值
        if confidence < 75:
            label_text = f"Known Face id:{label}"
        else:
            label_text = "Unknown Face"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('camera', frame)
    key = cv2.waitKey(1)
    if key != -1:
        break

capture.release()
cv2.destroyAllWindows()
