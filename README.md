# poseDetection
import cv2
import mediapipe as mp
import  time

pTime = 0
cap = cv2.VideoCapture(r"C:\Users\victus\OneDrive\Desktop\SHIVA\venv\.vscode\project\project videos\4.mp4")

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection )
            print(id, detection)
            print(detection.score)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih ,iw, ic = img.shape
            bbox =int (bboxC.xmin * iw), int (bboxC.ymin * ih), \
                   int (bboxC.width * iw), int (bboxC.height *ih)
            cv2.rectangle(img, bbox, (250,250,0), 2)
            cv2.putText(img,f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_COMPLEX, 2,(200,0,255), 2)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_COMPLEX, 3,(200,0,255), 2)
    cv2.imshow("image", img)
    cv2.waitKey(10)
