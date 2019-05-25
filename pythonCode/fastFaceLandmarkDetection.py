import cv2
import dlib
import sys
import numpy as np

#draw polylines
def drawPolyline(image, landmarks, start, end, isClosed=False):
        points = []
        for i in range(start, end+1):
                point = [landmarks.part(i).x, landmarks.part(i).y]
                points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], isClosed, (0, 255, 255), 2, 16)


def drawPolylines(image, landmarks):
        drawPolyline(image, landmarks, 0, 16)           # Jaw line
        drawPolyline(image, landmarks, 17, 21)          # Left eyebrow
        drawPolyline(image, landmarks, 22, 26)          # Right eyebrow
        drawPolyline(image, landmarks, 27, 30)          # Nose bridge
        drawPolyline(image, landmarks, 30, 35, True)    # Lower nose
        drawPolyline(image, landmarks, 36, 41, True)    # Left eye
        drawPolyline(image, landmarks, 42, 47, True)    # Right Eye
        drawPolyline(image, landmarks, 48, 59, True)    # Outer lip
        drawPolyline(image, landmarks, 60, 67, True)    # Inner lip

#finds face landmark points and draw polylines around face landmarks
def findFaceLandmarksAndDrawPolylines(frame, faceDetector, landmarkDetector, faces, resizeScale, skipFrames, frameCounter):
        #convert to dlib image format
        dlibImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #create resized image
        frameSmall = cv2.resize(frame, None, fx=1.0/resizeScale, fy=1.0/resizeScale)

        #convert resize image to dlib image format
        dlibImageSmall = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)
        
        #detect faces at interval of skipFrames
        if(frameCounter % skipFrames == 0):
                faces = faceDetector(dlibImageSmall, 0)
        
        #loop over faces
        for face in faces:
                #scale the rectangle coordinates as we did face detection on resized smaller image
                rect = dlib.rectangle(int(face.left() * resizeScale),
                                        int(face.top() * resizeScale),
                                        int(face.right() * resizeScale),
                                        int(face.bottom() * resizeScale))
                
                #Face landmark detection
                faceLandmarks = landmarkDetector(dlibImage, rect)

                #draw poly lines around face landmarks
                drawPolylines(frame, faceLandmarks)

        return faces


#create a video capture object
videoCapture = cv2.VideoCapture(-1)

#check if camera is opened
if videoCapture.isOpened() is False:
        print("can not open camera")
        sys.exit()

#define face detector
faceDetector = dlib.get_frontal_face_detector()

#define landmark detector and load face landmark model
landmarkDetector = dlib.shape_predictor("../dlibAndModel/shape_predictor_68_face_landmarks.dat")

#define resize height
resizeHeight = 480

#define skip frames
skipFrames = 3

#get first frame
ret, frame = videoCapture.read()

if(ret == True):
        height = frame.shape[0]
        #calculate resize scale
        resizeScale = float(height)/resizeHeight
else:
        print("can read frame from camera")
        sys.exit()

#start the tick counter
tick = cv2.getTickCount() 

frameCounter = 0

#starting dummy fps
fps = 30.0

#create window to display image
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

#to store faces detected
faces = []

while(True):
        if(frameCounter == 0):
                tick = cv2.getTickCount()

        ret, frame = videoCapture.read()

        if(ret != True):
               print("can read frame from camera")
               sys.exit() 
        else:
                faces = findFaceLandmarksAndDrawPolylines(frame, faceDetector, landmarkDetector, faces, 
                        resizeScale, skipFrames, frameCounter)

                print(len(faces))
                
                #draw fps over image
                cv2.putText(frame, str(fps), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)

                #show image
                cv2.imshow("frame", frame)

                #press esc to exit the program
                key = cv2.waitKey(1) & 0xff
                if(key == 27):
                        sys.exit()
                
                frameCounter = frameCounter + 1

                #calculate fps after every 100 frames
                if(frameCounter == 100):
                        tick = (cv2.getTickCount() - tick)/cv2.getTickFrequency()
                        fps = 100.0/tick
                        frameCounter = 0

#release video capture object
videoCapture.release()

#close all the opened windows
cv2.destroyAllWindows()
