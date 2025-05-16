"""
Hand Tracking Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import math

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# 設定 Matplotlib 圖表參數，隱藏軸線與刻度標籤，使影像顯示更清楚
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):

        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param modelComplexity: Complexity of the hand landmark model: 0 or 1.
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.staticMode,
                                        max_num_hands=self.maxHands,
                                        model_complexity=modelComplexity,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True, scaleZ=1):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        self.lmList = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w * scaleZ)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                self.lmList.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return self.lmList, img
        else:
            return self.lmList

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        fingers = []
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:

            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Find the distance between two landmarks input should be (x1,y1) (x2,y2)
        :param p1: Point1 (x1,y1)
        :param p2: Point2 (x2,y2)
        :param img: Image to draw output on. If no image input output img is None
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """
        if len(p1) == len(p2) == 3:
            # Get the landmarks
            x1, y1 = p1[0:2]
            x2, y2 = p2[0:2]
        else:
            x1, y1 = p1
            x2, y2 = p2

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

            return length, info, img
        else:
            return length, info

    def findDistance3D(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
           Find the distance between two landmarks input should be (x1,y1,z1) (x2,y2,z2)
           :param p1: Point1 (x1,y1,z1)
           :param p2: Point2 (x2,y2,z2)
           :param img: Image to draw output on. If no image input output img is None
           :return: Distance between the points
                    Image with output drawn
                    Line information
           """
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        cx, cy, cz = (x1 + x2) // 2, (y1 + y2) // 2, (z1 + z2) // 2
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        info = (x1, y1, z1, x2, y2, z2, cx, cy, cz)

        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

            return length, info, img
        else:
            return length, info

    def findAngle(self,p1, p2, p3, img=None, color=(255, 0, 255), scale=5):
        """
        Finds angle between three points.

        :param p1: Point1 - (x1,y1,z1)
        :param p2: Point2 - (x2,y2,z2)
        :param p3: Point3 - (x3,y3,z3)
        :param img: Image to draw output on. If no image input output img is None
        :return:
        """        
        if img is not None:
            angle, img = self.findAngle3D(p1[0:2], p2[0:2], p3[0:2],
                                      img=img,
                                      color=color,
                                      scale=scale)
            return angle, img
        else:
            angle = self.findAngle3D(p1[0:2], p2[0:2], p3[0:2],
                                      color=color,
                                      scale=scale)
            return angle

    def findAngle3D(self, point_a, point_b, point_c, img=None, color=(255, 0, 255), scale=5):
        """
        Finds angle between three points.

        :param point_a: Point1 - (x1,y1,z1)
        :param point_b: Point2 - (x2,y2,z2)
        :param point_c: Point3 - (x3,y3,z3)
        :param img: Image to draw output on. If no image input output img is None
        :return:   
        """
        a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # a、b、c三個點的x座標
        a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # a、b、c三個點的y座標

        if len(point_a) == len(point_b) == len(point_c) == 3:
            a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # a、b、c三個點的z座標
        else:
            a_z, b_z, c_z = 0,0,0  # 因為是二維座標, 所以z軸都為0

        # 向量 m=(x1,y1,z1), n=(x2,y2,z2)
        x1,y1,z1 = (a_x-b_x),(a_y-b_y),(a_z-b_z)
        x2,y2,z2 = (c_x-b_x),(c_y-b_y),(c_z-b_z)

        # 計算角度
        cos_b = (x1*x2 + y1*y2 + z1*z2) / (math.sqrt(x1**2 + y1**2 + z1**2) *(math.sqrt(x2**2 + y2**2 + z2**2))) 
        angle = math.degrees(math.acos(cos_b))
        
        # Draw
        if img is not None:
            cv2.line(img, (a_x, a_y), (b_x, b_y), (255, 255, 255), max(1,scale//5))
            cv2.line(img, (c_x, c_y), (b_x, b_y), (255, 255, 255), max(1,scale//5))
            cv2.circle(img, (a_x, a_y), scale, color, cv2.FILLED)
            cv2.circle(img, (a_x, a_y), scale+5, color, max(1,scale//5))
            cv2.circle(img, (b_x, b_y), scale, color, cv2.FILLED)
            cv2.circle(img, (b_x, b_y), scale+5, color, max(1,scale//5))
            cv2.circle(img, (c_x, c_y), scale, color, cv2.FILLED)
            cv2.circle(img, (c_x, c_y), scale+5, color, max(1,scale//5))
            cv2.putText(img, str(int(angle)), (b_x - 50, b_y + 50 + scale * 2),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, max(1,scale//5))
        
            return angle, img
        else:
            return angle

    def angleCheck(self, myAngle, targetAngle, offset=20):
        return targetAngle - offset < myAngle < targetAngle + offset

    def plotHandLandmarks3D(self, box_aspect=[1, 1, 1], elev=300, azim=330, roll=300):
        # 定義手掌和手指關鍵點索引
        thumb_indices = [0, 1, 2, 3, 4]             # 姆指
        index_finger_indices = [5, 6, 7, 8]         # 食指
        middle_finger_indices = [9, 10, 11, 12]     # 中指
        ring_finger_indices = [13, 14, 15, 16]      # 無名指
        pinky_indices = [17, 18, 19, 20]            # 小指
        palm_indices = [0, 5, 9, 13, 17, 0]         # 手掌 
        
        finger_lists = [thumb_indices, index_finger_indices, middle_finger_indices,
                        ring_finger_indices, pinky_indices, palm_indices]
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')      
        
        for hand in self.lmList:
            lmList = hand["lmList"]
            # 提取手勢座標
            x = [landmark[0] for landmark in lmList]
            y = [landmark[1] for landmark in lmList]
            z = [landmark[2] for landmark in lmList]
            # 繪製手指點
            ax.scatter(x, y, z)
            # 繪製手指連接
            for finger_points in finger_lists:
                ax.plot([x[i] for i in finger_points], [y[i] for i in finger_points],
                        [z[i] for i in finger_points], 'r')

        ax.view_init(elev=elev, azim=azim, roll=roll)
        ax.set_box_aspect(box_aspect)

        plt.show()
        
def main():
    # Initialize the webcam to capture video
    # The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
    cap = cv2.VideoCapture(0)

    # Initialize the HandDetector class with the given parameters
    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()

        # Find hands in the current frame
        # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
        # The 'flipType' parameter flips the image, making it easier for some detections
        hands, img = detector.findHands(img, draw=True, flipType=True)

        # Check if any hands are detected
        if hands:
            # Information for the first hand detected
            hand1 = hands[0]  # Get the first hand detected
            lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand
            bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
            center1 = hand1['center']  # Center coordinates of the first hand
            handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

            # Count the number of fingers up for the first hand
            fingers1 = detector.fingersUp(hand1)
            print(f'H1 = {fingers1.count(1)}', end=" ")  # Print the count of fingers that are up

            # Calculate distance between specific landmarks on the first hand and draw it on the image
            length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[12][0:2], img, color=(255, 0, 255),
                                                      scale=10)

            # Check if a second hand is detected
            if len(hands) == 2:
                # Information for the second hand
                hand2 = hands[1]
                lmList2 = hand2["lmList"]
                bbox2 = hand2["bbox"]
                center2 = hand2['center']
                handType2 = hand2["type"]

                # Count the number of fingers up for the second hand
                fingers2 = detector.fingersUp(hand2)
                print(f'H2 = {fingers2.count(1)}', end=" ")

                # Calculate distance between the index fingers of both hands and draw it on the image
                length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img, color=(255, 0, 0),
                                                          scale=10)

            print(" ")  # New line for better readability of the printed output

        # Display the image in a window
        cv2.imshow("Image", img)

        # Keep the window open and update it for each frame; wait for 1 millisecond between frames
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
