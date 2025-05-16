"""
Face Detection Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cvzone
import numpy as np


class FaceDetector:
    """
    Find faces in realtime using the light weight model provided in the mediapipe
    library.
    """

    def __init__(self, minDetectionCon=0.5, modelSelection=0):
        """
        :param minDetectionCon: Minimum confidence value ([0.0, 1.0]) for face
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/face_detection#min_detection_confidence.

        :param modelSelection: 0 or 1. 0 to select a short-range model that works
        best for faces within 2 meters from the camera, and 1 for a full-range
        model best for faces within 5 meters. See details in
        https://solutions.mediapipe.dev/face_detection#model_selection.
        """
        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection
        self.mpDraw = mp.solutions.drawing_utils
        base_options = python.BaseOptions(model_asset_path='models/blaze_face_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options,
                                             min_detection_confidence=self.minDetectionCon)  # 設置最低信心指數為0.5
        self.faceDetection = vision.FaceDetector.create_from_options(options)

    def findFaces(self, img, draw=True):
        """
        Find faces in an image and return the bbox info
        :param img: Image to find the faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings.
                 Bounding Box list.
        """

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        self.results = self.faceDetection.detect(mp_image)
        img = np.copy(mp_image.numpy_view())
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                if detection.categories[0].score > self.minDetectionCon:   
                    bboxC = detection.bounding_box
                    x1 = bboxC.origin_x
                    y1 = bboxC.origin_y
                    x2 = bboxC.origin_x + bboxC.width
                    y2 = bboxC.origin_y + bboxC.height
                    bbox = x1, y1, bboxC.width, bboxC.height
                    cx, cy = bbox[0] + (bbox[2] // 2), \
                             bbox[1] + (bbox[3] // 2)
                    bboxInfo = {"id": id, "bbox": bbox, "score": [detection.categories[0].score], "center": (cx, cy)}
                    bboxs.append(bboxInfo)
                    if draw:
                        img = cv2.rectangle(img, bbox, (255, 0, 255), 2)
                        cv2.putText(img, f'{int(detection.categories[0].score * 100)}%',
                                    (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                    2, (255, 0, 255), 2)
        return img, bboxs
    
    def getFaceKeypoints(self, img, face_idx=0):        
        detection = self.results.detections[face_idx]
        keypoints = []
        ih, iw, ic = img.shape
        keypoint_names = ['RIGHT_EYE', 'LEFT_EYE', 'NOSE_TIP', 'MOUTH_CENTER',
                          'RIGHT_EAR_TRAGION', 'LEFT_EAR_TRAGION']
        for i, keypoint in enumerate(detection.keypoints):
            cx = int(keypoint.x * iw)  
            cy = int(keypoint.y * ih)
            keypoint = {"name": keypoint_names[i], 
                        "keypoint" : (cx, cy)}
            keypoints.append(keypoint)
            
        return keypoints   

def main():
    # Initialize the webcam
    # '2' means the third camera connected to the computer, usually 0 refers to the built-in webcam
    cap = cv2.VideoCapture(2)

    # Initialize the FaceDetector object
    # minDetectionCon: Minimum detection confidence threshold
    # modelSelection: 0 for short-range detection (2 meters), 1 for long-range detection (5 meters)
    detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    # Run the loop to continually get frames from the webcam
    while True:
        # Read the current frame from the webcam
        # success: Boolean, whether the frame was successfully grabbed
        # img: the captured frame
        success, img = cap.read()

        # Detect faces in the image
        # img: Updated image
        # bboxs: List of bounding boxes around detected faces
        img, bboxs = detector.findFaces(img, draw=False)

        # Check if any face is detected
        if bboxs:
            # Loop through each bounding box
            for bbox in bboxs:
                # bbox contains 'id', 'bbox', 'score', 'center'

                # ---- Get Data  ---- #
                center = bbox["center"]
                x, y, w, h = bbox['bbox']
                score = int(bbox['score'][0] * 100)

                # ---- Draw Data  ---- #
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
                cvzone.putTextRect(img, f'{score}%', (x, y - 10))
                cvzone.cornerRect(img, (x, y, w, h))

        # Display the image in a window named 'Image'
        cv2.imshow("Image", img)
        # Wait for 1 millisecond, and keep the window open
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
