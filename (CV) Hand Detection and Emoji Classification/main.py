import cv2
import numpy as np
import mediapipe as mp

# hand detection system
mp_hands = mp.solutions.hands  # hands module
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils  # hand landmarks

def main():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("video not established. exitiing...")
        exit()
    
    while True:
        success, frame = video.read()
        if not success:
            print("stream end")
            break
        
        black_frame = np.zeros((480, 640, 3), np.uint8)
        coloredFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detectionResults = hands.process(coloredFrame)

        if detectionResults.multi_hand_landmarks:
            for landmark in detectionResults.multi_hand_landmarks:
                mp_draw.draw_landmarks(black_frame, landmark, mp_hands.HAND_CONNECTIONS)

                #coords = landmark.landmark[mp.hands.HandLandmark.WRIST]

        cv2.imshow("Hand Detection", black_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
