import cv2
import mediapipe as mp
import time
import math


class handDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon = 1, trackCon = 0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(frameRGB)
        if self.result.multi_hand_landmarks:
            for handlms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handlms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def find_positions(self, frame, handNo=0, draw=True):
        self.lm_list = []
        x_list = []
        y_list = []
        b_box = []
        if self.result.multi_hand_landmarks:
            my_hand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(my_hand.landmark):
                w, h, c = frame.shape
                cx, cy = int(lm.x * h), int(lm.y * w)
                self.lm_list.append([id, cx, cy])
                x_list.append(cx)
                y_list.append(cy)
                if draw:
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -1)

            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            b_box = x_min, y_min, x_max, y_max

            if draw:
                cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

        return self.lm_list, b_box

    def find_distance(self, p1, p2, frame, draw=True, r=10, t=1):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:

            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]

def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    det = handDetector()

    while True:
        ret, frame = cap.read()
        frame = det.find_hands(frame)
        lm_list = det.find_positions(frame)
        if len(lm_list) != 0:
            print(lm_list[4])

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow("IMAGE", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
