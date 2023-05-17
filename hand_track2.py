import cv2
import numpy as np
import HandDetectTrackModule as htm
import time
import mouse as ms
import pyautogui as pt
import streamlit as st

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    prev_time = 0

    det = htm.handDetector(maxHands=1)
    tips = [4, 8, 12, 16, 20]
    sm = 10
    clp_x, clp_y = 0, 0
    plp_x, plp_y = 0, 0

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = det.find_hands(frame)
        lm_list, b_box = det.find_positions(frame, draw=True)
        cv2.rectangle(frame, (100, 100), (540, 380), (45, 255, 255), 2)
        fingers = []

        if len(lm_list) != 0:
            if lm_list[tips[0]][1] > lm_list[tips[0] - 1][1]:
                fingers.append(0)
            else:
                fingers.append(1)

            for id in range(1, 5):
                if lm_list[tips[id]][2] < lm_list[tips[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]
            x3, y3 = lm_list[4][1:]

        if len(fingers) != 0:

            if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                cv2.circle(frame, (x1, y1), 10, (0, 0, 255), -1)

                x4 = np.interp(x1, (100, 540), (0, 1920))
                y4 = np.interp(y1, (100, 380), (0, 1080))
                clp_x = plp_x + (x4 - plp_x) // sm
                clp_y = plp_y + (y4 - plp_y) // sm

                ms.move(clp_x, clp_y, True)
                plp_x, plp_y = clp_x, clp_y

            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
                cv2.circle(frame, (x1, y1), 10, (0, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
                length, frame, info = det.find_distance(8, 12, frame)

                if length < 90:
                    pt.click(button='right', clicks=1, interval=0.3)

            cap.set(3, 640)
            cap.set(4, 480)
            if fingers[1] == 1 and fingers[0] == 1 and fingers[3] == 0:
                cv2.circle(frame, (x1, y1), 10, (0, 0, 255), -1)
                cv2.circle(frame, (x3, y3), 10, (255, 0, 0), -1)
                length, frame, info = det.find_distance(8, 4, frame)
                # print(length)

                if length > 135 or length < 100:
                    pt.click(button='left', clicks=1, interval=0.35)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f'FPS:{str(int(fps))}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame)
        #cv2.imshow('work_frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyWindow('work_frame')


if __name__ == "__main__":
    main()
