
import cv2
import numpy as np
import mediapipe as mp
from collections import deque


b_ind = 0
g_ind = 0
r_ind = 0
y_ind = 0



bp = [deque(maxlen=1024)]
gp = [deque(maxlen=1024)]
rp = [deque(maxlen=1024)]
yp = [deque(maxlen=1024)]


kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0
m_hands = mp.solutions.hands
hands = m_hands.Hands(max_num_hands=1, min_detection_confidence=0.70)
mpDraw = mp.solutions.drawing_utils


Digital_Canvas = np.zeros((471,636,3)) + 255
Digital_Canvas = cv2.rectangle(Digital_Canvas, (40,1), (140,65), (0,0,0), 2)
Digital_Canvas = cv2.rectangle(Digital_Canvas, (160,1), (255,65), (255,0,0), 2)
Digital_Canvas = cv2.rectangle(Digital_Canvas, (275,1), (370,65), (0,255,0), 2)
Digital_Canvas = cv2.rectangle(Digital_Canvas, (390,1), (485,65), (0,0,255), 2)
Digital_Canvas = cv2.rectangle(Digital_Canvas, (505,1), (600,65), (0,255,255), 2)

cv2.putText(Digital_Canvas, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(Digital_Canvas, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(Digital_Canvas, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(Digital_Canvas, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(Digital_Canvas, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)







cap = cv2.VideoCapture(0)
ret = True
while ret:
    #
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
    frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
    frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
   
    result = hands.process(framergb)

   
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx, lmy])


            
            mpDraw.draw_landmarks(frame, handslms, m_hands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        print(center[1]-thumb[1])
        if (thumb[1]-center[1]<30):
            bp.append(deque(maxlen=512))
            b_ind += 1
            gp.append(deque(maxlen=512))
            g_ind += 1
            rp.append(deque(maxlen=512))
            r_ind += 1
            yp.append(deque(maxlen=512))
            y_ind += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear Button
                bp = [deque(maxlen=512)]
                gp = [deque(maxlen=512)]
                rp = [deque(maxlen=512)]
                yp = [deque(maxlen=512)]

                b_ind = 0
                g_ind = 0
                r_ind = 0
                y_ind = 0

                Digital_Canvas[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    colorIndex = 0 # Blue
            elif 275 <= center[0] <= 370:
                    colorIndex = 1 # Green
            elif 390 <= center[0] <= 485:
                    colorIndex = 2 # Red
            elif 505 <= center[0] <= 600:
                    colorIndex = 3 # Yellow
        else :
            if colorIndex == 0:
                bp[b_ind].appendleft(center)
            elif colorIndex == 1:
                gp[g_ind].appendleft(center)
            elif colorIndex == 2:
                rp[r_ind].appendleft(center)
            elif colorIndex == 3:
                yp[y_ind].appendleft(center)
   
    else:
        bp.append(deque(maxlen=512))
        b_ind += 1
        gp.append(deque(maxlen=512))
        g_ind += 1
        rp.append(deque(maxlen=512))
        r_ind += 1
        yp.append(deque(maxlen=512))
        y_ind += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bp, gp, rp, yp]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(Digital_Canvas, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame) 
    cv2.imshow("Paint", Digital_Canvas)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
