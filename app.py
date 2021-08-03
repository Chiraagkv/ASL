import cv2
import mediapipe as mp
import numpy as np
from backend import load_model, predict_custom
import streamlit as st

st.title("ASL to Text")
FRAME_WINDOW = st.image([])
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
user_input = st.text_input("Select camera source number:", "0")
try:
  cap_num = int(user_input)
except ValueError:
  cap_num = 0

cap = cv2.VideoCapture(cap_num, cv2.CAP_DSHOW)
run = st.checkbox('Run')

model = load_model(model_path='C:\\Users\\abc\\OneDrive\\Desktop\\ASL_deploy\\my_model.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils


while run:
    _, frame = cap.read()
    h, w, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    path = 'C:\\Users\\abc\\OneDrive\\Desktop\\ASL_deploy\\images\\img.jpg'
    cv2.imwrite(path, frame)
    pred, conf = predict_custom(image=path, model=model)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (130, 255, 20), 5)
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
            frame = cv2.putText(frame, f"{pred}, {conf}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
  st.markdown("### Check the box to run the app :point_up:")


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
