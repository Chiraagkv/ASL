import cv2
import mediapipe as mp
import numpy as np
from backend import load_model, predict_custom
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

model = load_model(model_path='./20210711-162248-big-one.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

class VideoTransformer(VideoTransformerBase):

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        h, w, c = frame.shape
        result = hands.process(frame)
        pred, conf = predict_custom(image=frame, model=model)
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
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (130, 255, 20), 5)
                mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
                frame = cv2.putText(frame, f"{pred}, {conf}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
        return frame       
    
