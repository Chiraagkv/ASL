import cv2
import mediapipe as mp
import numpy as np
from backend import load_model, predict_custom
import streamlit as st
from webrtc import VideoTransformer
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, ClientSettings

st.set_page_config(
    page_title='ASL to Text Translation')


st.sidebar.title('Navigation')
page = st.sidebar.selectbox("Select page:", options=[
                            "Welcome", "About", "Demo"])

if page == "Demo":
    st.title("ASL to Text Demo")
    ctxt = webrtc_streamer(key="ASL",
                           video_transformer_factory=VideoTransformer, 
                           client_settings = ClientSettings(rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                           media_stream_constraints={"video": True, "audio": False}))
    if not ctxt.video_transformer:
        st.markdown("### Press the start button to run the app :point_up:")
    

elif page == "About":
    st.title("ASL to Text")
    st.markdown("""In recent years, ASL has become the medium of communication for thousands, if not millions of people. However, very few people with no hearing impairment know the various symbols of this sign-language. This project aims to bridge the gap between those who know and those who don't know this language. 
This project uses the MobileNet architecture to recognise patterns in the video frames (which are taken from your camera) and predicts the ASL sign you are making. The model is trained on the [ASL Alphabets](https://www.kaggle.com/grassknoted/asl-alphabet) dataset. This project is deployed on Streamlit with the help of the [streamlit-webrtc library](https://discuss.streamlit.io/t/new-component-streamlit-webrtc-a-new-way-to-deal-with-real-time-media-streams/8669). \nCheck out the Demo page to test the app yourself through a real-time translation of the symbols.""")
    st.markdown("""## Authors \n1. [Chiraag  KV](https://github.com/Chiraagkv/)
    \n2. [Iva Vratric](https://github.com/idontcalculate)
    \n3. [Rohit Kumar](https://github.com/rohitkumar9989)
    \n4. [Yash Pawar](https://github.com/yashppawar)
    \n5. [Yash Vardhan](https://github.com/YashVardhan-AI)""")

elif page == "Welcome":
    st.title("Welcome!")
    st.markdown("""Hey there! This is a machine-learning powered app that detects ASL symbols. To navigate through the app, open the sidebar and choose the page you wish to go to.""")
