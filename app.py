from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import streamlit as st
import tensorflow as tf
import numpy as np

trained_model = tf.keras.models.load_model('facial_emotions_model.h5')
classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class VideoProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")
		gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)


		faces = cascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)

		for x,y,w,h in faces:
			cv2.rectangle(frm, (x,y), (x+w, y+h), (0,255,0), 3)
			frm = frm[y:y+h, x:x+h]

		return av.VideoFrame.from_ndarray(frm, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)
