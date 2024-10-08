# Import all of the dependencies
import streamlit as st
import os 
import tensorflow as tf  
from utils import load_data, num_to_char
from modelutil import load_model
# import imageio
# import PIL
# from PIL import Image
# from matplotlib import cm
# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')
import numpy as np
# Setup the sidebar
with st.sidebar: 
    st.title('LipReading Master')
    st.image('logo.png')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('A Full Stack App for LipReading') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)
#st.info(selected_video)
# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The selected video in mp4 format is displayed below')
        file_path = os.path.join('..','data','s1', selected_video)
        #st.info(file_path)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test.mp4 -y')
        
        video = open('test.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        # st.info(type(video))
        # x=video.numpy()
        # st.info(type(x))
    
        # im = Image.fromarray(np.uint8(cm.gist_earth(x)*255))
        #imageio.mimsave('animation.gif',x , duration=10)
        st.image('animation.gif', width=400) 

        #st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        #st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
