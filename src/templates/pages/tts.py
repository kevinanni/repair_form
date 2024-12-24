import streamlit as st
from src.models.modelscope import asr_pipeline

from audio_recorder_streamlit import audio_recorder



def render_tts():
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        # 利用asr_pipeline解析录音内容
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        
        result = asr_pipeline(input="temp_audio.wav")
        text = result[0]['text']

        # 在页面上面展示
        st.write("识别结果:", text)