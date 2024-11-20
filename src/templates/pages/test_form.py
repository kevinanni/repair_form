import streamlit as st
import pandas as pd
from src.utils.item_excel_parser import ItemExcelParser
from src.business_logic.repair_items import insert_repair_order
import os
from src.models.ms_audio import inference_pipeline

from audio_recorder_streamlit import audio_recorder



def render_test_form():
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        # 利用inference_pipeline解析录音内容
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        
        result = inference_pipeline(input="temp_audio.wav")
        text = result[0]['text']

        # 在页面上面展示
        st.write("识别结果:", text)

def audio_test():
    # Add JavaScript code for audio recording
    st.markdown("""
    <script>
        let mediaRecorder;
        let audioChunks = [];
        
        async function startRecording() {
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = (e) => {
                audioChunks.push(e.data);
            };
            mediaRecorder.start();
        }
        
        function stopRecording() {
            return new Promise(resolve => {
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        const base64Audio = reader.result;
                        window.parent.postMessage({type: "audio_data", data: base64Audio}, "*");
                    };
                };
                mediaRecorder.stop();
            });
        }
    </script>
    
    <button onclick="javascript:startRecording()">开始录音</button>
    <button onclick="javascript:stopRecording()">停止录音</button>
    """, unsafe_allow_html=True)

    # Create a placeholder for the transcribed text
    text_placeholder = st.empty()
    
    # Get the audio data from the frontend
    if 'audio_data' in st.session_state:
        audio_data = st.session_state.audio_data
        
        # Save audio data to temporary file
        temp_audio_file = "temp_recording.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(audio_data)
            
        # Process audio with the model
        rec_result = inference_pipeline(input=temp_audio_file)
        
        # Display the transcribed text
        text_placeholder.write(f"识别结果: {rec_result['text']}")
        
        # Clean up temporary file
        os.remove(temp_audio_file)
    
def import_ex():
    if st.button('开始导入'):

        # try:

        # 示例路径，实际使用时需要替换为真实路径
        template_path = os.path.join(os.getcwd(), 'data', 'raw', '宜宾局-模板.xlsx')
        data_file_path = os.path.join(os.getcwd(), 'data', 'raw',
                                      '宜宾局-2023.xlsx')

        # 创建解析器实例并解析Excel文件
        parser = ItemExcelParser()
        parser.parse_template(template_path)
        data_list = parser.parse_excel_file(data_file_path)

        # print(df_list)

        for data in data_list:
            st.dataframe(data['items'])
            insert_repair_order(data)
            st.success('导入成功')
