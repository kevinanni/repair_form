import streamlit as st

from src.business_logic.consumption import show_fuel_predict, consumption_train
from src.utils.custom_utils import streamlit_metrics


def render_fuel_predict():

    if st.button("训练模型"):
        # 初始化损失数据
        streamlit_metrics.loss_history = []
        # 创建一个用于展示损失变化的图表
        streamlit_metrics.chart = st.line_chart([])
        # 创建一个用于展示当前迭代次数的文本框
        streamlit_metrics.iteration_text = st.empty()
        consumption_train(is_save=True)

    if st.button("显示预测结果"):
        df_pred = show_fuel_predict()
        st.write(df_pred)
