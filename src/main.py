import streamlit as st
from src.templates.pages.history_form import render_history_form
from src.templates.pages.fuel_predict import render_fuel_predict
from src.templates.pages.results import render_results


def main_app():
    # 侧边栏导航
    page = st.sidebar.selectbox("选择页面", ["历史修理单", "油耗预测", "Results"])

    if page == "历史修理单":
        render_history_form()
    elif page == "油耗预测":
        render_fuel_predict()
    elif page == "Results":
        render_results()


if __name__ == "__main__":
    main_app()
