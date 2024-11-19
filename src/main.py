import streamlit as st
from src.templates.pages.history_form import render_history_form
from src.templates.pages.fuel_predict import render_fuel_predict
from src.templates.pages.results import render_results
from src.templates.pages.items_match import render_items_match
from src.templates.pages.test_form import render_test_form


def main_app():

    # 设置侧边栏宽度最小
    st.markdown("""
        <style>
            [data-testid="stSidebar"][aria-expanded="true"]{
                min-width: 150px;
                max-width: 150px;
            }
            .main .block-container {
                max-width: 1000px;
                padding-top: 2rem;
                padding-right: 2rem;
                padding-left: 2rem;
                padding-bottom: 2rem;
            }
        </style>
    """,
                unsafe_allow_html=True)

    # 侧边栏导航
    page = st.sidebar.selectbox("选择页面", ["修理项匹配", "历史修理单", "油耗预测", "测试"])

    if page == "修理项匹配":
        render_items_match()
    elif page == "历史修理单":
        render_history_form()
    elif page == "油耗预测":
        render_fuel_predict()
    elif page == "测试":
        render_test_form()


if __name__ == "__main__":
    main_app()
