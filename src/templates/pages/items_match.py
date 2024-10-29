import os
import streamlit as st
import pandas as pd
from src.business_logic.repair_items import get_items_match


def render_items_match():
    item_file_path = os.path.join(os.getcwd(), 'data/processed/items_test.csv')

    st.header("修理项匹配")
    df = pd.read_csv(item_file_path, encoding='gbk')

    st.dataframe(df)  # 显示DataFrame
    if st.button('点击按钮匹配修理项'):
        result = get_items_match(df['raw_items'])
        st.write("匹配结果:")
        for input_sentence, match_id, match_text in result:
            st.write(f"输入: {input_sentence}")
            st.write(f"匹配ID: {match_id}, 匹配项: {match_text}")

            if st.button('修改匹配项'):
                selected_item = st.selectbox("选择要修改的匹配项", [
                    f"{input_sentence} - {match_text}"
                    for input_sentence, match_id, match_text in result
                ])
                new_value = st.text_input("输入新的匹配项文本")
                if st.button('确认修改'):
                    # 这里可以添加逻辑来处理修改后的值
                    st.success(f"已将 '{selected_item}' 修改为 '{new_value}'")
            st.write("---")  # 加分隔线
