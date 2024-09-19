import streamlit as st


def render_results():

    import pandas as pd
    import streamlit as st

    # 定义选项
    options = ['选项 A', '选项 B', '选项 C']

    # 创建一个 DataFrame，其中包含选项的索引
    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Choice': [0, 1, 2]  # 假设 Choice 列代表选项的索引
    })

    # 显示数据编辑器
    edited_df = st.data_editor(df)

    # 将索引映射回实际的选项文本
    def map_choice(index):
        return options[index]

    # 转换索引到实际选项
    edited_df['Choice'] = edited_df['Choice'].apply(map_choice)

    # 显示转换后的 DataFrame
    st.dataframe(edited_df)
