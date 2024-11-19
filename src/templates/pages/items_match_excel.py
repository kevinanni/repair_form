import os
import streamlit as st
import pandas as pd
from src.business_logic.repair_items import get_items_match, get_all_standard_items


def go_prev():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()
    else:
        st.warning("已经是第一条数据了！")


def go_next(max):
    if st.session_state.current_index < max - 1:
        st.session_state.current_index += 1
        st.rerun()
    else:
        st.warning("已经是最后一条数据了！")


def render_items_match():
    item_file_path = os.path.join(os.getcwd(), 'data/processed/items_test.csv')

    st.header("修理项匹配")
    df = None
    if 'df' not in st.session_state:
        df = pd.read_csv(item_file_path, encoding='gbk')
    else:
        df = st.session_state.df

    st.dataframe(df, use_container_width=True)  # 显示DataFrame，宽度占满页面

    # 点击按钮匹配修理项
    if st.button('匹配修理项'):
        result = get_items_match(df['raw_items'])
        df['std_ids'] = [match_id for _, match_id, _ in result]
        df['std_items'] = [match_text for _, _, match_text in result]
        if 'current_index' not in st.session_state or 'df' not in st.session_state:
            st.session_state.df = df  # 将DataFrame存入session
            st.session_state.current_index = 0

    # 显示当前数据
    if 'current_index' in st.session_state:
        if st.session_state.current_index < len(df):
            # 使用滑块选择当前索引
            st.session_state.current_index = st.slider(
                '选择当前修理项',
                min_value=0,
                max_value=len(df) - 1,
                value=st.session_state.current_index,
                step=1)

            row = st.session_state.df.iloc[st.session_state.current_index]

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"原始项目: {row['raw_items']}")
            with col2:
                # 选择标准修理项
                standard_items = get_all_standard_items()
                item_options = {
                    item['item_name']: item['id']
                    for item in standard_items
                }
                selected_item_id = st.selectbox(
                    "匹配的标准修理项",
                    options=list(item_options.keys()),
                    format_func=lambda x: x,
                    index=list(item_options.values()).index(row['std_ids'])
                    if row['std_ids'] else 0,
                )

            # 按钮布局
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button('上一条'):
                    go_prev()
            with col2:
                if st.button('下一条'):
                    go_next(len(df))
            with col3:
                if st.button('保存'):
                    # 将选中的标准修理项ID回写到DataFrame中
                    st.session_state.df.at[
                        st.session_state.current_index,
                        'std_ids'] = item_options[selected_item_id]
                    st.session_state.df.at[st.session_state.current_index,
                                           'std_items'] = selected_item_id
                    st.session_state.df.at[st.session_state.current_index,
                                           'is_std'] = 1
                    go_next(len(df))
            with col4:
                if st.button('找不到标准项目'):
                    st.session_state.df.at[st.session_state.current_index,
                                           'std_ids'] = ''
                    st.session_state.df.at[st.session_state.current_index,
                                           'std_items'] = ''
                    st.session_state.df.at[st.session_state.current_index,
                                           'is_std'] = 0
                    go_next(len(df))
        else:
            st.write("没有更多记录可显示。")

        if st.button('保存编辑结果'):
            st.session_state.df.to_csv(item_file_path, encoding='gbk')
            st.success(f"数据已成功保存")

        # st.dataframe(df)
        # st.write("匹配结果:")
        # for input_sentence, match_id, match_text in result:
        #     st.write(f"输入: {input_sentence}")
        #     st.write(f"匹配ID: {match_id}, 匹配项: {match_text}")

        #     if st.button('修改匹配项'):
        #         selected_item = st.selectbox("选择要修改的匹配项", [
        #             f"{input_sentence} - {match_text}"
        #             for input_sentence, match_id, match_text in result
        #         ])
        #         new_value = st.text_input("输入新的匹配项文本")
        #         if st.button('确认修改'):
        #             # 这里可以添加逻辑来处理修改后的值
        #             st.success(f"已将 '{selected_item}' 修改为 '{new_value}'")
        #     st.write("---")  # 加分隔线
