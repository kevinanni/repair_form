import os
import streamlit as st
import pandas as pd
from src.business_logic.repair_items import get_items_match, get_all_standard_items, \
get_corpus_for_match, update_corpus_from_df, get_standard_item, export_modelscope_dataset

import pdb

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

# 写一个函数，输入df和一个集合，元素包括df的行号和对应的std_id，输出是更新是否成功
def update_df_with_std_id(df, matches):
    """
    根据标准项目ID更新DataFrame中的数据
    
    Args:
        df: 需要更新的DataFrame
        matches: 包含(行号,std_id)元组的集合
        
    Returns:
        bool: 更新是否成功
    """
    # 遍历，根据std_id，去RepairItem中获取对应的记录，取出item_name和item_combine
    for row_idx, std_id in matches:
        std_item = get_standard_item(std_id)
        
        # 没找到，就更新失败，退出
        if not std_item:
            return False
            
        # 找到了，就用取出的值更新df中对应的值
        df.loc[row_idx, 'std_id'] = std_item.std_id
        df.loc[row_idx, 'std_combine'] = std_item.std_combine
        df.loc[row_idx, 'std_name'] = std_item.std_name
    
    # 返回成功
    return True

def render_items_match():

    st.header("修理项匹配")
    df = None
    if 'df' not in st.session_state:
        df = get_corpus_for_match()
    else:
        df = st.session_state.df

    st.dataframe(df[['item_name', 'is_std', 'std_name']], use_container_width=True)  # 显示DataFrame，宽度占满页面

    # 点击按钮匹配修理项
    if st.button('匹配修理项'):
        # 只对std_id为空的行进行匹配
        empty_mask = df['std_id'].isna()
        if empty_mask.any():
            # 只获取需要匹配的行的item_combine
            results = get_items_match(df.loc[empty_mask, 'item_combine'])
            # 将匹配结果更新到对应的行
            matches = set(zip(df[empty_mask].index, [match_id for _, match_id, _ in results]))
            update_df_with_std_id(df, matches)

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

            st.write(f"原始项目: {row['item_combine']}")
            # 选择标准修理项
            standard_items = get_all_standard_items()
            item_options = {
                item['item_combine']: item['id']
                for item in standard_items
            }
            selected_item = st.selectbox(
                "匹配的标准修理项", 
                options=list(item_options.keys()),
                format_func=lambda x: x,
                index=list(item_options.values()).index(row['std_id'])
                if row['std_id'] else 0,
            )
            selected_item_id = item_options[selected_item]
            st.write(f"原始项目: {row['item_combine']}")
            # 显示匹配状态
            status_map = {1: "已匹配", 0: "无法匹配", None: "未匹配"}
            status = status_map.get(row['is_std'], "未匹配")
            st.write(f"匹配状态: {status}")

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
                    # 将选中的标准修理项ID回写到DataFrame中-kevin
                    idx = st.session_state.current_index
                    
                    # pdb.set_trace()
                    update_df_with_std_id(st.session_state.df, [(idx, selected_item_id)])
                    st.session_state.df.at[st.session_state.current_index,
                                           'is_std'] = 1
                    go_next(len(df))
            with col4:
                if st.button('找不到标准项目'):
                    # st.session_state.df.at[st.session_state.current_index,
                    #                        'std_ids'] = ''
                    # st.session_state.df.at[st.session_state.current_index,
                    #                        'std_items'] = ''
                    st.session_state.df.at[st.session_state.current_index,
                                           'is_std'] = 0
                    go_next(len(df))
        else:
            st.write("没有更多记录可显示。")

        if st.button('保存编辑结果'):

            # st.session_state.df.to_csv(item_file_path, encoding='gbk')
            if update_corpus_from_df(st.session_state.df):
                st.success(f"数据已成功保存")
        
        # 添加导出数据集按钮
        if st.button('导出到ModelScope数据集'):
            # 设置默认导出路径
            export_path = os.path.join(os.getcwd(), 'data/dataset/repair-form')
            
            # 调用导出函数
            if export_modelscope_dataset(st.session_state.df, export_path):
                st.success(f"数据集已成功导出到 {export_path}")
            else:
                st.error("导出数据集失败，请检查日志")

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
