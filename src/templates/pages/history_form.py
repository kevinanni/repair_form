import streamlit as st
from src.business_logic.repair_items import get_matching_items


def show_match():
    st.session_state.match_clicked = True


def render_history_form():
    st.header("历史修理单")
    uploaded_file = st.file_uploader("上传历史修理单pdf或csv文件", type=["pdf", "csv"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            # 如果是pdf，调用tabula转换出excel
            import tabula
            df = tabula.read_pdf(uploaded_file,
                                 pages='all')[0]  # 读取PDF并转换为DataFrame
        else:
            # 处理csv文件
            import pandas as pd
            df = pd.read_csv(
                uploaded_file, encoding='gbk'
            )  # 读取csv文件为DataFrame，使用latin1编码以避免UnicodeDecodeError

        # 显示历史修理单表格
        # st.dataframe(df)  # 显示DataFrame
        st.data_editor(df)  # 显示DataFrame

        if 'match_clicked' not in st.session_state:
            st.session_state.match_clicked = False

        st.button('匹配修理单项目', on_click=show_match)

        if st.session_state.match_clicked:

            match_results = []
            select_results = []

            for index, row in df.iterrows():
                input_sentence = row['维修内容']
                matches = get_matching_items(input_sentence, top_n=3)  # 获取匹配结果
                match_results.append((input_sentence, matches))

            for input_sentence, matches in match_results:
                st.write(f"输入: {input_sentence}")

                # 在循环内部记录radio的选择
                if f"selected_match_{input_sentence}" not in st.session_state:
                    st.session_state[
                        f"selected_match_{input_sentence}"] = matches[0][
                            0]  # 默认选择第一个匹配项

                selected_match = st.radio(
                    "选择匹配项:",
                    options=[match[0] for match in matches],
                    index=[match[0] for match in matches].index(
                        st.session_state[f"selected_match_{input_sentence}"]
                    ),  # 根据session_state的选择设置默认选项
                    key=input_sentence,  # 添加唯一的key以避免DuplicateWidgetID错误
                    on_change=lambda input_sentence=input_sentence: st.
                    session_state.update({
                        f"selected_match_{input_sentence}":
                        st.session_state[f"selected_match_{input_sentence}"]
                    })  # 更新session_state中的选择
                )

                selected_match_index = next(
                    idx for idx, match in enumerate(matches)
                    if match[0] == selected_match)
                st.write(
                    f"选择的匹配项: {selected_match} (相似度: {matches[selected_match_index][1]:.2f})"
                )

                select_results.append((input_sentence, selected_match))
                st.markdown("---")  # 添加分割线

            # 添加按钮以显示最终匹配结果
            if st.button("显示最终匹配结果"):
                # 展示匹配结果的表格
                st.write("最终的匹配结果:")
                results_df = pd.DataFrame(select_results,
                                          columns=["原始内容", "匹配结果"])
                st.dataframe(results_df)


#     uploaded_pdf = st.file_uploader("上传历史修理单pdf", type=["pdf"])
#     if uploaded_pdf is not None:
#         # 将上传的pdf文件保存到临时位置
#         with open("temp_file.pdf", "wb") as f:
#             f.write(uploaded_pdf.getbuffer())

#         # 访问后端接口，解析pdf并回传表格数据
#         result, error = fetch_data_from_backend('/parse_pdf',
#                                                 method='POST',
#                                                 files={"file": uploaded_pdf})
#         if error:
#             st.error(error)
#         else:
#             # 显示表格数据并允许编辑
#             df = pd.DataFrame(result)  # 假设result是表格数据
#             edited_df = st.data_editor(df)

#             if st.button("匹配修理单项目"):
#                 # 点击匹配按钮，从后端逐条匹配修理单项目
#                 match_results = []
#                 for index, row in edited_df.iterrows():
#                     match_result, match_error = fetch_data_from_backend(
#                         f'/match_item/{row["item_id"]}', method='GET')
#                     if match_error:
#                         st.error(match_error)
#                     else:
#                         match_results.append(match_result)

#                 # 对于不正确的匹配，弹出窗口修改
#                 for match in match_results:
#                     if not match['is_correct']:
#                         corrected_value = st.text_input(
#                             f"修改匹配项: {match['item_name']}",
#                             value=match['suggested_value'])
#                         if st.button("保存修改"):
#                             save_result, save_error = fetch_data_from_backend(
#                                 f'/save_correction/{match["item_id"]}/{corrected_value}',
#                                 method='POST')
#                             if save_error:
#                                 st.error(save_error)
#                             else:
#                                 st.success(save_result)

#             # 保存匹配完成的历史修理单
#             if st.button("保存历史修理单"):
#                 save_result, save_error = fetch_data_from_backend(
#                     '/save_history',
#                     method='POST',
#                     json={"data": edited_df.to_dict()})
#                 if save_error:
#                     st.error(save_error)
#                 else:
#                     st.success(save_result)
#     return
