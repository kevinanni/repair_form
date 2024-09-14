import streamlit as st
from src.business_logic.repair_items import get_matching_items


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
        st.dataframe(df)  # 显示DataFrame

        # 点击匹配，显示表格中每条记录匹配的语句
        if st.button("匹配修理单项目"):
            match_results = []
            for index, row in df.iterrows():
                input_sentence = row['维修内容']
                matches = get_matching_items(input_sentence, top_n=3)  # 获取匹配结果
                match_results.append((input_sentence, matches))

            # 逐个显示匹配结果
            for input_sentence, matches in match_results:
                st.write(f"输入: {input_sentence}")
                for idx, match in enumerate(matches, start=1):
                    st.write(f"匹配项{idx}: {match[0]} (相似度: {match[1]:.2f})")
                st.markdown("---")  # 添加分割线

            # # 表格显示匹配结果
            # results_df = pd.DataFrame(match_results, columns=["输入", "匹配项"])
            # results_df['匹配项'] = results_df['匹配项'].apply(lambda x: ', '.join(
            #     [f"{match[0]} (相似度: {match[1]:.2f})" for match in x]))
            # st.write("匹配结果:")
            # st.dataframe(results_df, use_container_width=True)

            # selected_row = st.dataframe(results_df, use_container_width=True)
            # if selected_row is not None:
            #     selected_row_index = selected_row.index[
            #         0] if selected_row is not None else None
            #     if selected_row_index is not None:
            #         st.write(f"您选择的行: {results_df.iloc[selected_row_index]}")


# def history_repair_match_page():
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
