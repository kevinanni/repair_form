import streamlit as st
from src.utils.item_excel_parser import ItemExcelParser
from src.business_logic.repair_items import insert_repair_order
import os



def render_test_form():
    if st.button('开始语义匹配示例'):
        text_similarity()

def text_similarity():
    from src.models.modelscope import similarity_pipeline


    result = similarity_pipeline(input=('小孩咳嗽感冒', '小孩感冒过后久咳嗽该吃什么药育儿问答宝宝树'))
    st.text(result)

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
