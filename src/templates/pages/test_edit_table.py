import streamlit as st
import pandas as pd


def render_test_form():
    # 示例 DataFrame
    data = {'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']}
    df = pd.DataFrame(data)

    # 初始化 session state
    if 'edit_index' not in st.session_state:
        st.session_state.edit_index = None

    # 显示表格
    st.write("原始表格:")
    st.dataframe(df)

    # 监听表格的双击事件
    st.write("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var tables = document.querySelectorAll('.stDataFrame table');
        tables.forEach(function(table) {
            table.addEventListener('dblclick', function(event) {
                var row = event.target.closest('tr').rowIndex;
                var col = event.target.closest('td').cellIndex;
                window.parent.postMessage({source: 'child', rowIndex: row, columnIndex: col}, location.origin);
            });
        });
    });
    </script>
    """,
             unsafe_allow_html=True)

    # 监听消息事件
    if st.session_state.edit_index is not None:
        if st.button("编辑"):
            # 获取双击的单元格信息
            rowIndex = st.session_state.edit_index['rowIndex']
            columnIndex = st.session_state.edit_index['columnIndex']

            # 获取当前单元格的值
            current_value = df.iat[rowIndex, columnIndex]

            # 编辑单元格的值
            new_value = st.text_input(f"编辑单元格 ({rowIndex}, {columnIndex})",
                                      value=current_value)

            # 更新表格
            if st.button("保存"):
                df.iat[rowIndex, columnIndex] = new_value
                st.session_state.edit_index = None
                st.experimental_rerun()

    # 处理消息事件
    if st.button("双击表格"):
        st.session_state.edit_index = {
            'rowIndex': 0,
            'columnIndex': 0
        }  # 示例双击位置
    else:
        st.session_state.edit_index = None

    # 显示更新后的表格
    st.write("更新后的表格:")
    st.dataframe(df)
