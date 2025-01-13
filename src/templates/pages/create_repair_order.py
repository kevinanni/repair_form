import streamlit as st
from src.business_logic.repair_items import call_create_repair_order

def render_create_repair_order():
    # 输入船舶编号
    ship_id = st.text_input("请输入船舶编号", value="793051")

    if ship_id:
        # 调用call_create_repair_order获取修理单json
        repair_order_json = call_create_repair_order(ship_id)
        
        # if repair_order_json:
        #     # 展示json
        #     st.json(repair_order_json)
        # else:
        #     st.error("未能生成修理单，请检查船舶编号是否正确")
        st.json(repair_order_json)


