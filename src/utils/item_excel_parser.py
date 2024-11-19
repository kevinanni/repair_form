import pandas as pd
import openpyxl
import pdb

# python -m src.utils.item_excel_parser


# 用于读取excel文件(gbk编码)并解析，获取历史修理单的信息
class ItemExcelParser:

    def __init__(self):
        self.column_mappings = {
            'id': ['序号'],
            'item_name': ['维修项目', '维修内容', '项目内容', '项目名称', '修理项目'],
            'item_content': ['规格型号'],
            'unit': ['单位'],
            'count': ['数量'],
            'price': ['单价'],
            'remark': ['备注'],
        }
        self.category_mapping = [
            {
                'cid': 27,
                'cname': ['坞修(水下)', '特检项目', '中检项目', '坞修工程']
            },
            {
                'cid': 28,
                'cname': ['船体(水上)', '船体工程', '船体工程（含附属设施）', '船体项目']
            },
            {
                'cid': 29,
                'cname': ['轮机', '轮机设备', '轮机工程', '轮机项目']
            },
            {
                'cid': 30,
                'cname': ['电气', '电气设备', '电气工程', '电气项目']
            },
            {
                'cid': 31,
                'cname': ['其他维修', '其他', '其它', '专项', '无人船吊机改造']
            },
        ]
        self.title_positions = None

    # 数据字典，用于匹配excel中的中文标题和最终df的英文标题
    def get_column_mappings(self):
        return self.column_mappings

    # 读入excel文件模板，用于定位需要解析数据的标题和内容
    def parse_template(self, template_path):

        template_wb = openpyxl.load_workbook(template_path)
        template_ws = template_wb.active

        title_positions = {}
        for row in template_ws.iter_rows():
            for cell in row:
                # 遍历每个英文key对应的中文标题列表
                for eng_key, cn_titles in self.column_mappings.items():
                    if cell.value in cn_titles:
                        # 多中文标题支持
                        if eng_key not in title_positions:
                            title_positions[eng_key] = []
                        # 模板实际给出的是数据开始行
                        title_positions[eng_key].append(cell.coordinate)

        self.title_positions = title_positions
        return title_positions

    # 读取需要解析的excel文件，并遍历sheet，调用parse_sheet，输出df列表
    def parse_excel_file(self, data_file_path):
        wb = openpyxl.load_workbook(data_file_path)
        df_list = []

        # 从文件路径中提取文件名
        filename = data_file_path.replace('\\',
                                          '/').split('/')[-1].split('.')[0]
        # 用'-'分隔文件名，获取部门和年份
        parts = filename.split('-')
        department = parts[0] if len(parts) > 0 else ''
        year = parts[1] if len(parts) > 1 else ''

        # 遍历每个工作表
        for sheet_name in wb.sheetnames:
            # 解析工作表内容
            df = self.parse_sheet(wb[sheet_name], self.title_positions)
            # 添加类别解析
            df = self.process_category(df)
            # 将部门、年份、船名和数据一起添加到结果列表
            df_list.append({
                'department': department,
                'year': year,
                'ship': sheet_name,
                'items': df
            })

        return df_list

    # 根据模板，对sheet进行解析，定位到标题，将对应列中后续的行读入，最终输出一个df
    def parse_sheet(self, worksheet, title_positions):
        # 用title_positions中的列名初始化空DataFrame
        df = pd.DataFrame(columns=title_positions.keys())

        # 从任意列的第一个坐标获取起始行
        start_row = worksheet[next(iter(title_positions.values()))[0]].row
        current_row = start_row

        while True:
            # 检查当前行是否有任何单元格包含值
            row_data = {}
            has_value = False

            # 解析当前行的每一列
            for column_name, cell_coords in title_positions.items():
                cell_values = []
                for coord in cell_coords:
                    col_letter = coord[0]  # 从坐标获取列字母
                    cell = worksheet[f"{col_letter}{current_row}"]
                    # 获取合并单元格的值
                    merged_cell_ranges = [
                        merged_range
                        for merged_range in worksheet.merged_cells.ranges
                        if cell.coordinate in merged_range
                    ]
                    if merged_cell_ranges:
                        # 如果是合并单元格，获取合并区域左上角单元格的值
                        merged_range = merged_cell_ranges[0]
                        cell_value = worksheet[
                            merged_range.start_cell.coordinate].value
                    else:
                        cell_value = cell.value

                    if cell_value is not None:
                        has_value = True
                        cell_values.append(str(cell_value))
                    else:
                        cell_values.append("")

                # 如果存在多个单元格值，用破折号连接
                row_data[column_name] = '-'.join(filter(None,
                                                        cell_values)) or None

            if not has_value:
                break

            # 将行数据添加到DataFrame
            df.loc[len(df)] = row_data
            current_row += 1
        return df

    # 处理类别相关的内容
    def process_category(self, df):
        # 如果df中没有category_id列，增加该列
        if 'category_id' not in df.columns:
            df['category_id'] = None

        current_category_id = None
        rows_to_drop = []

        # pdb.set_trace()
        # 循环每行，找到id不是普通数字1~9组成的
        for index, row in df.iterrows():
            # print('current_category_id:', current_category_id)

            item_id = str(row['id'])
            if not item_id.isdigit():
                # 只要不是数字序号，都不插入
                rows_to_drop.append(index)

                if row['item_name'] is None:
                    continue

                # 将item_name用-分开，取开始的那部分
                item_name_parts = row['item_name'].split('-')
                item_name = item_name_parts[0]

                # # 如果id不是数字，item_name中又有小计、合计等等，就剔除
                # if '小计' in item_name or '合计' in item_name:
                #     rows_to_drop.append(index)
                #     continue

                # 如果取出的部分在self.category_mapping的cname中，表示这是类别行
                for mapping in self.category_mapping:
                    if item_name in mapping['cname']:
                        # 取出self.category_mapping对应的cid
                        current_category_id = mapping['cid']
                        # 标记需要删除的类别行
                        # rows_to_drop.append(index)
                        break
            else:
                # 为普通行设置category_id
                if current_category_id is not None:
                    df.at[index, 'category_id'] = current_category_id

        # 删除类别行
        df.drop(rows_to_drop, inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df


def main():
    # 设置日志记录
    import logging
    import os
    logger = logging.getLogger()

    from src.business_logic.repair_items import insert_repair_order

    # try:

    # 示例路径，实际使用时需要替换为真实路径
    template_path = os.path.join(os.getcwd(), 'data', 'raw', '芜湖航道处-模板.xlsx')
    data_file_path = os.path.join(os.getcwd(), 'data', 'raw',
                                  '芜湖航道处-2023.xlsx')

    # 创建解析器实例并解析Excel文件
    parser = ItemExcelParser()
    parser.parse_template(template_path)
    df_list = parser.parse_excel_file(data_file_path)

    print(f'处理了{len(df_list)}个sheet')
    print(df_list[0])

    for df in df_list:
        insert_repair_order(df)

    # 打印每个sheet的解析结果
    # for i, df in enumerate(df_list):
    #     print(f"Sheet {i+1} 解析结果:")
    #     print(df)
    # logger.info(f"Sheet {i+1} 解析结果:")
    # logger.info(df)

    # except Exception as e:
    #     logger.error(f"解析过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()
