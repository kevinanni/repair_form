import os
import re
import subprocess
import tabula
import pandas as pd
import pdb

# 设置环境变量，禁用 jpype
os.environ['TABULA_JPYPE_DISABLED'] = 'true'


def perform_ocr(input_pdf_path, output_pdf_path):
    from PIL import Image

    # Check the file extension of the input path
    if input_pdf_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Convert the image to PDF
        image = Image.open(input_pdf_path)
        input_pdf_path = input_pdf_path.rsplit('.', 1)[0] + '.pdf'
        image.save(input_pdf_path, 'PDF', resolution=100.0)
    # 使用 OCRmyPDF 对 PDF 文件进行 OCR 处理
    subprocess.run([
        'ocrmypdf', '--deskew', '--rotate-pages', '--language', 'chi_sim',
        '--force-ocr', input_pdf_path, output_pdf_path
    ])


def read_pdf_tables(pdf_path):
    # 使用 Tabula 读取 PDF 文件中的表格数据
    dfs = tabula.read_pdf(pdf_path, pages='all')
    return dfs


def process_dataframe(df):
    """
    处理DataFrame，例如去除空格，改列名等等
    """
    # 匹配所有空白字符
    pattern = r'\u0020+|\u3000+'
    df.columns = df.columns.str.replace(pattern, '', regex=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)))

    # 找到平均字数最多的列
    average_word_counts = df.apply(lambda col: col.str.len().mean()
                                   if col.dtype == 'object' else 0)
    # pdb.set_trace()
    max_average_col = average_word_counts.idxmax()
    max_average_value = average_word_counts.max()
    print(f"平均字数最多的列: {max_average_col}, 平均字数: {max_average_value:.2f}")

    # import pdb
    # pdb.set_trace()  # 插入断点
    # 如果所有的列都很短，那么不处理了
    if (max_average_value) > 3:

        # # 找到项目名称行，删除前面多余的行，这段代码不一定合适，先注释
        # name_list = ['项目名称', '项目内容']  # 示例名称列表
        # for index, value in df[max_average_col].items():
        #     # 打印字符
        #     print('value:', value)
        #     print([hex(ord(c)) for c in value])

        #     if isinstance(value, str) and any(name in value
        #                                       for name in name_list):
        #         # matching_cell = (index, max_average_col, value)
        #         print(f"找到的单元格: 行: {index}, 列: {max_average_col}, 值: {value}")
        #         # 删除df中该行之前的所有行，并重置索引
        #         df = df.loc[index + 1:].reset_index(drop=True)
        #         break  # 找到一个cell后立即退出循环

        # pdb.set_trace()
        # 将该列内容前面的数字去掉
        df[max_average_col] = df[max_average_col].str.replace(r'^\d+\s*',
                                                              '',
                                                              regex=True)
        # 将该列统一命名
        df.rename(columns={max_average_col: "项目名称"}, inplace=True)

        return df
    else:
        # 说明没找到合适的表格，直接返回空值
        return None


def convert_file(file_path):
    """
    输入pdf或者jpg文件路径，进行识别，输出转换状态和excel文件路径
    """
    temp_path = os.path.join(os.getcwd(), 'temp.pdf')
    conversion_status = "转换成功"  # 默认转换状态
    excel_file_path = ""
    print('start...')
    try:
        # 对 PDF 文件进行 OCR 处理
        print('perform_ocr...')
        perform_ocr(file_path, temp_path)

        # 读取处理后的 PDF 文件中的表格数据
        print('read_pdf_tables...')
        dfs = read_pdf_tables(temp_path)

        # 写入新的excel文件
        print('write to excel...')
        if not dfs:
            conversion_status = "未发现表格"
        else:
            # 识别成功，返回excel文件路径
            excel_file_path = file_path.rsplit('.', 1)[0] + '.xlsx'

            # 将读取的所有表格写入Excel文件
            with pd.ExcelWriter(excel_file_path) as writer:
                for i, df_raw in enumerate(dfs):
                    df_done = process_dataframe(df_raw)
                    # 非空表示找到维修清单，才执行下面的
                    if df_done is not None and not df_done.empty:
                        df_done.to_excel(writer,
                                         sheet_name=f'Table_{i+1}',
                                         index=False)

    except Exception as e:
        conversion_status = f"转换失败: {str(e)}"  # 捕获异常并更新状态

    return conversion_status, excel_file_path


def process_files(repair_excel, root_directory, iter_times=10):
    """
    
    在输出文件中写入转换状态和excel路径
    """
    repair_excel_file = os.path.join(root_directory, repair_excel)
    # 读取repair_excel文件
    df_repair = pd.read_excel(repair_excel_file)

    # 逐行处理
    count = 0  # 计数器
    for index, row in df_repair.iterrows():
        if pd.notna(row['F_TRANS']):
            continue
        file_path = os.path.join(root_directory, row['F_FILEPATH'])
        print('index:', index)
        print('file_path:', file_path)
        conversion_status, excel_file_path = convert_file(file_path)
        # 写入索引excel，直接使用loc来设置行数据
        df_repair.loc[index, 'F_TRANS'] = conversion_status
        # 只写入root_directory之后的路径，且用/分隔
        if excel_file_path:
            df_repair.loc[index, 'F_EXCEL'] = os.path.relpath(
                excel_file_path, root_directory).replace('\\', '/')

        count += 1  # 每次成功调用convert_file时计数
        if count >= iter_times:  # 检查是否超过iter_times
            print("Reached the maximum number of iterations, exiting...")
            break

    # 写入repair_excel文件
    df_repair.to_excel(repair_excel_file, index=False)


def main(test_type, iter_times):
    """
    main函数，调用函数2进行转换
    """
    # original_filename = 'example.pdf'  # 示例原始文件名
    # attachment_directory = './attachments'  # 附件目录
    # output_filename = 'conversion_status.txt'  # 输出文件名

    # process_files(original_filename, attachment_directory, output_filename)

    if test_type == 'single':
        file_path = 'E:\\35_code\\20_personal\\repair_form\\data\\offline\\AnnexesFile/d9dca542-372b-43d2-9fe8-5f621ecd9c37/20210331/5b7bd538-10e3-4a03-b15d-945ff0b28f66.pdf'

        status, _ = convert_file(file_path)
        print(status)
    elif test_type == 'multi':
        repair_excel = 'dsf.xlsx'
        root_directory = os.getcwd()
        process_files(repair_excel, root_directory, iter_times)
    elif test_type == 'char':
        text = '2 右 蝎旋 桨、 校 正 只'
        # 匹配所有空白字符
        pattern = r'[\u0020\u3000]+'

        # 替换为空字符串
        cleaned_text = re.sub(pattern, '', text)
        print(cleaned_text)
    else:
        print("Invalid test_type. Please use 'single' or 'multi'.")


if __name__ == "__main__":
    import time
    import sys

    start_time = time.time()  # 记录开始时间
    print(f"===========开始时间: {time.ctime(start_time)}")

    # 获取命令行参数，默认为'multi'
    test_type = sys.argv[1] if len(sys.argv) > 1 else "multi"
    iter_times = int(
        sys.argv[2]) if test_type == "multi" and len(sys.argv) > 2 else 20
    main(test_type, iter_times)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"===========结束时间: {time.ctime(end_time)}")
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"运行时间: {minutes}分钟 {seconds:.2f}秒")
    # input_pdf_path = 'E:\\35_code\\20_personal\\repair_form\\data\\offline\\AnnexesFile\\5e85f2e3-3e6e-438e-a7ca-fb0fdab49729\\20210825\\f9d9a69b-a9f8-4351-bc6a-16bc7c07cf06.pdf'
    # output_pdf_path = 'E:\\35_code\\20_personal\\repair_form\\data\\offline\\temp.pdf'
    # perform_ocr(input_pdf_path, output_pdf_path)
