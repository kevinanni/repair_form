import os
import subprocess
import tabula

# 设置环境变量，禁用 jpype
os.environ['TABULA_JPYPE_DISABLED'] = 'true'


def perform_ocr(input_pdf_path, output_pdf_path):
    # 使用 OCRmyPDF 对 PDF 文件进行 OCR 处理
    subprocess.run([
        'ocrmypdf', '--deskew', '--rotate-pages', '--language', 'chi_sim',
        input_pdf_path, output_pdf_path
    ])


def read_pdf_tables(pdf_path):
    # 使用 Tabula 读取 PDF 文件中的表格数据
    dfs = tabula.read_pdf(pdf_path, pages='all')
    return dfs


def main():
    # 输入和输出 PDF 文件路径
    input_pdf_path = 'input.pdf'
    output_pdf_path = 'output.pdf'

    # 对 PDF 文件进行 OCR 处理
    perform_ocr(input_pdf_path, output_pdf_path)

    # 读取处理后的 PDF 文件中的表格数据
    dfs = read_pdf_tables(output_pdf_path)

    # 输出读取的所有表格
    for i, df in enumerate(dfs):
        print(f"Table {i+1}:")
        print(df)
        print("\n---\n")


if __name__ == "__main__":
    main()
