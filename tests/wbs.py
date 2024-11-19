def parse_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    wbs = []
    current_level = 0
    parent_stack = []
    level_indent = 4  # 每一级的缩进量（假设为4个空格）

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue  # 跳过空行

        # 根据前导空格数确定层级
        indent = len(line) - len(line.lstrip(' '))
        level = indent // level_indent

        # 如果当前层级小于之前的最大层级，则清除超出部分的父级栈
        while len(parent_stack) > level:
            parent_stack.pop()

        # 生成完整的序号
        if level == 0:
            number = f"{len(wbs) + 1}."
        else:
            parent_number = '.'.join([item['number'] for item in parent_stack])
            number = f"{parent_number}.{len(parent_stack[-1]['children']) + 1}"

        # 添加新任务到WBS列表
        wbs.append({
            'number': number,
            'text': stripped_line,
            'level': level,
            'children': []
        })

        # 将当前任务添加到父任务的子任务列表中
        if level > 0:
            parent_stack[-1]['children'].append(wbs[-1])

        # 将当前任务添加到父级栈中
        if level > len(parent_stack) - 1:
            parent_stack.append(wbs[-1])

    return wbs


def print_wbs(wbs):

    def print_helper(task, depth=0):
        print(f"{'  ' * depth}{task['number']} {task['text']}")
        for child in task.get('children', []):
            print_helper(child, depth + 1)

    for task in wbs:
        print_helper(task)


if __name__ == "__main__":
    file_path = 'wbs.md'  # 替换为你的Markdown文件路径
    wbs = parse_markdown(file_path)
    print_wbs(wbs)
