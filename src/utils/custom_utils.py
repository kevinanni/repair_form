import os


def get_absolute_path(relative_path):
    """
    将相对路径转换为绝对路径。
    
    Args:
        relative_path (str): 相对于项目根目录的路径。
    
    Returns:
        str: 绝对路径。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute_path = os.path.join(project_root, relative_path)
    return absolute_path


class StreamlitMetrics:

    def __init__(self):
        self.loss_history = []
        self.chart = None  # Initialize with appropriate chart object
        self.iteration_text = None  # Initialize with appropriate text object


streamlit_metrics = StreamlitMetrics()


# 回调函数：用于刷新训练图表
def update_streamlit(loss_value, iteration):
    streamlit_metrics.loss_history.append(loss_value)
    if streamlit_metrics.chart is not None:
        streamlit_metrics.chart.add_rows([loss_value])
    streamlit_metrics.iteration_text.text(
        f"Iteration: {iteration}, Loss: {loss_value:.6f}")
