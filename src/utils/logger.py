import logging
import sys

# 创建一个 logger 对象
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 设置日志级别为 DEBUG

# 创建一个 handler，用于写入日志文件
file_handler = logging.FileHandler('application.log')
file_handler.setLevel(logging.DEBUG)  # 设置 handler 的日志级别

# 定义 handler 的输出格式
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加 handler 到 logger
logger.addHandler(file_handler)


# 自定义类将标准输出和标准错误重定向到 logger
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


# 将 stdout 和 stderr 重定向到 logger
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)


# 示例代码
def main():
    print("This is a print statement.")
    logger.info("This is an info message.")
    logger.error("This is an error message.")
    logger.debug("This is a debug message.")


if __name__ == "__main__":
    main()
