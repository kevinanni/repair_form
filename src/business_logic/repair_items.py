from src.models.similarity_model import SimilarityModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import Numeric  # Use Numeric instead of Decimal
from src.business_logic.data_handling import session

Base = declarative_base()


class RepairBaseItem(Base):
    __tablename__ = 'repair_base_item'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键')
    item_name = Column(String(255), nullable=False, comment='修理名称')
    item_order = Column(Integer, default=None, comment='序号')
    item_require = Column(Text, comment='修理要求')
    category_id = Column(Integer, nullable=False, comment='维修类别')
    unit = Column(String(50), nullable=False, comment='单位')
    max_price = Column(Numeric(10, 2), default=None, comment='最高限价')
    remark = Column(Text, comment='备注')
    audit_state = Column(Integer, nullable=False, default=0, comment='审核状态')
    delete_state = Column(Integer, nullable=False, default=0, comment='删除状态')
    update_time = Column(DateTime,
                         server_default=func.now(),
                         onupdate=func.now(),
                         comment='最后更新时间')
    update_user = Column(String(100), default=None, comment='最后更新人员')


def init_similarity():
    # 从数据库中获取所有修理基础项目
    items = session.query(RepairBaseItem).all()
    # 使用id作为序号，item_name作为文本初始化SimilarityModel
    corpus = [item.item_name for item in items]
    similarity_model = SimilarityModel(corpus)
    return similarity_model


def get_matching_items(input_sentence, top_n=1):
    """
    根据输入句子和数量返回匹配的修理基础项目。

    Args:
        input_sentence (str): 输入的句子。
        top_n (int): 返回的匹配项目数量。

    Returns:
        list: 匹配的修理基础项目及其相似度。
    """
    similarity_model = init_similarity()  # 初始化相似度模型
    matches = similarity_model.find_most_similar(input_sentence,
                                                 top_n)  # 获取匹配结果
    return matches
