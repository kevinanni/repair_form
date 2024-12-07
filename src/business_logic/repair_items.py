from src.models.similarity_model import SimilarityModel
from sqlalchemy import ForeignKey, create_engine, Column, Integer, String, Text, DateTime, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import Numeric  # Use Numeric instead of Decimal
from sqlalchemy.orm import relationship
from src.business_logic.data_handling import get_session

from src.models.repair_item_matcher import RepairItemMatcher

import pandas as pd

import pdb

Base = declarative_base()


class RepairBaseItem(Base):
    __tablename__ = 'repair_base_item'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键')
    item_name = Column(String(255), nullable=False, comment='修理名称')
    item_order = Column(Integer, default=None, comment='序号')
    item_content = Column(String(255), default=None, comment='修理要求')
    category_id = Column(Integer, default=None, comment='维修类别')
    unit = Column(String(50), default=None, comment='单位')
    count = Column(Numeric(10, 2), default=None, comment='数量')
    price = Column(Numeric(10, 2), default=None, comment='价格')
    max_price = Column(Numeric(10, 2), default=None, comment='最高限价')
    remark = Column(Text, comment='备注')
    audit_state = Column(Integer, default=0, comment='审核状态')
    delete_state = Column(Integer, default=0, comment='删除状态')
    update_time = Column(DateTime,
                         server_default=func.now(),
                         onupdate=func.now(),
                         comment='最后更新时间')
    update_user = Column(String(100), default=None, comment='最后更新人员')
    @classmethod
    def get_item(cls, item_id):
        """查询单个基础修理项目"""
        with get_session() as session:
            return session.query(cls).filter_by(id=item_id, delete_state=0).first()

    @classmethod 
    def get_items(cls, **filters):
        """查询基础修理项目列表"""
        with get_session() as session:
            query = session.query(cls).filter_by(delete_state=0)
            for key, value in filters.items():
                if hasattr(cls, key):
                    query = query.filter(getattr(cls, key) == value)
            return query.all()

    @classmethod
    def add_item(cls, **kwargs):
        """添加基础修理项目"""
        with get_session() as session:
            item = cls(**kwargs)
            session.add(item)
            session.commit()
            return item

    @classmethod
    def update_item(cls, item_id, **kwargs):
        """更新基础修理项目"""
        with get_session() as session:
            item = session.query(cls).filter_by(id=item_id).first()
            if item:
                for key, value in kwargs.items():
                    setattr(item, key, value)
                session.commit()
            return item

    @classmethod
    def delete_item(cls, item_id):
        """删除基础修理项目(软删除)"""
        with get_session() as session:
            item = session.query(cls).filter_by(id=item_id).first()
            if item:
                item.delete_state = 1
                session.commit()
            return item


class RepairItem(Base):
    __tablename__ = 'repair_item'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键')
    item_name = Column(String(255), nullable=False, comment='修理名称')
    std_id = Column(Integer, default=None, comment='标准修理单id')
    is_std = Column(Integer, default=None, comment='是否匹配标准修理单')
    std_name = Column(String(255), default=None, comment='标准修理单名称')
    item_order = Column(Integer, default=None, comment='序号')
    item_content = Column(String(255), default=None, comment='规格型号')
    category_id = Column(Integer, default=None, comment='维修类别')
    unit = Column(String(50), default=None, comment='单位')
    count = Column(Numeric(10, 2), default=None, comment='数量')
    price = Column(Numeric(10, 2), default=None, comment='价格')
    max_price = Column(Numeric(10, 2), default=None, comment='最高限价')
    remark = Column(Text, comment='备注')
    order_id = Column(Integer, ForeignKey('repair_order.id'))
    audit_state = Column(Integer, default=0, comment='审核状态')
    delete_state = Column(Integer, default=0, comment='删除状态')
    update_time = Column(DateTime,
                         server_default=func.now(),
                         onupdate=func.now(),
                         comment='最后更新时间')
    update_user = Column(String(100), default=None, comment='最后更新人员')

    @classmethod
    def add_item(cls, session, **kwargs):
        """添加维修项目"""
        item = cls(**kwargs)
        session.add(item)
        session.commit()
        return item

    @classmethod
    def update_item(cls, session, item_id, **kwargs):
        """更新维修项目"""
        item = session.query(cls).filter_by(id=item_id).first()
        if item:
            for key, value in kwargs.items():
                setattr(item, key, value)
            session.commit()
        return item

    @classmethod
    def delete_item(cls, session, item_id):
        """删除维修项目(软删除)"""
        item = session.query(cls).filter_by(id=item_id).first()
        if item:
            item.delete_state = 1
            session.commit()
        return item

    @classmethod
    def get_item(cls, item_id):
        """查询单个维修项目"""
        with get_session() as session:
            return session.query(cls).filter_by(id=item_id, delete_state=0).first()

    @classmethod
    def get_items_by_order(cls, session, order_id):
        """查询订单下的所有维修项目"""
        return session.query(cls).filter_by(order_id=order_id,
                                            delete_state=0).all()


class RepairOrder(Base):
    __tablename__ = 'repair_order'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键')
    ship = Column(String(50), nullable=False, comment='船名')
    department = Column(String(50), nullable=False, comment='部门')
    year = Column(String(4), nullable=False, comment='年份')
    audit_state = Column(Integer, nullable=False, default=0, comment='审核状态')
    delete_state = Column(Integer, nullable=False, default=0, comment='删除状态')
    update_time = Column(DateTime,
                         server_default=func.now(),
                         onupdate=func.now(),
                         comment='最后更新时间')
    update_user = Column(String(100), default=None, comment='最后更新人员')

    # 定义与RepairItem的一对多关系
    items = relationship("RepairItem", backref="order")

    @classmethod
    def add_order(cls, session, **kwargs):
        """添加维修订单"""
        order = cls(**kwargs)
        session.add(order)
        session.commit()
        return order

    @classmethod
    def update_order(cls, session, order_id, **kwargs):
        """更新维修订单"""
        order = session.query(cls).filter_by(id=order_id).first()
        if order:
            for key, value in kwargs.items():
                setattr(order, key, value)
            session.commit()
        return order

    @classmethod
    def delete_order(cls, session, order_id):
        """删除维修订单(软删除)"""
        order = session.query(cls).filter_by(id=order_id).first()
        if order:
            order.delete_state = 1
            # 同时删除订单下的所有维修项目
            for item in order.items:
                item.delete_state = 1
            session.commit()
        return order

    @classmethod
    def get_order(cls, session, order_id):
        """查询单个维修订单"""
        return session.query(cls).filter_by(id=order_id,
                                            delete_state=0).first()

    @classmethod
    def get_orders(cls, session, **filters):
        """查询维修订单列表"""
        query = session.query(cls).filter_by(delete_state=0)
        for key, value in filters.items():
            if hasattr(cls, key):
                query = query.filter(getattr(cls, key) == value)
        return query.all()


def init_similarity():
    with get_session() as session:
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


#########################
# 这两个函数用于RepairItemMatcher
def init_items_match():
    with get_session() as session:
        # 从视图中获取标准修理项目

        sql_query = text("SELECT std_id, std_combine FROM v_repair_std")
        items = session.execute(sql_query).fetchall()

        # 使用std_id和std_combine初始化语料库
        corpus = [(item.std_id, item.std_combine) for item in items]
        model = RepairItemMatcher()  # 使用RepairItemMatcher进行初始化
        model.load_standard_items(corpus)  # 加载标准维修项目
        return model


class RepairCorpus(Base):
    __tablename__ = 'repair_corpus'

    id = Column(Integer, primary_key=True, autoincrement=True, comment='主键')
    item_name = Column(String(255), comment='项目名称')
    item_combine = Column(String(255), comment='项目名称')
    is_std = Column(Integer, comment='是否标准项目')
    std_id = Column(Integer, comment='标准项目id')
    std_name = Column(String(255), comment='标准项目名称')
    std_combine = Column(String(255), comment='标准项目名称及规格')
    remark = Column(Text, comment='备注')
    audit_state = Column(Integer, default=0, comment='审核状态')
    delete_state = Column(Integer, default=0, comment='删除状态')
    update_time = Column(DateTime,
                         server_default=func.now(),
                         onupdate=func.now(),
                         comment='最后更新时间')
    update_user = Column(String(100), comment='最后更新人员')

    @classmethod
    def add_corpus(cls, session, **kwargs):
        """添加语料"""
        corpus = cls(**kwargs)
        session.add(corpus)
        session.commit()
        return corpus

    @classmethod
    def update_corpus(cls, session, corpus_id, **kwargs):
        """更新语料"""
        corpus = session.query(cls).filter_by(id=corpus_id).first()
        if corpus:
            for key, value in kwargs.items():
                setattr(corpus, key, value)
            session.commit()
        return corpus

    @classmethod
    def delete_corpus(cls, session, corpus_id):
        """删除语料(软删除)"""
        corpus = session.query(cls).filter_by(id=corpus_id).first()
        if corpus:
            corpus.delete_state = 1
            session.commit()
        return corpus

    @classmethod
    def get_corpus(cls, session, corpus_id):
        """查询单个语料"""
        return session.query(cls).filter_by(id=corpus_id,
                                            delete_state=0).first()

    @classmethod
    def get_corpus_list(cls, session, **filters):
        """查询语料列表"""
        query = session.query(cls).filter_by(delete_state=0)
        for key, value in filters.items():
            if hasattr(cls, key):
                query = query.filter(getattr(cls, key) == value)
        return query.all()


def get_corpus_for_match():
    with get_session() as session:
        # 查询所需字段
        query = session.query(
            RepairCorpus.id, RepairCorpus.item_name, RepairCorpus.item_combine,
            RepairCorpus.is_std, RepairCorpus.std_id, RepairCorpus.std_name,
            RepairCorpus.std_combine).filter(RepairCorpus.delete_state == 0)

        # 执行查询并转换为DataFrame
        result = pd.DataFrame([{
            'id': row.id,
            'item_name': row.item_name,
            'item_combine': row.item_combine,
            'is_std': row.is_std,
            'std_id': row.std_id,
            'std_name': row.std_name,
            'std_combine': row.std_combine
        } for row in query.all()])

        return result


def update_corpus_from_df(df):
    """
    将DataFrame中的语料数据更新回数据库

    Args:
        df (pd.DataFrame): 包含要更新的语料数据的DataFrame，
                          需要包含id, item_name, item_combine, is_std, std_name, std_combine字段

    Returns:
        bool: 更新是否成功
    """
    with get_session() as session:
        try:
            for _, row in df.iterrows():
                corpus = session.query(RepairCorpus).filter_by(
                    id=row['id']).first()
                if corpus:
                    corpus.item_name = row['item_name']
                    corpus.item_combine = row['item_combine']
                    corpus.is_std = row['is_std'] if pd.notna(row['is_std']) else None
                    corpus.std_id = row['std_id']
                    corpus.std_name = row['std_name']
                    corpus.std_combine = row['std_combine']
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"更新语料时发生错误: {str(e)}")
            return False


def get_items_match(input_sentences):
    """
    根据输入句子和数量返回匹配的修理基础项目。

    Args:
        input_sentence (str): 输入的句子。

    Returns:
        list: 匹配的修理基础项目及其相似度。
    """
    model = init_items_match()  # 初始化RepairItemMatcher
    # matches = model.find_matches([input_sentence])  # 获取匹配结果
    result = model.find_matches(input_sentences)
    return result
    # model = RepairItemMatcher()  # 初始化RepairItemMatcher
    # result = model.find_matches(input_sentences)
    # return result


def get_all_standard_items():
    """
    获取所有标准修理项目的列表。

    Returns:
        list: 包含所有标准修理项目的列表，每个项目为一个字典，包含id和item_combine。
    """
    with get_session() as session:
        sql_query = text(
            "SELECT std_id as id, std_combine as item_combine FROM v_repair_std"
        )
        items = session.execute(sql_query).fetchall()
        return [{
            'id': item.id,
            'item_combine': item.item_combine
        } for item in items]

def get_standard_item(item_id):
    """
    根据item_id获取标准修理项目。

    Args:
        item_id: 标准修理项目ID

    Returns:
        dict: 包含id和item_combine的字典，如果未找到则返回None
    """
    with get_session() as session:
        sql_query = text(
            "SELECT * FROM v_repair_std WHERE std_id = :item_id"
        )
        item = session.execute(sql_query, {'item_id': item_id}).first()
        return item
        # if item:
        #     return {
        #         'id': item.id,
        #         'item_combine': item.item_combine,
        #         'item_name': item.item_name
        #     }
        # return None


def insert_repair_order(order_data):
    with get_session() as session:
        try:
            # 创建维修订单
            repair_order = RepairOrder(ship=order_data['ship'],
                                       department=order_data['department'],
                                       year=order_data['year'])
            session.add(repair_order)
            # 刷新，否则没有id
            session.flush()

            # 添加维修项目
            for i in range(len(order_data['items'])):
                item_data = order_data['items'].iloc[i]
                # # 获取RepairItem模型的所有必填字段
                # required_fields = [
                #     column.name for column in RepairItem.__table__.columns
                #     if not column.nullable and not column.primary_key
                # ]

                # # 检查必填字段是否存在且非空
                # for field in required_fields:
                #     if field not in item_data or not item_data[field]:
                #         raise ValueError(f"维修项目缺少必填字段: {field}")

                # pdb.set_trace()

                repair_item = RepairItem(
                    order_id=repair_order.id,
                    item_name=item_data.get('item_name'),
                    std_id=item_data.get('std_id'),
                    is_std=item_data.get('is_std'),
                    std_name=item_data.get('std_name'),
                    item_order=item_data.get('item_order'),
                    item_content=item_data.get('item_content'),
                    category_id=item_data.get('category_id'),
                    unit=item_data.get('unit'),
                    count=item_data.get('count'),
                    price=item_data.get('price'),
                    max_price=item_data.get('max_price'),
                    remark=item_data.get('remark'))

                session.add(repair_item)
                # session.flush()

            session.commit()
            print(f"成功创建维修订单: {repair_order.id}")
        except Exception as e:
            session.rollback()
            raise e


def main():
    # 示例数据
    order_data = {
        'department':
        '宜宾局',
        'year':
        '2023',
        'ship':
        '航道01106',
        'items': [{
            'id': '一',
            'item_name': '船体工程-船体工程',
            'count': None,
            'unit': None,
            'price': None,
            'item_content': None,
            'remark': None
        }, {
            'id': '1',
            'item_name': '船底板-船体外板测厚',
            'count': None,
            'unit': '工天',
            'price': None,
            'item_content': None,
            'remark': None
        }, {
            'id': '2',
            'item_name': '船尾-船名及船籍港标识焊接钢字',
            'count': None,
            'unit': '工天',
            'price': None,
            'item_content': '船名及船籍港标识钢字加工',
            'remark': None
        }]
    }
    insert_repair_order(order_data)


if __name__ == "__main__":
    main()
