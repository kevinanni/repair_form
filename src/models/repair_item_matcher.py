import threading
from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os


class RepairItemMatcher:

    def __init__(self, model_name='hfl--chinese-bert-wwm'):

        model_path = os.path.join(os.getcwd(), 'src', 'models', 'local',
                                  model_name)
        self.model = SentenceTransformer(model_path)
        self.std_embeddings = None
        self.std_items = None

    def load_standard_items(self, std_items):
        """
        加载标准维修项目并计算其嵌入向量。
        :param std_items: 标准维修项目的列表，格式为[(id, item), ...]。
        """
        # 提取id和item
        ids, items = zip(*std_items)
        self.std_embeddings = self.model.encode(items, convert_to_tensor=True)
        self.std_items = list(zip(ids, items))  # 保存id和item的组合

    def find_matches(self, input_sentences):
        """
        为每个输入句子找到最匹配的标准维修项目。
        :param input_sentences: 输入文本的列表。
        :return: 包含输入文本、匹配id和匹配文本的列表。
        """
        input_embeddings = self.model.encode(input_sentences,
                                             convert_to_tensor=True)
        results = []
        for idx, input_emb in enumerate(input_embeddings):
            similarity_scores = cosine_similarity(
                [input_emb.cpu().detach().numpy()],
                self.std_embeddings.cpu().detach().numpy())[0]
            match_index = np.argmax(similarity_scores)
            results.append(
                (input_sentences[idx], self.std_items[match_index][0],
                 self.std_items[match_index][1]))  # 返回输入文本、匹配id和匹配文本
        return results

    def prepare_training_data(self, matches, std_items):
        """
        准备训练数据。
        :param matches: 匹配结果的字典。
        :param std_items: 标准维修项目的列表。
        :return: 训练数据集。
        """
        train_examples = [
            InputExample(texts=[train_item, std_items[match_index]], label=1)
            for train_item, match_index in matches.items()  # 使用整数索引
        ]

        # 创建SentencesDataset对象
        train_dataset = SentencesDataset(examples=train_examples,
                                         model=self.model)

        return DataLoader(train_dataset, shuffle=True, batch_size=16)

    def fine_tune_model(self, train_dataloader, epochs=3, warmup_steps=50):
        """
        微调模型。
        :param train_dataloader: 训练数据加载器。
        :param epochs: 微调轮数。
        :param warmup_steps: 预热步数。
        """
        self.model.fit(train_objectives=[
            (train_dataloader, losses.CosineSimilarityLoss(model=self.model))
        ],
                       epochs=epochs,
                       warmup_steps=warmup_steps)

    def test_model(self, test_items):
        """
        测试模型。
        :param test_items: 测试维修项目的列表。
        :param std_items: 标准维修项目的列表。
        """
        test_embeddings = self.model.encode(test_items, convert_to_tensor=True)
        results = []  # 用于存储所有测试项和匹配项
        for test_emb in test_embeddings:
            similarity_scores = cosine_similarity(
                [test_emb.cpu().detach().numpy()],
                self.std_embeddings.cpu().detach().numpy())[0]
            match_index = np.argmax(similarity_scores)
            results.append(
                (test_items[test_embeddings.tolist().index(test_emb.tolist())],
                 self.std_items[match_index]))

        # 返回所有测试项和匹配项
        for test_item, match in results:
            print(f"Test item: {test_item}, Matches with: {match}")
        return results

    def save_model(self, output_path):
        """
        保存微调后的模型。
        :param output_path: 保存模型的路径。
        """
        self.model.save(output_path)

    def load_model(self, model_path):
        """
        加载已保存的模型。
        :param model_path: 模型保存的路径。
        """
        self.model = SentenceTransformer(model_path)

    # 单例实例
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """
        使用__new__方法实现单例模式。
        确保RepairItemMatcher只有一个实例。

        Returns:
            RepairItemMatcher: RepairItemMatcher的单例实例
        """
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance


# 使用示例
if __name__ == "__main__":
    std_items = ['更换空调滤芯', '更换刹车片', '检查电瓶']
    train_items = ['更换车内空调过滤器', '换前轮刹车片', '电瓶检查', '保养：空滤更换']
    test_items = ['换车内空调滤清器', '前刹车片更换', '电瓶状态检查', '保养：空滤']

    # 中文支持
    # matcher = RepairItemMatcher(
    #     'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # matcher = RepairItemMatcher('paraphrase-MiniLM-L6-v2')
    matcher = RepairItemMatcher()

    matcher.load_standard_items(std_items)
    initial_matches = matcher.find_matches(train_items)
    print('initial_matches:', initial_matches)

    # 手动校正匹配结果
    # corrected_matches = {k: v for k, v in initial_matches.items()}  # 这里不需要更改
    corrected_matches = {'更换车内空调过滤器': 0, '换前轮刹车片': 1, '电瓶检查': 2, '保养：空滤更换': 0}

    # 准备训练数据
    train_dataloader = matcher.prepare_training_data(corrected_matches,
                                                     std_items)

    # 微调模型
    matcher.fine_tune_model(train_dataloader)

    # 测试模型
    matcher.test_model(test_items, std_items)
