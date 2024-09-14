from sentence_transformers import SentenceTransformer, util
import numpy as np


class SimilarityModel:
    # 初始化相似度模型
    def __init__(self, corpus):
        self.corpus = corpus
        # 使用Sentence-BERT模型
        self.model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2')
        # 计算语料库的句子嵌入
        self.corpus_embeddings = self.model.encode(corpus,
                                                   convert_to_tensor=True)

    # 查找与输入句子最相似的句子
    def find_most_similar(self, input_sentence, top_n=1):
        # 将输入句子转换为嵌入
        input_embedding = self.model.encode(input_sentence,
                                            convert_to_tensor=True)
        # 计算输入句子与语料库中句子的余弦相似度
        cosine_similarities = util.pytorch_cos_sim(input_embedding,
                                                   self.corpus_embeddings)[0]
        # 获取最相似的句子的索引
        top_indices = np.argsort(cosine_similarities.numpy())[-top_n:][::-1]
        results = []
        # 收集最相似句子及其相似度
        for idx in top_indices:
            results.append((self.corpus[idx], cosine_similarities[idx].item()))
        return results
