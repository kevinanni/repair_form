from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba


class SimilarityModel:
    # 初始化相似度模型
    def __init__(self, corpus):
        self.corpus = corpus
        # 对语料库中的每个句子进行分词
        self.corpus_cut = [
            ' '.join(jieba.cut(sentence)) for sentence in corpus
        ]
        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer()
        # 计算语料库的TF-IDF矩阵
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_cut)

    # 查找与输入句子最相似的句子
    def find_most_similar(self, input_sentence, top_n=1):
        # 对输入句子进行分词
        input_cut = ' '.join(jieba.cut(input_sentence))
        # 将输入句子转换为TF-IDF向量
        input_vector = self.vectorizer.transform([input_cut])
        # 计算输入句子与语料库中句子的余弦相似度
        cosine_similarities = cosine_similarity(input_vector,
                                                self.tfidf_matrix).flatten()
        # 获取最相似的句子的索引
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]
        results = []
        # 收集最相似句子及其相似度
        for idx in top_indices:
            results.append((self.corpus[idx], cosine_similarities[idx]))
        return results
