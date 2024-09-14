import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.business_logic.data_handling import db
from src.models.MLP import MLP
from src.utils.custom_utils import update_streamlit

import pdb

SHIP_ATTRIBUTES_TRANSLATION = {
    'ship_name': '船舶名称',
    'former_name': '曾用名',
    'ship_type': '船舶类型',
    'gross_tonnage': '总吨',
    'total_power': '总功率',
    'year_of_manufacture': '出厂年度',
    'service_life': '使用年限',
    'number_of_main_engines': '主机数量',
    'main_engine_power': '主机功率',
    'generator_power': '发电机功率',
    'sailing_distance': '出航里程(公里)',
    'sailing_time': '出航时间(小时)',
    'sailing_count': '出航次数(次)',
    'fuel_consumption': '燃油消耗(吨)',
    'stat_year': '统计年份'
}
FUEL_MODEL_NAME = 'MLP_20240715_155846.pt'
FUEL_FEATURE_SCALER_NAME = 'feature_scaler_20240715_155846.pkl'
FUEL_LABEL_SCALER_NAME = 'label_scaler_20240715_155846.pkl'


def get_real_path(filename, directory=''):
    return os.path.join(os.getcwd(), directory, filename)


# 线性回归简单计算剔除异常数据
def remove_outliers(df, per=0.95):
    """用线性回归剔除油耗明显异常的数据，避免影响训练结果

    Args:
        df (Dataframe): 训练数据集
        per (float, optional): 数据范围，低于这个范围的被剔除. Defaults to 0.95.

    Returns:
        Dataframe: 处理后的数据集
    """
    X = df.drop(['fuel_consumption', 'ship_name', 'former_name', 'ship_type'],
                axis=1)
    y = df['fuel_consumption']
    model = LinearRegression().fit(X, y)
    df_temp = df.copy()
    df_temp['residuals'] = model.predict(X) - y
    threshold = df_temp['residuals'].abs().quantile(per)
    df = df[df_temp['residuals'].abs() <= threshold]
    # df = df.drop(['residuals'], axis=1)
    return df


# 特征工程
def features_process(df):

    # 未出航修正
    df.loc[(df['sailing_distance'] == 0) & (df['sailing_time'] == 0),
           'fuel_consumption'] = 0

    # 分离特征和标签
    X = df.drop(['ship_id', 'ship_name', 'former_name', 'fuel_consumption'],
                axis=1).values  # 特征
    y = df['fuel_consumption'].values  # 标签

    # 格式化列的排列顺序
    X = df[[
        'number_of_main_engines', 'main_engine_power', 'generator_power',
        'sailing_distance', 'sailing_count', 'sailing_time', 'stat_year',
        'ship_type', 'gross_tonnage', 'total_power', 'year_of_manufacture',
        'service_life'
    ]].values

    return X, y


def consumption_train(is_save=False):

    # 数据预处理
    query = "SELECT * FROM AI_FEAT_SHIP_FUEL"
    df_raw = pd.read_sql(query, db)

    # 将列名转换为小写以匹配
    df_raw.columns = df_raw.columns.str.lower()

    df = remove_outliers(df_raw)

    # 特征工程
    X, y = features_process(df)

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # 归一化
    feature_scaler = RobustScaler()
    label_scaler = RobustScaler()

    feature_scaler.fit(X_train)
    label_scaler.fit(y_train.reshape(-1, 1))

    X_train_scaled = feature_scaler.transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    y_train_scaled = label_scaler.transform(y_train.reshape(-1, 1))
    y_test_scaled = label_scaler.transform(y_test.reshape(-1, 1))

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # 创建TensorDataset和DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor,
                                                   y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # 模型
    input_size = X_train.shape[1]  # 特征数量
    hidden_sizes = [64, 32, 16]  # 隐藏层大小
    output_size = 1  # 输出大小（预测燃油消耗）

    # 特征权重，变更依次为：里程、时间、总吨、总功率
    weights = torch.tensor([1, 1, 1, 2, 1, 2, 1, 1, 2, 4, 1, 1])
    model = MLP(input_size, hidden_sizes, output_size, weights)

    criterion = nn.MSELoss()  # 回归问题使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            update_streamlit(loss.item(), epoch + 1)

        # 在测试集上评估模型
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
        test_loss /= len(test_loader.dataset)
        print(f'Epoch {epoch+1}, Test Loss: {test_loss}')

    if is_save:
        save_model_scaler(model, feature_scaler, label_scaler)


def save_model_scaler(model, feature_scaler, label_scaler, dir='src/models/'):
    """
    将训练好的模型和scaler保存到包含模型名称、当前日期和时间的文件中。

    参数:
        model (torch.nn.Module): 要保存的训练模型。
        scaler (object): 要保存的scaler对象。
        dir (str): 模型和scaler将被保存的基本文件路径。
    """

    # 格式化当前日期和时间
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用模型的类名和当前日期时间来创建文件名
    model_filename = f"{model.__class__.__name__}_{date_time}.pt"
    feature_scaler_filename = f"feature_scaler_{date_time}.pkl"
    label_scaler_filename = f"label_scaler_{date_time}.pkl"
    # 合并路径和文件名
    # model_full_path = get_real_path(model_filename, dir)
    # feature_scaler_full_path = get_real_path(feature_scaler_filename, dir)
    # label_scaler_full_path = get_real_path(label_scaler_filename, dir)

    # 临时调试
    model_full_path = get_real_path(FUEL_MODEL_NAME, dir)
    feature_scaler_full_path = get_real_path(FUEL_FEATURE_SCALER_NAME, dir)
    label_scaler_full_path = get_real_path(FUEL_LABEL_SCALER_NAME, dir)

    torch.save(model, model_full_path)
    with open(feature_scaler_full_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(label_scaler_full_path, 'wb') as f:
        pickle.dump(label_scaler, f)
    print(f"model已保存到 {model_full_path}")
    print(f"feature_scaler已保存到 {feature_scaler_full_path}")
    print(f"label_scaler已保存到 {label_scaler_full_path}")


def load_model_scaler(model_name,
                      feature_scaler_name,
                      label_scaler_name,
                      dir='src/models/'):
    """
    加载模型和scaler，文件名中不使用当前日期时间，而是使用提供的后缀。

    参数:
        model_class (class): 要加载的模型的类。
        suffix (str): 加载模型和scaler文件的后缀。
        dir (str): 模型和scaler文件的存储路径。
    """

    model_full_path = get_real_path(model_name, dir)
    feature_scaler_full_path = get_real_path(feature_scaler_name, dir)
    label_scaler_full_path = get_real_path(label_scaler_name, dir)
    model = torch.load(model_full_path)
    with open(feature_scaler_full_path, 'rb') as f:
        feature_scaler = pickle.load(f)
    with open(label_scaler_full_path, 'rb') as f:
        label_scaler = pickle.load(f)
    return model, feature_scaler, label_scaler


def extract_chinese_columns(df):
    """
    接收一个dataframe，将其所有中文名字段取出，输出中英文名对照的字典。
    
    Args:
        df (pd.DataFrame): 输入的数据帧。
        
    Returns:
        dict: 包含中文名和对应英文名的字典。
    """
    import re
    chinese_columns = {
        col: re.findall(r'[\u4e00-\u9fff]+', col)
        for col in df.columns if re.search(r'[\u4e00-\u9fff]+', col)
    }
    return {
        col: ''.join(ch_names)
        for col, ch_names in chinese_columns.items() if ch_names
    }


def predict_fuel_consumption(df):
    """
    接收一个DataFrame，字段名为SHIP_ATTRIBUTES_TRANSLATION中的部分或者全部key，
    将其对应翻译为中文名后，按照SHIP_ATTRIBUTES_TRANSLATION原有的顺序输入到feature_process函数中，
    然后进行预测，最终输出ship_name和预测油耗字段。

    Args:
        df (pd.DataFrame): 输入的数据帧，包含船舶相关属性。

    Returns:
        pd.DataFrame: 包含船舶名称和预测油耗的数据帧。
    """

    # 载入预测模型和归一化模型
    model, feature_scaler, label_scaler = load_model_scaler(
        FUEL_MODEL_NAME, FUEL_FEATURE_SCALER_NAME, FUEL_LABEL_SCALER_NAME)
    # model, _, _ = load_model_scaler(FUEL_MODEL_NAME, FUEL_FEATURE_SCALER_NAME,
    #                                 FUEL_LABEL_SCALER_NAME)

    # 特征处理
    X, _ = features_process(df)
    # 这里不能fit
    X_scaled = feature_scaler.transform(X)

    # # 增加重要数据的权重，例如总功率和里程

    # X_scaled['total_power'] = FEATURE_WEIGHTS * X_scaled['total_power']
    # X_scaled[
    #     'sailing_distance'] = FEATURE_WEIGHTS * X_scaled['sailing_distance']

    # 使用训练好的model进行预测
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(torch.tensor(X_scaled, dtype=torch.float32))
        y_pred = y_pred_tensor.squeeze().cpu().numpy()
        print(y_pred)
        # 预测时，反向转换模型输出
        y_pred = label_scaler.inverse_transform(y_pred.reshape(-1, 1))
        # y_pred = np.expm1(y_pred)

    # 使用 squeeze 或 ravel 方法将 y_pred 转换为一维数组
    y_pred = y_pred.squeeze().round(0)  # 或者使用 y_pred.ravel().round(2)

    # 准备输出结果
    result_df = pd.DataFrame({
        'ship_name': df['ship_name'],
        'total_power': df['total_power'],
        'sailing_distance': df['sailing_distance'],
        'fuel_real': df['fuel_consumption'],
        'fuel_predict': y_pred
    })

    return result_df
    # return y_pred


if __name__ == '__main__':
    # 训练模型
    consumption_train(is_save=True)


def show_fuel_predict():
    # 预测油耗
    query = "SELECT * FROM AI_FEAT_SHIP_FUEL"
    df_raw = pd.read_sql(query, db)
    # 将列名转换为小写以匹配
    df_raw.columns = df_raw.columns.str.lower()

    df = df_raw[:30]
    df_pred = predict_fuel_consumption(df)
    return df_pred
