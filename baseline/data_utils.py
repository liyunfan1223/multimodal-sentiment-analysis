import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import os
import cv2
from collections import defaultdict
from nltk import sent_tokenize, word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

torch.manual_seed(233)
np.random.seed(233)

def get_image(file_path_image):
    im = cv2.imread(file_path_image)
    return np.asarray(im).transpose(2, 0, 1)


def get_text(file_path_text):
    with open(file_path_text, 'rb') as f:
        text = f.readline().strip()
    return text


def get_image_and_text(data_folder_path, guid):
    file_path_image = os.path.join(data_folder_path, f'{int(guid)}.jpg')
    file_path_text = os.path.join(data_folder_path, f'{int(guid)}.txt')
    return get_image(file_path_image), get_text(file_path_text)


def get_data_list(data_folder_path = 'data', label_path = 'label') -> (list, list):
    """
    读取训练和测试数据，分别返回训练集和测试集
    :param data_folder_path: 存放图像和文本的文件夹
    :param label_path: 存放标签的文件夹
    :return: train_data_list, test_data_list
    """
    train_data_list = []
    test_data_list = []
    train_label_path = os.path.join(label_path, 'train.txt')
    test_label_path = os.path.join(label_path, 'test_without_label.txt')

    train_label = pd.read_csv(train_label_path)
    test_label = pd.read_csv(test_label_path)

    tag_mapping = {
        'positive' : 2,
        'neutral' : 1,
        'negative' : 0,
    }
    for idx, (guid, tag) in enumerate(train_label.values):
        data_dict = {}
        data_dict['guid'] = int(guid)
        data_dict['tag'] = tag_mapping[tag]
        data_dict['image'], data_dict['text'] = get_image_and_text(data_folder_path, guid)
        train_data_list.append(data_dict)

    # for guid, tag in test_label.values:
    #     data_dict = {}
    #     data_dict['guid'] = int(guid)
    #     data_dict['tag'] = None
    #     data_dict['image'], data_dict['text'] = get_image_and_text(data_folder_path, guid)
    #     test_data_list.append(data_dict)
        # break

    return train_data_list, test_data_list


def clean_text(text: bytes):
    try:
        decode = text.decode(encoding='utf-8')
    except:
        try:
            decode = text.decode(encoding='GBK')
        except:
            try:
                decode = text.decode(encoding='gb18030')
            except:
                decode = str(text)

    tokens = word_tokenize(decode)
    interruptions = [',', '.', ':', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    tokens = [token for token in tokens if token not in interruptions]
    tokens = [token.lower() for token in tokens if token not in interruptions]
    return tokens


def data_preprocess(train_data_list, test_data_list):
    """
    数据预处理，调整图像大小、清洗文本数据
    :param train_data_list: 训练数据列表
    :param test_data_list: 测试数据列表
    :return: train_data_list, test_data_list, vocab
    """

    m = nn.AdaptiveMaxPool2d(64)
    vocab = defaultdict()
    for data in train_data_list:
        data['image'] = m(torch.Tensor(data['image']))
        data['text'] = clean_text(data['text'])
        mapped_text = []
        for word in data['text']:
            if not word in vocab:
                vocab[word] = len(vocab) + 1
            mapped_text.append(vocab[word])
        mapped_text.extend([0] * (50 - len(mapped_text)))
        data['text'] = mapped_text[:50]

    for data in test_data_list:
        data['image'] = m(torch.Tensor(data['image']))
        data['text'] = clean_text(data['text'])
        mapped_text = []
        for word in data['text']:
            if not word in vocab:
                vocab[word] = len(vocab) + 1
            mapped_text.append(vocab[word])
        mapped_text.extend([0] * (50 - len(mapped_text)))
        data['text'] = mapped_text[:50]

    return train_data_list, test_data_list, vocab

def collate_fn(data_list):
    guid = [data['guid'] for data in data_list]
    tag = [data['tag'] for data in data_list]
    image = [data['image'].cpu().numpy() for data in data_list]
    image = np.array(image)
    text = [data['text'] for data in data_list]

    return guid, torch.LongTensor(tag), torch.Tensor(image), torch.LongTensor(text)

def get_data_loader(train_data_list, test_data_list) -> (DataLoader, DataLoader, DataLoader):
    """
    生成数据负载器
    :param train_data_list: 训练数据列表
    :param test_data_list: 测试数据列表
    :return: train_data_loader, valid_data_loader, test_data_loader
    """

    train_data_length = int(len(train_data_list) * 0.8)
    valid_data_length = len(train_data_list) - train_data_length
    train_dataset, valid_dataset = random_split(dataset=train_data_list, lengths = [train_data_length, valid_data_length])

    train_data_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )

    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        collate_fn=collate_fn,
        batch_size=32,
        shuffle=True,
        drop_last=False,
    )

    test_data_loader = DataLoader(
        dataset=test_data_list,
        collate_fn=collate_fn,
        batch_size=32,
        shuffle=False,
        drop_last=False,
    )

    return train_data_loader, valid_data_loader, test_data_loader

def calc_metrics(target, pred):
    precision_w = precision_score(target, pred, average='weighted')
    recall_w = recall_score(target, pred, average='weighted')
    f1_w = f1_score(target, pred, average='weighted')
    precision = precision_score(target, pred, average='macro')
    recall = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')
    return precision_w, recall_w, f1_w, precision, recall, f1


if __name__ == "__main__":
    train_data_list, test_data_list = get_data_list()
    train_data_list, test_data_list = data_preprocess(train_data_list, test_data_list)
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(train_data_list, test_data_list)
    for batch in train_data_loader:
        print(batch)
        break