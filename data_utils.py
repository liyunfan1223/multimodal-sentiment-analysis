import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import os
from PIL import Image
from transformers import AutoFeatureExtractor
from transformers import BertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_image(file_path_image):
    image = Image.open(file_path_image)
    return image


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

    for guid, tag in test_label.values:
        data_dict = {}
        data_dict['guid'] = int(guid)
        data_dict['tag'] = None
        data_dict['image'], data_dict['text'] = get_image_and_text(data_folder_path, guid)
        test_data_list.append(data_dict)

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
    return decode

def data_preprocess(train_data_list, test_data_list):
    """
    数据预处理，清洗文本数据
    """

    for data in train_data_list:
        data['text'] = clean_text(data['text'])

    for data in test_data_list:
        data['text'] = clean_text(data['text'])

    return train_data_list, test_data_list

def collate_fn(data_list):
    guid = [data['guid'] for data in data_list]
    tag = [data['tag'] for data in data_list]
    image = [data['image'] for data in data_list]
    image = feature_extractor(image, return_tensors="pt")
    text = [data['text'] for data in data_list]
    text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=30)

    return guid, None if tag[0] == None else torch.LongTensor(tag), image, text

def get_data_loader(train_data_list, test_data_list) -> (DataLoader, DataLoader, DataLoader):
    """
    生成数据负载器
    """

    train_data_length = int(len(train_data_list) * 0.9)
    valid_data_length = len(train_data_list) - train_data_length
    train_dataset, valid_dataset = random_split(dataset=train_data_list, lengths = [train_data_length, valid_data_length])

    train_data_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=True,
        drop_last=False,
    )

    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=True,
        drop_last=False,
    )

    test_data_loader = DataLoader(
        dataset=test_data_list,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=False,
        drop_last=False,
    )

    return train_data_loader, valid_data_loader, test_data_loader


def calc_metrics(target, pred):
    """
    计算评估指标，分别为加权准确率、加权召回率、加权F1-score、宏准确率、宏召回率、宏F1-score
    """

    precision_w = precision_score(target, pred, average='weighted')
    recall_w = recall_score(target, pred, average='weighted')
    f1_w = f1_score(target, pred, average='weighted')
    precision = precision_score(target, pred, average='macro')
    recall = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')
    return precision_w, recall_w, f1_w, precision, recall, f1