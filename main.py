import torch
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from data_utils import get_data_list, data_preprocess, get_data_loader, calc_metrics
from model import MultimodalModel

import argparse


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_only', action='store_true', help='仅使用文本')
    parser.add_argument('--image_only', action='store_true', help='仅使用图片')
    parser.add_argument('--do_test', action='store_true', help='使用训练后的模型对测试集进行预测')
    parser.add_argument('--lr', default=5e-5, help='设置学习率', type=float)
    parser.add_argument('--weight_decay', default=1e-2, help='设置权重衰减', type=float)
    parser.add_argument('--epochs', default=10, help='设置训练轮数', type=int)
    parser.add_argument('--seed', default=233, help='设置随机种子', type=int)
    args = parser.parse_args()
    return args


args = init_argparse()
print('args:', args)
"""text_only和image_only互斥"""
assert((args.text_only and args.image_only) == False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
"""在种子不变的情况下保证结果一致"""
torch.backends.cudnn.deterministic = True


def model_train():
    """训练模型并保存至./model.pth"""

    train_data_list, test_data_list = get_data_list()
    train_data_list, test_data_list = data_preprocess(train_data_list, test_data_list)
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(train_data_list, test_data_list)
    model = MultimodalModel.from_pretrained('bert-base-uncased')
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(lr=args.lr, params=optimizer_grouped_parameters)
    criterion = CrossEntropyLoss()
    best_rate = 0
    print('[START_OF_TRAINING_STAGE]')
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        model.train()
        for idx, (guid, tag, image, text) in enumerate(train_data_loader):
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)
            if args.text_only:
                out = model(image_input=None, text_input=text)
            elif args.image_only:
                out = model(image_input=image, text_input=None)
            else:
                out = model(image_input=image, text_input=text)
            loss = criterion(out, tag)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(guid)
            pred = torch.max(out, 1)[1]
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        total_loss /= total
        print('[EPOCH{:02d}]'.format(epoch + 1), end='')
        print('[TRAIN] - LOSS:{:.6f}'.format(total_loss), end='')
        rate = correct / total * 100
        print(' ACC_RATE:{:.2f}%'.format(rate), end='')
        metrics = calc_metrics(target_list, pred_list)
        print(' WEIGHTED_ACC: {:.2f}% WEIGHTED_F1: {:.2f}% MAC_ACC: {:.2f}% MAC_F1: {:.2f}%'.format(metrics[0] * 100,
                                                                                                    metrics[2] * 100,
                                                                                                    metrics[3] * 100,
                                                                                                    metrics[5] * 100))

        total_loss = 0
        correct = 0
        total = 0
        target_list = []
        pred_list = []
        model.eval()

        for guid, tag, image, text in valid_data_loader:
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)

            if args.text_only:
                out = model(image_input=None, text_input=text)
            elif args.image_only:
                out = model(image_input=image, text_input=None)
            else:
                out = model(image_input=image, text_input=text)

            loss = criterion(out, tag)

            total_loss += loss.item() * len(guid)
            pred = torch.max(out, 1)[1]
            total += len(guid)
            correct += (pred == tag).sum()

            target_list.extend(tag.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())

        total_loss /= total
        print('         [EVAL]  - LOSS:{:.6f}'.format(total_loss), end='')
        rate = correct / total * 100
        print(' ACC_RATE:{:.2f}%'.format(rate), end='')
        metrics = calc_metrics(target_list, pred_list)
        print(' WEIGHTED_ACC: {:.2f}% WEIGHTED_F1: {:.2f}% MAC_ACC: {:.2f}% MAC_F1: {:.2f}%'.format(metrics[0] * 100,
                                                                                                    metrics[2] * 100,
                                                                                                    metrics[3] * 100,
                                                                                                    metrics[5] * 100))

        if rate > best_rate:
            best_rate = rate
            print('         [SAVE] BEST ACC_RATE ON THE VALIDATION SET:{:.2f}%'.format(rate))
            torch.save(model.state_dict(), 'model.pth')
        print()
    print('[END_OF_TRAINING_STAGE]')

def model_test():
    """利用训练好的./model.pth对测试集进行预测，结果保存至output/test_with_label.txt"""

    train_data_list, test_data_list = get_data_list()
    train_data_list, test_data_list = data_preprocess(train_data_list, test_data_list)
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(train_data_list, test_data_list)
    model = MultimodalModel.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)
    print('[START_OF_TESTING_STAGE]')
    guid_list = []
    pred_list = []
    model.eval()

    for guid, tag, image, text in test_data_loader:
        image = image.to(device)
        text = text.to(device)

        if args.text_only:
            out = model(image_input=None, text_input=text)
        elif args.image_only:
            out = model(image_input=image, text_input=None)
        else:
            out = model(image_input=image, text_input=text)

        pred = torch.max(out, 1)[1]
        guid_list.extend(guid)
        pred_list.extend(pred.cpu().tolist())

    pred_mapped = {
        0: 'negative',
        1: 'neutral',
        2: 'positive',
    }
    with open('output/test_with_label.txt', 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for guid, pred in zip(guid_list, pred_list):
            f.write(f'{guid},{pred_mapped[pred]}\n')
        f.close()
        print('[PREDICTION] SAVE TO output/test_with_label.txt')
    print('[END_OF_TESTING_STAGE]')

if __name__ == "__main__":
    if args.do_test:
        model_test()
    else:
        model_train()