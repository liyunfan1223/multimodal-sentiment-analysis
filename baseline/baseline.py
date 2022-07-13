import torch
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from data_utils import get_data_list, data_preprocess, get_data_loader
from model import MultiModalModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 10

def model_train():
    train_data_list, test_data_list = get_data_list()
    train_data_list, test_data_list, vocab = data_preprocess(train_data_list, test_data_list)
    train_data_loader, valid_data_loader, test_data_loader = get_data_loader(train_data_list, test_data_list)
    model = MultiModalModel(num_embeddings=len(vocab) + 1)
    model.to(device)

    optimizer = Adam(lr=3e-4, params=model.parameters())
    criterion = CrossEntropyLoss()
    best_rate = 0


    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        print('[EPOCH{:03d}]'.format(epoch + 1), end='')
        for guid, tag, image, text in train_data_loader:
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)

            out = model(image, text)

            loss = criterion(out, tag)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * len(guid)
            pred = torch.max(out, 1)[1]
            total +=  len(guid)
            correct += (pred == tag).sum()

        print('[TRAIN] - LOSS:{:.6f}'.format(total_loss), end='')
        rate = correct / total * 100
        print(' CORRECT_RATE:{:.2f}%'.format(rate), end='')

        print()
        total_loss = 0
        correct = 0
        total = 0
        model.eval()
        print('          [EVAL]', end='')
        for guid, tag, image, text in valid_data_loader:
            tag = tag.to(device)
            image = image.to(device)
            text = text.to(device)

            out = model(image, text)

            loss = criterion(out, tag)

            total_loss += loss.item() * len(guid)
            pred = torch.max(out, 1)[1]
            total +=  len(guid)
            correct += (pred == tag).sum()

        print(' LOSS:{:.6f}'.format(total_loss), end='')
        rate = correct / total * 100
        print(' CORRECT_RATE:{:.2f}%'.format(rate), end='')
        print('')

if __name__ == "__main__":
    model_train()