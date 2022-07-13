import torch
import torch.nn as nn

class MultiModalModel(nn.Module):

    def __init__(self, num_embeddings, image_size = 64, image_channels = 3, embedding_dim = 300, hidden_size = 256, ):
        super(MultiModalModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.image_size = image_size
        self.image_channels = image_channels
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.vgg = VGG(self.image_channels, self.image_size, res = True)
        self.att_weights = nn.Linear(in_features=(self.image_size // 16) ** 2 * 512, out_features = self.hidden_size * 2)


        self.embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim = self.embedding_dim, padding_idx=0)
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

        self.classifier = nn.Linear(in_features=self.hidden_size * 2, out_features=3)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, image, text):
        # VGG网络计算图片向量表示
        vgg_features = self.vgg(image)
        att_vector = self.att_weights(vgg_features)
        att_vector = self.tanh(att_vector)

        # 词嵌入
        text = self.embedding(text)

        # 双层GRU提取文本特征
        text_hidden, _ = self.gru(text)
        text_hidden = self.tanh(text_hidden)

        # 计算注意力得分
        att_score = torch.einsum('BJ,BIJ->BI', att_vector, text_hidden)
        att_score = self.softmax(att_score)

        # 根据注意力得分计算上下文向量
        context_vector = torch.einsum('BIJ,BI->BJ', text_hidden, att_score)

        # 根据上下文向量计算分类情况
        out = self.classifier(context_vector)

        return out


class VGG(nn.Module):
    def __init__(self, input_channels, size, cfg=[64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512,'M'], res=True):
        super(VGG, self).__init__()
        self.res = res       # 是否带残差连接
        self.cfg = cfg       # 配置列表
        self.input_channels = input_channels   # 初始输入通道数
        self.futures = self.make_layer()

    def make_layer(self):
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Block(self.input_channels, v, self.res))
                self.input_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.futures(x)
        out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return out

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, res=True, stride=1):
        super(Block, self).__init__()
        self.res = res     # 是否带残差连接
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.shortcut = nn.Sequential()
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out