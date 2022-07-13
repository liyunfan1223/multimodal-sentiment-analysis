import torch
import torch.nn as nn
from transformers import ResNetModel
from transformers import BertModel, BertPreTrainedModel, BertLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultimodalModel(BertPreTrainedModel):

    def __init__(self, config):
        super(MultimodalModel, self).__init__(config)
        self.bert = BertModel(config)
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-152")
        self.comb_attention = BertLayer(config)
        self.W = nn.Linear(in_features=2048, out_features=config.hidden_size)
        self.image_pool = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.text_pool = nn.Sequential (
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Linear(in_features=config.hidden_size * 2, out_features=3)
        self.classifier_single = nn.Linear(in_features=config.hidden_size, out_features=3)

    def forward(self, image_input = None, text_input = None):
        if (image_input is not None) and (text_input is not None):
            """both image and text"""

            """提取文本特征"""
            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state

            """提取图像特征"""
            image_features = self.resnet(**image_input).last_hidden_state.view(-1, 2048, 49).permute(0, 2, 1).contiguous()
            image_pooled_output, _ = image_features.max(1)
            image_hidden_state = self.W(image_pooled_output).unsqueeze(1)
            """拼接文本和图像，拼接得到共同特征"""
            image_text_hidden_state = torch.cat([image_hidden_state, text_hidden_state], 1)

            """设置attention_mask，padding部分置-10000"""
            attention_mask = text_input.attention_mask
            image_attention_mask = torch.ones((attention_mask.size(0), 1)).to(device)
            attention_mask = torch.cat([image_attention_mask, attention_mask], 1).unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000

            """利用self-attention机制进行多模态融合"""
            image_text_attention_state = self.comb_attention(image_text_hidden_state, attention_mask)[0]

            """分别提取图像和文本特征，拼接"""
            image_pooled_output = self.image_pool(image_text_attention_state[:, 0, :])
            text_pooled_output = self.text_pool(image_text_attention_state[:, 1, :])
            final_output = torch.cat([image_pooled_output, text_pooled_output], 1)

            """利用拼接向量进行分类"""
            out = self.classifier(final_output)
            return out

        elif image_input is None:
            """text only"""
            assert(text_input is not None)
            text_features = self.bert(**text_input)
            text_hidden_state = text_features.last_hidden_state
            attention_mask = text_input.attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000
            attention_state = self.comb_attention(text_hidden_state, attention_mask)[0]
            text_pooled_output = self.text_pool(attention_state[:, 0, :])
            out = self.classifier_single(text_pooled_output)
            return out

        elif text_input is None:
            """image only"""
            assert(image_input is not None)
            image_features = self.resnet(**image_input).last_hidden_state.view(-1, 2048, 49).permute(0, 2, 1).contiguous()
            image_pooled_output, _ = image_features.max(1)
            image_hidden_state = self.W(image_pooled_output).unsqueeze(1)
            image_attention_mask = torch.ones((image_hidden_state.size(0), 1)).to(device)
            attention_mask = (1.0 - image_attention_mask) * -10000
            attention_state = self.comb_attention(image_hidden_state, attention_mask)[0]
            image_pooled_output = self.image_pool(attention_state[:, 0, :])
            out = self.classifier_single(image_pooled_output)
            return out