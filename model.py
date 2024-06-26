import torch.nn as nn
import torch.nn.functional as F
from config import *
from transformers import BertModel, XLNetModel, DistilBertModel, RobertaModel

from transformers import logging
logging.set_verbosity_error()

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, embedding_dim):
        super(SKConv, self).__init__()
        self.kernels = kernels
        self.convs = nn.ModuleList()
        for k in kernels:
            padding = (k - 1) // 2
            self.convs.append(nn.Conv2d(in_channels, out_channels, (k, embedding_dim), padding=(padding, 0)))
        self.mish = Mish()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(out_channels, out_channels // 16, 1),
            Mish(),
            nn.Conv2d(out_channels // 16, out_channels, 1)
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        conv_outs = [conv(x) for conv in self.convs]

        for i, k in enumerate(self.kernels):
            if k == 3:
                conv_outs[i] = conv_outs[i][:, :, :99, :]
        conv_outs = torch.cat([out.unsqueeze(dim=2) for out in conv_outs], dim=2)

        attention_weights = self.attention(torch.sum(conv_outs, dim=2))
        attention_weights = self.softmax(attention_weights).unsqueeze(2)

        out = (conv_outs * attention_weights).sum(dim=2)

        return out

class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True)
        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
        self.num_layers = 4  # BERT的层数
        self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)  # 可学习的层权重参数

        self.convs = SKConv(1, NUM_FILTERS, FILTER_SIZES, EMBEDDING_DIM)
        self.mish = Mish()
        self.fc = nn.Linear(NUM_FILTERS, NUM_CLASSES)

    def conv_and_pool(self, x):
        x = self.convs(x)
        x = self.mish(x)
        x = F.max_pool2d(x, (x.shape[2], 1))
        return x.squeeze(2)

    def forward(self, input, mask):
        bert_out = self.bert(input, mask)
        hidden_states = bert_out.hidden_states

        # 只选择最后四层的输出
        last_four_layers = hidden_states[-4:]
        stacked_layers = torch.stack(last_four_layers, dim=0)  # 将最后四层堆叠

        weights = F.softmax(self.layer_weights, dim=0)
        weights = weights.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # 使weights形状为 (layers, 1, 1, 1)
        weighted_sum = torch.sum(weights * stacked_layers, dim=0)  # (batch_size, seq_length, hidden_size)

        out = weighted_sum.unsqueeze(1)
        x = self.conv_and_pool(out)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

if __name__ == '__main__':
    model = TextCNN()
    input = torch.randint(0, 3000, (2, TEXT_LEN))
    mask = torch.ones_like(input)
    print(model(input, mask))
