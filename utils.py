from torch.utils import data
from config import *
import torch
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

from transformers import logging
logging.set_verbosity_error()

class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        if type == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif type == 'test':
            sample_path = TEST_SAMPLE_PATH

        # 读取所有行，并可能跳过标题行
        with open(sample_path, encoding='utf-8') as file:
            self.lines = file.readlines()
            if self.lines[0].strip().split(',')[1] == 'label':
                self.lines = self.lines[1:]  # 跳过第一行

        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        # 使用rsplit安全地分割文本和标签
        line = self.lines[index]
        text, label, _ = line.rsplit(',', 2)
        text = text.strip('"')
        label = label.strip()  # 清除可能的空格和换行符
        tokened = self.tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']
        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len
        target = int(label)
        return torch.tensor(input_ids[:TEXT_LEN]), torch.tensor(mask[:TEXT_LEN]), torch.tensor(target)

def get_label():
    text = open(LABEL_PATH, encoding='utf-8').read()
    id2label = text.split()
    return id2label, {v: k for k, v in enumerate(id2label)}

def evaluate(pred, true, target_names=None, output_dict=False):
    return classification_report(
        true,
        pred,
        digits=4,
        target_names=target_names,
        labels=range(NUM_CLASSES),
        output_dict=output_dict,
        zero_division=0,
    )

if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=2)
    print(iter(loader).next())
