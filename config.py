TRAIN_SAMPLE_PATH = './data/input/train.csv'
TEST_SAMPLE_PATH = './data/input/test.csv'

LABEL_PATH = './data/input/class.txt'

BERT_PAD_ID = 0
TEXT_LEN = 100

BATCH_SIZE = 32

BERT_MODEL = './huggingface/bert-base-uncased'

MODEL_DIR = './output/models/'

EMBEDDING_DIM = 768
NUM_FILTERS = 256
NUM_CLASSES = 2
FILTER_SIZES = [2, 3, 4]

EPOCH = 60
LR = 3e-4


DROPOUT = 0.1

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(torch.tensor([1,2,3]).to(DEVICE))