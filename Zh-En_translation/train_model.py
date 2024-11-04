# train_model.py
import os
import torch
from torch.utils.data import DataLoader
from models import SeqAttnLSTM
from data import SequenceDataset, sequence_collate_fn
from build_data import build_paralell_data, build_data_tokenized
from train import train, test  # 替换为您的训练和测试函数

# 步骤 1: 处理数据
def process_data(data_dir):
    # 构建平行数据
    build_paralell_data()
    # 生成标记化数据
    build_data_tokenized(data_dir, lang='en')  # 英文
    build_data_tokenized(data_dir, lang='zh')  # 中文

# 步骤 2: 创建数据集
def create_dataset(data_dir):
    return SequenceDataset(data_dir=data_dir,  max_length=100, block_ids=0)

# 步骤 3: 训练模型
def train_model(dataset):
    input_dim = 256
    hidden_dim = 256
    bidirectional = True
    num_layers = 2
    vocab_size = 15901
    epochs = 25
    clip = 1.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad_idx = 0

    seq2seq = SeqAttnLSTM(vocab_size, input_dim, hidden_dim, bidirectional, num_layers, device=device).to(device)
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for e in range(epochs):
        train_loss = train(seq2seq, train_loader, optimizer, criterion, clip)
        print(f'Epoch: {e + 1:02} | Train Loss: {train_loss:.5f}')
        test(seq2seq, test_loader)

if __name__ == "__main__":
    data_directory = "./sequencedataset"  # 替换为数据目录
    process_data(data_directory)
    dataset = create_dataset(data_directory)
    train_model(dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def collate_fn(batch):
        return sequence_collate_fn(batch, 100, device=device)
