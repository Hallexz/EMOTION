import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from Word2Vec import  weights
from Token_Lem import y_train


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, embedding_dim, n_layers, bidirectional, dropout):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

model = RNN(input_size=10, hidden_size=20, output_size=1, vocab_size=1000, embedding_dim=300, n_layers=2, bidirectional=True, dropout=0.5)

# Предположим, что у нас есть входные данные и целевые данные
input_data = torch.randn(64, 10)  # размерность: (batch_size, sequence_length)
target_data = torch.randint(0, 1, (64,))  # размерность: (batch_size)

tokens = weights
labels = y_train

train_data = [torch.LongTensor(sentence) for sentence in tokens]
train_lenghts = [len(sentence) for sentence in train_data]
train_labels = torch.FloatTensor(labels)

train_data, train_lenghts, train_labels = zip(*sorted(zip(train_data,
                                                          train_lenghts,
                                                          train_labels),
                                                      key=lambda x: x[1], reverse=True))

# Преобразуем train_data в тензор
train_data_tensor = torch.stack(train_data)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in tqdm(range(10), leave=True):  
    optimizer.zero_grad()
    output = model(train_data_tensor, train_lenghts)  # предполагаем, что все последовательности имеют длину 10
    loss = criterion(output.squeeze(), train_labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')






















    