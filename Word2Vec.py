import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim 
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Token_Lem import X_train, y_train
from work_data import data

X_train_flat = [word for sentence in X_train for word in sentence]

word_counts = Counter(X_train_flat)
word_list = list(word_counts.keys())
word_dict = {word: i for i, word in enumerate(word_list)}

pairs = []

for i in range(1, len(X_train_flat) - 1):
    pairs.append((word_dict[X_train_flat[i-1]], word_dict[X_train_flat[i]]))
    pairs.append((word_dict[X_train_flat[i+1]], word_dict[X_train_flat[i]]))


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)  

    def forward(self, context_word):
        out = self.embeddings(context_word)
        out = self.linear(out)  
        return out


vocab_size = len(word_list)
embedding_dim = 50

model = Word2Vec(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in tqdm(range(1), leave=True):  
    loss_val = 0
    for context, target in tqdm(pairs, leave=True):
        context_var = Variable(torch.LongTensor([context]))
        model.zero_grad()
        log_probs = model(context_var)
        loss = criterion(log_probs.view(1, -1), Variable(torch.LongTensor([target])))
        loss.backward()
        optimizer.step()

        loss_val += loss.data
    print(f'Loss at epoch {epoch}: {loss_val/len(pairs)}')
    
    
weights = model.embeddings.weight.data.numpy()

tsne = TSNE(n_components=2, perplexity=min(30, len(weights)-1))  
embed_tsne = tsne.fit_transform(weights)

plt.figure(figsize=(10, 10))
colors = cm.rainbow(np.linspace(0, 1, len(word_list)))  

for idx in range(len(word_list)):
    plt.scatter(*embed_tsne[idx, :], color=colors[idx], s=900)
    plt.annotate(word_list[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

plt.title('t-SNE visualization of Word2Vec embeddings')
plt.show()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, embedding_dim, n_layers, bidirectional, dropout):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

model = RNN(input_size=10, hidden_size=20, output_size=1, vocab_size=1000, embedding_dim=300, n_layers=2, bidirectional=True, dropout=0.5)









