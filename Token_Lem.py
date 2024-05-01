from work_data import data
import spacy
import squarify
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


tokn = spacy.load("en_core_web_sm")
texts = data['text'].tolist()  
tokens = []
lemmas = []

for doc in tokn.pipe(texts, batch_size=50):
    tokens.append([token.text for token in doc])
    lemmas.append([token.lemma_ for token in doc])

data['tokens'] = tokens
data['lemmas'] = lemmas

all_lemmas = [lemma for sublist in lemmas for lemma in sublist]
lemma_freq = Counter(all_lemmas)

most_common_lemmas = lemma_freq.most_common(10)

lemmas, freqs = zip(*most_common_lemmas)


plt.figure(figsize=(10,5))
plt.bar(lemmas, freqs)
plt.title('Top 10 Lemmas')
plt.xlabel('Lemmas')
plt.ylabel('Frequency')
plt.show()


X = data['lemmas']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)













 









