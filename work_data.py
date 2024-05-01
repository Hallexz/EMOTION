import pandas as pd
import squarify
import matplotlib.pyplot as plt
import spacy
from sklearn.model_selection import train_test_split

data = pd.read_csv('text.csv')
vocab = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
data = data.drop(columns = ['Unnamed: 0'])
data = data.dropna()
data['label'] = data['label'].astype(float)


data['emotion_name'] = data['label'].map(vocab)
values = data['emotion_name'].value_counts()
values = values.reset_index()
values.columns = ['Emotion', 'Count']
percentages = 100.*values['Count']/sum(values['Count'])
labels = ['%s, \n%.1f %%' % (label, percentage) for label, percentage in zip(values['Emotion'], percentages)]

color_dict = {'sadness': 'blue', 
              'joy': 'yellow', 
              'love': 'red', 
              'anger': 'black', 
              'fear': 'purple', 
              'surprise': 'green'}
colors = [color_dict[emotion] for emotion in values['Emotion']]

plt.rcParams['font.family'] = 'monospace'

fig, ax = plt.subplots()
squarify.plot(sizes=values['Count'], 
              label=labels, alpha=0.7, 
              ax=ax, color=colors, 
              text_kwargs={'fontsize': 12, 'color': 'black'}, pad=1)

plt.axis('off')
plt.show()









