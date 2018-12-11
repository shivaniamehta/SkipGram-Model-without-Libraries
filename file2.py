import random
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

st = set(stopwords.words('english'))
file = open("001.txt", "r")
result = []
for line in file.readlines():
    result.append(line)
corpus = []
for line in result:
    tmp = ''.join(line)
    tmp = tmp.lower()
    tmp = word_tokenize(tmp)
    tmp = ' '.join(w for w in tmp if w not in st)
    corpus.append(tmp)

print("corpus ", corpus)
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)

words = set(words)
print("words ", words)

word2int = {}
int2word = {}

for i, word in enumerate(words):
    word2int[word] = i
print("\nMappings", word2int)

for i, word in enumerate(words):
    int2word[i] = word
print("\n Inverse Mapping", int2word)
sentences = []
for sentence in corpus:
    sentences.append(sentence.split())
print(sentences)

WINDOW_SIZE = random.randint(2, 10)

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])
print(data)

df = pd.DataFrame(data, columns=['input', 'label'])
print(df)

ONE_HOT_DIM = len(words)


def to_one_hot_encoding(pos):
    one_hot_encoding = np.zeros(ONE_HOT_DIM, dtype=int)
    one_hot_encoding[pos] = 1
    return one_hot_encoding


for i in range(len(df.index)):
    df.iloc[i]['input'] = to_one_hot_encoding(word2int.get(df.iloc[i]['input']))
    df.iloc[i]['label'] = to_one_hot_encoding(word2int.get(df.iloc[i]['label']))

print(df)
EMBEDDING_DIM = 2

from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


def activation_function(vec):
    return vec


class NeuralNetwork:

    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate,
                 epochs
                 ):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs

    def feedForwardMechanism(self, input_vector):

        output_vector1 = np.dot(self.wih.T,
                                input_vector)
        output_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.who.T,
                                output_hidden)
        output_network = activation_function(output_vector2)
        output_network_softmax = self.softmax(output_network)
        return output_network_softmax, output_network, output_hidden

    def train(self, df):

        self.wih = np.random.uniform(0.1, 0.8, (len(words), self.no_of_hidden_nodes))
        self.who = np.random.uniform(0.1, 0.8, (self.no_of_hidden_nodes, len(words)))

        for i in range(0, self.epochs):
            self.loss = 0
            for c, t in df:
                second_final, first_final, hidden = self.feedForwardMechanism(c)
                # Calculating error
                EI = np.sum([np.subtract(second_final, word) for word in t], axis=0)
                self.backPropagation(EI, hidden, c)

                # Calculating cross entropy loss
                term = np.dot(np.log1p(second_final), t)
                self.loss = -term

            print("Entropy Loss", -self.loss)

    def run(self, input_vector):

        output_vector1 = np.dot(self.wih.T,
                                input_vector)
        output_vector1 = activation_function(output_vector1)

        output_vector2 = np.dot(self.who.T,
                                output_vector1)
        output_vector2 = activation_function(output_vector2)
        final_vector = self.softmax(output_vector2)
        maximum = np.max(final_vector)
        index = np.argmax(final_vector)

        return int2word[index]

    def softmax(self, x):
        # e_x=np.e**(x-np.max(x))
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def evaluate(self, data):

        res = self.run(data)
        print(res)

    def backPropagation(self, EI, output_hidden, input_vector):
        dl_dw2 = np.outer(output_hidden, EI)
        dl_dw1 = np.outer(input_vector, np.dot(self.who, EI.T))

        # UPDATE WEIGHTS
        self.wih = self.wih - (self.learning_rate * dl_dw1)
        self.who = self.who - (self.learning_rate * dl_dw2)

    def CosineDistance(self, vec1, vec2):
        mul = np.dot(vec1, vec2)
        vec1Magnitude = np.sqrt(np.sum(np.square(vec1)))
        vec2Magnitude = np.sqrt(np.sum(np.square(vec2)))
        # print("Cosine value: ", mul / (vec1Magnitude * vec2Magnitude))
        return (mul / (vec1Magnitude * vec2Magnitude))

    def word_distances(self, word):
        min_cos_value = -1
        min_index = -1
        index = word2int[word]
        word_vector = self.wih[index, :]
        for i in range(len(words)):
            if (self.CosineDistance(word_vector, self.wih[i, :]) > min_cos_value) and (i != index):
                min_cos_value = self.CosineDistance(word_vector, self.wih[i, :])
                min_index = i
        print("Closest word ", int2word[min_index])

    # ....................Train and Test..........................................


ANN = NeuralNetwork(no_of_in_nodes=len(words),
                    no_of_out_nodes=len(words),
                    no_of_hidden_nodes=60,
                    learning_rate=0.1,
                    epochs=5000
                    )

ANN.train(np.array(df))

ANN.evaluate(df.iloc[0]['input'])
ANN.word_distances("ink")
