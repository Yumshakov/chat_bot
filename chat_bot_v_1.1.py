'''Чат бот версии 1.0'''
import nltk
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


with open("BOT_CONFIG_new.json") as f:
    BOT_CONFIG = json.load(f)

x = []
y = []
for intent in BOT_CONFIG['intents']:
    for example in BOT_CONFIG['intents'][intent]['examples']:
        x.append(example)
        y.append(intent)

vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="char_wb")
x_vectorized = vectorizer.fit_transform(x)
clf = RandomForestClassifier()
clf.fit(x_vectorized, y)


def get_intent_by_module(text):

    return clf.predict(vectorizer.transform([text]))[0]


def bot(text):
    intent = get_intent_by_module(text)
    return random.choice(BOT_CONFIG['intents'][intent]['responses'])


text = ''
while text != 'exit':
    text = input()
    print(bot(text))

