#!/usr/bin/env python
# coding: utf-8


# Интернет-магазин «Викишоп» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. 
# 
# Обучите модель классифицировать комментарии на позитивные и негативные. В вашем распоряжении набор данных с разметкой о токсичности правок.
# 
# Постройте модель со значением метрики качества *F1* не меньше 0.75. 
# Для выполнения проекта применять *BERT* необязательно, но вы можете попробовать.

# # 1. Подготовка

import pandas as pd
data = pd.read_csv('/datasets/toxic_comments.csv')
data.head()
data.info()


# текст английский, пропусков нет. посмотрю, как выглядят тексты

data['text'][0]
test_text = data['text'][1]
test_text


# сохраню в переменной, чтобы потом тестировать регулярное выражение на нем. предполагаю что D'aww это какая-то эмоция, ее надо оставить. а вот I'm не знаю. отрезала бы 'm

# есть цифры, символы переноса строк; подозреваю, что стопслова тоже имеются. "Прежде чем извлечь признаки из текста, упростим его." лемматизация. а когда решают делать лемматизацию или токенизацию? есть какое-то условие, что при этом делай это , при этом другой.  в части bert было про токенизацию.

# pymorphy2 (англ. python morphology, «морфология для Python»), **
# UDPipe (англ. universal dependencies pipeline, «конвейер для построения общих зависимостей»),
# pymystem3.

from pymystem3 import Mystem
import re
m = Mystem()
def lemmatize(text):
    a = m.lemmatize(text)
    a = ''.join(a)
    return a
def clear_text(text):
    text = re.sub(r"[^a-zA-z]+", ' ', text)
    text = text.split()
    text = ' '.join(text)
    return text
clear_text(test_text)
data['lemma_text'] = data['text'].apply(lemmatize)
data.head()
data['lemma_text'] = data['lemma_text'].apply(clear_text)
data.head()
data['lemma_text'] = data['lemma_text'].str.lower()
data.head()
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords as nltk_stopwords
nltk.download('stopwords')
stopwords = set(nltk_stopwords.words('english'))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
features = data['lemma_text']
target = data['toxic']
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=12345)
from sklearn.feature_extraction.text import TfidfVectorizer
count_tf_idf = TfidfVectorizer(stop_words=stopwords)
tf_idf = count_tf_idf.fit_transform(features_train)
tf_idf.shape
# # 2. Обучение
#классы не сбалансированы, добавила еще class_weight. а можно было бы здесь применять upsampling/downsampling?
model = LogisticRegression(random_state=12345, class_weight = 'balanced')
model.fit(tf_idf, target_train)
from sklearn.metrics import f1_score
tf_idf_test = count_tf_idf.transform(features_test)
predicted = model.predict(tf_idf_test)
f1_score(target_test, predicted)


# нужная метрика достигнута. попробую другие модели

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=12345, n_estimators=5)
model.fit(tf_idf, target_train)
#tf_idf_test = count_tf_idf.transform(features_test)
predicted = model.predict(tf_idf_test)
f1_score(target_test, predicted)


# значение очень маленькое и долго считает. пробовала больше estimators еще дольше. мне раньше казалось, что лес обычно лучше предсказывает.. или для леса прлучается слишком много ветвей?

# bert обрабатывался очень долго, поэтому решила попробовать https://habr.com/ru/post/498144/
import numpy as np
import pandas as pd
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('/datasets/toxic_comments.csv', header=None)
df.head()
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
# Загрузка предобученной модели/токенизатора 
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
#tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
#Token indices sequence length is longer than the specified maximum sequence length for this model (715 > 512). Running this sequence through the model will result in indexing errors


# https://github.com/huggingface/transformers/issues/1791
# предлагается или обрезать или попробовать Transformer-XL or XLNet. я попробую первое
count = 0
for i in df[0]:
    if len(i) > 512:
        count+= 1
count


# много таких последовательностей
df[0] = df[0].apply(lambda x: x[:min(512, len(x))])
tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
np.array(padded).shape
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape


# input_ids = torch.tensor(padded)  
# attention_mask = torch.tensor(attention_mask)
# 
# with torch.no_grad():
#     last_hidden_states = model(input_ids, attention_mask=attention_mask)
# RuntimeError: [enforce fail at CPUAllocator.cpp:64] . DefaultCPUAllocator: can't allocate memory: you tried to allocate 241671155712 bytes. Error code 12 (Cannot allocate memory)

# не хватило 241 gb оперативной памяти
# нужная метрика 0.75 была достигнута при использовании TfidfVectorizer и логистической регрессии с учетом дисбалана классов. дерево показало низкие F1. что можно было бы еще попробовать? если бы баланс не помог?
# попыталась сделать предобработку с DistilBERT 
