from sklearn.utils import shuffle
import pandas as pd
from konlpy.tag import Okt
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# 1. data preprocessing ----------------

X_data = []
Y_data = []
X_test = []
Y_test = []
train_data = pd.read_csv("dataset.csv", encoding='utf-8',sep='|')
test_data = pd.read_csv("testset.csv", encoding='utf-8',sep='|')
train_data = train_data.dropna()
train_data = shuffle(train_data)
k=0
stopwords = ['의','가','이','은','들','는','과','도','를','으로','에','와']

mc = Okt() #okt 형태소 분석기를 사용하였습니다.

import Preprocess as pr
p = pr.Preprocess(mc)

for sentence in train_data['Sentence']:
    X_data = p.work(sentence,stopwords,X_data)
for sentence in test_data['Sentence']:
    X_test = p.work(sentence,stopwords,X_test)

Y_data = p.labeling(train_data, Y_data)
Y_test = p.labeling(test_data, Y_test)

maxlen = 50
vocab_size = 9150
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV',filters='')
tokenizer.fit_on_texts(X_data)
X_data = tokenizer.texts_to_sequences(X_data)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = []
Y_train = []
X_val = []
Y_val = []
X_train = np.array(X_data[:24400])
Y_train = np.array(Y_data[:24400])
X_val = np.array(X_data[24400:])
Y_val = np.array(Y_data[24400:])
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_train = pad_sequences(X_train, maxlen=maxlen) #문장길이 30으로 제한
X_val = pad_sequences(X_val, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

#------------------------ends

from Transformer import TransformerBlock1
from Transformer import TransformerBlock2
from Transformer import TransformerBlock3
from Transformer import TokenAndPositionEmbedding
from tensorflow.keras import layers
from tensorflow import keras

# 2. train --------------------------------

embed_dim = 32  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block1 = TransformerBlock1(embed_dim, num_heads, ff_dim)
transformer_block2 = TransformerBlock2(embed_dim, num_heads, ff_dim)
transformer_block3 = TransformerBlock3(embed_dim, num_heads, ff_dim)
x = transformer_block1(x)
x = transformer_block2(x)
x = transformer_block3(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(25, activation="relu")(x)
#x = layers.Dropout(0.1)(x)
outputs = layers.Dense(5, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

model.compile("adam", keras.losses.CategoricalCrossentropy(label_smoothing=0.06), metrics=["accuracy"])

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('Project_transformer_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history = model.fit(X_train, Y_train, batch_size=32, epochs=15, validation_data=(X_val, Y_val), callbacks=[es, mc],
                    verbose=1)
print("--------------------model saved!------------------")

# 3. test --------------------------------
model.load_weights('Project_transformer_model.h5')
tp = model.predict(X_test)
result = []
for temp in tp:
    temp = np.argmax(temp)
    result.append(temp)

k=0
for i in range(len(result)):
    if result[i] == np.argmax(Y_test[i]):
        k+=1

print("정확도는", k/len(Y_test),"입니다")

