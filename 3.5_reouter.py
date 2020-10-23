from keras.datasets import reuters
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Keras에서 데이터 불러오기
# (train / validation) train
(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=10000)

# 총 train_data의 개수 = 8982
# 총 test_data의 개수 = 2246
# 형식 : 'python list'의 numpy array


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # sequence = python list
        results[i, sequence] = 1.
    return results


# 훈련 데이터 벡터 변환
x_train = vectorize_sequences(train_data)
# 테스트 데이터 벡터 변환
x_test = vectorize_sequences(test_data)

# 범주형 인코딩 (categorical encoding)
# label 데이터를 one hot encoding 방식으로 벡터로 변경
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# 모델 구성
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# model을 최적화하고 그 결과를 history에 저장한다
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

# train loss & validation loss
epochs = range(1, len(loss) + 1)  # 1부터 len(loss)만큼의 range객체 반환

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#train accuracy & validation accuracy

plt.clf()  # plot 그래프를 초기화한다

plt.plot(epochs, acc, 'bo', label='Traiining Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.plot()

# test데이터(=새로운 데이터)를 통해 모델의 성능 예측하기
predictions = model.predict(x_test)

predictions.shape