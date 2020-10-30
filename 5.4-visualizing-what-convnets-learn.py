from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import load_model

model = load_model('cats_and_dogs_small_2.h5')

model.summary()
type(model.layers)
img_path = './datasets/cats_and_dogs_small/test/cats/cat.1700.jpg'

# 5-25. 단일 개별 이미지 전처리하기
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))  #이미지를 150 * 150으로 불러온다
img_tensor = image.img_to_array(img)  # 이미지 객체를 4D 텐서로 변경
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

# 각 활성화 값을 반환하는 다충 출력 모델 정의

from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]
                 ]  # (input부터 시작해서) 상위 8개층의 출력을 담고 있는 리스트
# print(layer_outputs[0])
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# 입력텐서와 출력텐서를 이용해 모델 인스턴스를 생성

# 예측 모드로 모델 실행하기
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
""" 
(1, 148, 148, 32) 의 넘파이 배열, 특성 맵
 """

# 층의 이름을 그래프 제목으로 사용합니다
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# 특성 맵을 그립니다
for layer_name, layer_activation in zip(layer_names, activations):
    # 특성 맵에 있는 특성의 수
    n_features = layer_activation.shape[-1]

    # 특성 맵의 크기는 (1, size, size, n_features)입니다
    size = layer_activation.shape[1]

    # 활성화 채널을 위한 그리드 크기를 구합니다
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # 각 활성화를 하나의 큰 그리드에 채웁니다
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :,
                                             col * images_per_row + row]
            # 그래프로 나타내기 좋게 특성을 처리합니다
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size:(col + 1) * size,
                         row * size:(row + 1) * size] = channel_image

    # 그리드를 출력합니다
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()

from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K

model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# gradients 함수가 반환하는 텐서 리스트(여기에서는 크기가 1인 리스트)에서 첫 번째 텐서를 추출합니다
grads = K.gradients(loss, model.input)[0]