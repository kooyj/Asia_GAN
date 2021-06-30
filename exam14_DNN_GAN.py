import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

#GAN
#mnist랑 비슷한 손글씨이미지 만들어서 학습시킬거다
#출력되는 img저장할 OUT_img디렉토리 만듦
OUT_DIR = './OUT_img/'
img_shape = (28,28,1)
epoch = 100000
batch_size = 128
noise = 100
sample_interval = 100


(X_train, _), (_, _) = mnist.load_data()
print(X_train.shape)    #28X28짜리 이미지 6만장

#mnist에서 가져온 이미지랑 비슷한 이미지 만들도록 GAN모델 만들어서 학습시키기
X_train = X_train / 127.5 - 1   #X_train값이 0일때 -1나오도록, 255일때 1나오도록. 즉 데이터가 -1~1사이 값 나오도록 함
#모델에 넣기위해 reshape
X_train = np.expand_dims(X_train, axis=3)  #차원을 하나 늘리는.  #reshape랑 같은 기능 다른 함수
print(X_train.shape)   #(60000,28,28,1)

#모델 2개 만듦(g와 d)
#build generator
generator_model = Sequential()
generator_model.add(Dense(128, input_dim=noise))  #랜덤하게 만든 잡음 100개 받게됨
generator_model.add(LeakyReLU(alpha=0.01))  #activation함수. 데이터에 마이너스값이 있어서 LeakyReLU사용
generator_model.add(Dense(784, activation='tanh'))  #tanh은 다른 함수보다 값이 크게나와서 발산할 가능성 큼. GAN의 generator은 tanh많이 쓰긴함
generator_model.add(Reshape(img_shape))  #최종 출력 이미지
print(generator_model.summary())

#build discriminator
discriminator_model = Sequential()
discriminator_model.add(Flatten(input_shape=img_shape))   #한줄로 reshape
discriminator_model.add(Dense(128))
discriminator_model.add(LeakyReLU(alpha=0.01))  #alpha로 민감도 조절 alpha는 데이터가 마이너스일때의 기울기값.
#LeakyReLU는 alpha값 줘야해서 다른 activation함수처럼 주지 않고 위처럼 따로 add한다.
discriminator_model.add(Dense(1, activation='sigmoid'))  #출력층. 진품인지 가품인지 판단하는 출력 한개
print(discriminator_model.summary())

discriminator_model.compile(loss='binary_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
discriminator_model.trainable = False   #discriminator은 학습 안하는. forward만하고 backward안함

#build GAN
#모델 이어붙
gan_model = Sequential()
gan_model.add(generator_model)  #첫번째 레이어
gan_model.add(discriminator_model)
print(gan_model.summary())
gan_model.compile(loss='binary_crossentropy', optimizer='adam')

#학습 시키기

#라벨 달기
real = np.ones((batch_size, 1))   #모든값이 1인 행렬만듦  ((행렬크기))
print(real)
fake = np.zeros((batch_size, 1)) #모든값이 0인 행렬 만듦  ((행렬크기))
print(fake)

for itr in range(epoch):  #epoch수 만큼 discr랑 gan을 따로 번갈아가면서 학습시킴
    idx = np.random.randint(0,X_train.shape[0], batch_size)  #0~59999 사이 int값 랜덤하게 뽑음. batch_size개수만큼.
    real_imgs = X_train[idx]   #128개 이미지 X_train에서 뽑

    z = np.random.normal(0,1,(batch_size, noise))  #평균 0, 표편1인 정규분포 따르는 데이터를 만듦. 잡음 100개짜리 128개-이게 한epoch
    fake_imgs = generator_model.predict(z)  #gene모델에 잡음 128개 줘서 fake이미지 28X28크기 128장 만듦.

    #real이랑 fake img는 disc모델한테 줘서 맞추게 함
    #discr모델 학습. fake랑 real따로 학습. 한epoch 학습
    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)  #문제랑 정답(real_img니까 정답1)주고 학습  #d_hist_real은 real이미지로 학습한 결과
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)  #문제랑 정답(fake_img니까 정답0)주고 학습  #d_hist_fake도 fake이미지로 학습한 결과

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)  #real이랑 fake로 학습했을때의 loss랑 acc값의 평균낸거
    discriminator_model.trainable = False   #gene학습 시 discri는 학습하면 안됨. discri는 이미 한번 학습했으니까

    z = np.random.normal(0,1,(batch_size, noise))   #잡음데이터
    #gan모델 학습. gene만 학습하고 disc는 학습안함. disc는 real인지 fake인지 판단만 하도록
    gan_hist = gan_model.train_on_batch(z, real)  #한epoch 학습  #train_on_batch는 z데이터 주고 그거에대한 답인 real주고 딱 한번만 학습하는.
    #gan모델은 출력이 1이 되게 학습-gene만 학습-해야한다. gene는 dis에서 gene가 만든 이미지를 real(1)이라고 판단하게 만들어야하니까.

    if itr%sample_interval == 0:  #100epoch마다 한번씩 이미지 출력
        print('%d [D loss: %f, acc: %.2f%%] [G loss: %f]'%(itr, d_loss, d_acc*100, gan_hist))
        row = col = 4
        z = np.random.normal(0,1,(row*col, noise))
        fake_imgs = generator_model.predict((z))
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(row, col), sharey=True, sharex=True)  #가로4개 새로4개 총 16개 이미지 그리겠다
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].axis('off')  #x,y축 눈금 지우기
                cnt += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(itr+1))
        plt.savefig(path)
        plt.close()