import matplotlib.pyplot as plt
import numpy as np
import os  #python내장 패키지. python에 자동으로 붙어있어서 따로 설치할 필요 없. import는 해야함.
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

#GAN
#mnist랑 비슷한 손글씨이미지 만들어서 학습시킬거다
#출력되는 img저장할 CNN_OUT_img디렉토리 만듦
OUT_DIR = './CNN_OUT_img/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)  #OUT_DIR경로 없으면 만들어라
img_shape = (28,28,1)
epoch = 5000  #시간 오래걸린다
batch_size = 128
noise = 100
sample_interval = 100

#build generator
generator_model = Sequential()
generator_model.add(Dense(256*7*7, input_dim=noise))  #잡음 100개 들어오고 256*7*7사이즈로 뻥튀기(256*7*7숫자는 그냥 임의로 정하는값이다)
generator_model.add(Reshape((7,7,256)))
#CNN 1겹
#Conv2DTranspose: upsampling하고 conv하는
generator_model.add(Conv2DTranspose(128, kernel_size=3,  #이미지 128장
            strides=2, padding='same'))  #픽셀 키운다음(strides=2니까 7->14)에 conv
generator_model.add(BatchNormalization())  #데이터 정규화해서 값 계속 커지는것 방지..?
generator_model.add(LeakyReLU(alpha=0.01))
#CNN 2겹
generator_model.add(Conv2DTranspose(64, kernel_size=3,
            strides=1, padding='same'))
generator_model.add(BatchNormalization())  #데이터 정규화해서 값 계속 커지는것 방지
generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(Conv2DTranspose(1, kernel_size=3,  #마지막 출력은 1장만
            strides=2, padding='same'))
generator_model.add(Activation('tanh'))

print(generator_model.summary())

#build discriminator
discriminator_model = Sequential()
discriminator_model.add(Conv2D(32, kernel_size=3, strides=2,
                padding='same', input_shape=img_shape))
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
# discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
# discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))

discriminator_model.add(Flatten())
discriminator_model.add(Dense(1, activation='sigmoid'))
print(discriminator_model.summary())

discriminator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
discriminator_model.trainable = False   #gene학습 시 discri는 학습하면 안됨. discri는 이미 한번 학습했으니까

(X_train, _), (_, _) = mnist.load_data()
print(X_train.shape)    #28X28짜리 이미지 6만장

#mnist에서 가져온 이미지랑 비슷한 이미지 만들도록 GAN모델 만들어서 학습시키기
X_train = X_train / 127.5 - 1   #X_train값이 0일때 -1나오도록, 255일때 1나오도록. 즉 데이터가 -1~1사이 값 나오도록 함
#모델에 넣기위해 reshape
X_train = np.expand_dims(X_train, axis=3)
print(X_train.shape)

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

    z = np.random.normal(0,1,(batch_size, noise))  #평균 0, 표편1인 데이터를 만듦. 잡음 100개짜리 128개
    fake_imgs = generator_model.predict(z)  #gene모델에 잡음 128개 줘서 fake이미지 28X28크기 128장 만듦.

    #real이랑 fake img는 disc모델한테 줘서 맞추게 함
    #CNN에서는 disc가 학습이 잘됨
    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)  #문제랑 정답(real_img니까 정답1)주고 학습  #d_hist_real은 real이미지로 학습한 결과
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)  #문제랑 정답(fake_img니까 정답0)주고 학습  #d_hist_fake도 fake이미지로 학습한 결과

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)  #real이랑 fake로 학습했을때의 loss랑 acc값의 평균낸거
    discriminator_model.trainable = False

    # for i in range(10):   #CNN에서 disc는 학습이 잘되고 gene는 학습이 덜되기때문에 disc가 1번 학습할 때 genen는 5번 학습하도록 함.
    z = np.random.normal(0,1,(batch_size, noise))   #잡음데이터
    #gan모델 학습. gene만 학습하고 disc는 학습안함. disc는 real인지 fake인지 판단만 하도록
    gan_hist = gan_model.train_on_batch(z, real)  #한epoch 학습  #train_on_batch는 z데이터 주고 그거에대한 답인 real주고 딱 한번만 학습하는.
    #gan모델은 출력이 1이 되게 학습-gene만 학습-해야한다. gene는 dis에서 gene가 만든 이미지를 real(1)이라고 판단하게 만들어야하니까.

    #이미지 확인하기 위한 과정. 학습이랑은 상관없다.
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
        path = os.path.join(OUT_DIR, 'img-{}'.format(itr))
        plt.savefig(path)
        plt.close()
#뭔가 잘못된 결과가 나옴. 엉망진창이구나~ 강사님 코드 보렴
#코드는 같은데 왜 결과가 잘못된것이지 알 수 없넴