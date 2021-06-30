import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  #v1에서 만들어진걸 tensorflow v2에서도 돌아가게 해줌
import numpy as np

detector = dlib.get_frontal_face_detector()  #이미지에서 얼굴 위치찾아주는
#만들어진 모델-학습까지 된- 갖다가 쓰겠다.
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')  #이목구비 위치 찾아주는 모델

img = dlib.load_rgb_image('./imgs/12.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()

img_result = img.copy()
dets = detector(img)   #얼굴 좌표 줌. 얼굴 여러개면 각 얼굴 좌표로 리스트 구성해서 줌
if len(dets) == 0:  #dets가 빈리스트면=이미지에 얼굴 없으면
    print('cannot find faces!')
else :
    fig, ax = plt.subplots(1,figsize=(16, 10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y), w,h,linewidth=2, edgecolor='r',
                            facecolor='none')  #facecolor는 사각형 안 채우는 색
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()

#얼굴의 landmar-눈 양끝, 인중-k찾기
fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)  #img와 detection주면 sp모델이 predict해준게 s
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3,   #radius는 지름
                                edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
plt.show()

#얼굴부분만 따기
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)  #padding은 얼굴 주변에 그만큼 여유있게 따주는
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)
plt.show()

#지금껏 한거 하나의 함수로 만들기
def align_faces(img):
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
    faces = dlib.get_
#Asia_GAN_fork파일에 완성본 있음