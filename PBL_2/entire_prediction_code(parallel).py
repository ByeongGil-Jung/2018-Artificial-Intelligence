'''
[ Multicore Version ]

Usage

Execution arguments: <python_name>.py <training_data> <test_data>
Training data must be patterns, label pair.
so,

Execution arguments: <python_name>.py <training_data_pattern> <training_data_label> <test_data>

(It must be run on 'Python', not 'Anaconda')
'''
import os
import sys
import struct
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


class StochasticClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, eta=0.01, n_iter=1000, b_size=1, lamb=0.001, random_state=None, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.b_size = b_size  # batch size
        self.lamb = lamb  # lambda == c
        self.random_state = random_state
        self.shuffle = shuffle
        self.labels = None
        self.w_ = {}
        self.w_bar_ = {}
        self.b_ = {}  # 벡터
        self.b_bar_ = {}

    def fit(self, X, y=None):
        self.labels = np.unique(y)  # list will be array (0, ....9)

        Parallel(n_jobs=-1, require='sharedmem')(delayed(self.fit_inner)(label, X, y) for label in self.labels)

        return self

    def fit_inner(self, label, X, y):
        self._initialize_weights(label, X.shape[1])
        self.b_[label] = 0
        ova_y = []

        for xi, yi in zip(X, y):  # make new label for OvA.
            if label == yi:
                ova_y.append(1)
            else:
                ova_y.append(-1)

        n_list = []

        for iteration in range(self.n_iter):
            samples = []

            for i in range(self.b_size):
                if not n_list:  # list empty
                    n_list = self.shuffle_index(len(y))
                n = n_list.pop()
                samples.append((X[n], ova_y[n]))

            self._update_weights(label, iteration, samples, X.shape[1])

    def shuffle_index(self, n):
        n_list = [i for i in range(0, n)]

        if self.shuffle:
            np.random.shuffle(n_list)  # just shuffle index

        return n_list

    def _initialize_weights(self, label, m):  # 작은 값으로 weight 값 초기화
        self.rgen = np.random.RandomState(self.random_state)
        self.w_[label] = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)

    def _update_weights(self, label, iteration, samples, m):
        w = [0 for num in range(m)]
        b = 0

        for xi, y in samples:  # samples[0] 데이터, samples[1] 라벨
            output = self.net_input_for_learning(label, xi)
            if y * output <= 1:
                w += self.eta * xi.dot(y)
                b += self.eta * y

        self.w_[label][1:] += np.divide(w, self.b_size) - self.lamb * self.w_[label][1:]
        self.b_[label] += b / self.b_size

        if iteration == 0:
            self.w_bar_[label] = copy.copy(self.w_[label])
            self.b_bar_[label] = copy.copy(self.b_[label])
        else:
            self.w_bar_[label][1:] = np.multiply(iteration/(iteration+1), self.w_bar_[label][1:]) + np.multiply(1/(iteration+1), self.w_[label][1:])
            self.b_bar_[label] = iteration/(iteration+1)*self.b_bar_[label] + 1/(iteration+1)*self.b_[label]

    def net_input_for_learning(self, label, X):
        return np.dot(self.w_[label][1:], X) + self.b_[label]

    def net_input_for_predict(self, label, X):
        return np.dot(self.w_bar_[label][1:], X) + self.b_bar_[label]

    def predict(self, X):
        predictions = []

        for xi in X:
            predict_output = {}

            for label in self.labels:
                predict_output[label] = self.net_input_for_predict(label, xi)

            output = None

            for item in predict_output:
                if not output:
                    output = item
                else:
                    if predict_output[output] < predict_output[item]:
                        output = item
            predictions.append(output)

        return predictions


def read(pattern_path, label_path=None):
    if label_path:
        with open(label_path, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8)) ## 바이너리 데이터를 추출하면서 >||로 구분하는 듯함. flbl.read라는 퍼버에 저장
            lbl = np.fromfile(flbl, dtype=np.int8) #np 형식으로 버퍼의 내용을 저장.

    with open(pattern_path, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows, cols)

    if label_path:
        get_img = lambda idx: (lbl[idx], img[idx])
    else:
        get_img = lambda idx: (img[idx])

    # Create an iterator which returns each image in turn
    for i in range(num):
        yield get_img(i)


import random
import copy


"""
Features
"""
import random
import copy

def make_input_image_sharp(image_input, value): # 리턴 값은 그냥 펴진 스트링임.
    new_image = []
    image = np.reshape(image_input, 28*28)
    for pixel in image:
        if pixel >= value:
            new_image.append(pixel)
        else:
            new_image.append(0)
    return new_image

def draw_to_end(image_input):
    image = image_input
    new_image = []
    for row in image:
        new_row = list(row)
        threshold = 128
        start_point = 0
        last_point = 0
        for i in range(len(row)):
            if row[i] > threshold:
                start_point = i
                break
        for i in reversed(range(len(row))):
            if row[i] > threshold:
                last_point = i
                break
        for i in range(start_point, last_point):
            new_row[i] = 255
        new_image += new_row
    return new_image

def draw_to_end_by_col(image_input):
    image = copy.copy(image_input)
    col_image = [[] for i in range(28)]
    for row in image:
        for item_n in range(len(row)):
            col_image[item_n].append(row[item_n])
    image = col_image
    new_image = []
    for row in image:
        new_row = list(row)
        threshold = 128
        start_point = 0
        last_point = 0
        for i in range(len(row)):
            if row[i] > threshold:
                start_point = i
                break
        for i in reversed(range(len(row))):
            if row[i] > threshold:
                last_point = i
                break
        for i in range(start_point, last_point):
            new_row[i] = 255
        new_image += new_row
    return new_image


def new_feature(image_input, length): #자기랑 하는 거 안 뺌. 곱하기 연산
    new_ftr = []
    for i in range(length):
        for j in range(i, length):
            new_ftr.append(image_input[i] * image_input[j])
    return new_ftr


def new_feature2(image_input, length): #자기랑 하는 거 안 뺌. 더하기 연산
    new_ftr = []
    for i in range(length):
        for j in range(i, length):
            new_ftr.append(image_input[i] + image_input[j])
    return new_ftr


#####

def add_new_features(original_features, new_features):
    # 2차원 array 인 경우
    if isinstance(new_features[0], list):
        for i in new_features:
            for j in range(len(i)):
                original_features.append(i[j])
    # 1차원 array 인 경우
    elif isinstance(new_features, list):
        for i in range(len(new_features)):
            original_features.append(new_features[i])

            
def get_four_direction_features(pixel, density=0):
    row_len = len(pixel)
    col_len = len(pixel[0])
    out = {x: np.zeros(row_len) for x in ("N", "W", "S", "E")}  # 북서남동 각각 28 개씩 있음
    transform_ratio = 255 * 1/28
    out_features = list()
    
    # North
    for i in range(col_len):
        for j in range(row_len):
            if pixel[j, i] > density:
                result_ = j
                out["N"][i] = result_ * transform_ratio
                break
                
    # West
    for i in range(row_len):
        for j in range(col_len):
            if pixel[i, j] > density:
                result_ = j
                out["W"][i] = result_ * transform_ratio
                break
                
    # South
    for i in range(col_len):
        for j in range(row_len):
            if pixel[(row_len - 1) - j, i] > density:
                result_ = (row_len - 1) - j
                out["S"][i] = result_ * transform_ratio
                break
            
    # East
    for i in range(row_len):
        for j in range(col_len):
            if pixel[i, (col_len - 1) - j] > density:
                result_ = (col_len - 1) - j
                out["E"][i] = result_ * transform_ratio
                break
    
    # processing data
    for f in out.values():
        out_features.append(f.tolist())
    
    return out_features


def get_diagonal_direction_features(pixel, density=0):
    row_len = len(pixel)
    col_len = len(pixel[0])
    out = [[0 for _ in range(2 * row_len - 1)],  # 2
           [0 for _ in range(2 * row_len - 1)],  # 3
           [0 for _ in range(2 * row_len - 1)],  # 4
           [0 for _ in range(2 * row_len - 1)]]  # 1
    
    # transform_ratio = 255 * 1/28
    
    # 2사분면
    for i in range(row_len - 1):
        count = 1
        for j in range(i + 1):
            if pixel[(row_len - 1) - i + j, j] > density:
                out[0][i] = count * (255 * 1 / (i + 1))
                break
            count += 1
    count = 1
    for i in range(row_len):
        if pixel[i, i] > density:
            out[0][row_len - 1] = count * (255 * 1 / (i + 1))
            break
        count += 1
    count = 0
    for i in range(row_len - 1):
        count = 1
        for j in range(i + 1):
            if pixel[j, (col_len - 1) - i + j] > density:
                out[0][row_len + i] = count * (255 * 1 / (i + 1))
                break
            count += 1
                
    # 3사분면
    for i in range(row_len - 1):
        count = 1
        for j in range(i + 1):
            if pixel[i - j, j] > density:
                out[1][i] = count * (255 * 1 / (i + 1))
                break
            count += 1
    count = 1
    for i in range(row_len):
        if pixel[(row_len - 1) - i, i] > density:
            out[1][row_len - 1] = count * (255 * 1 / (i + 1))
            break
        count += 1
    count = 0
    for i in range(row_len - 1):
        count = 1
        for j in range(i + 1):
            if pixel[(row_len - 1) - j, (col_len - 1) - i + j] > density:
                out[1][row_len + i] = count * (255 * 1 / (i + 1))
                break
            count += 1
                
    # 4사분면
    for i in range(col_len - 1):
        count = 1
        for j in range(i + 1):
            if pixel[i - j, (col_len - 1) - j] > density:
                out[2][i] = count * (255 * 1 / (i + 1))
                break
            count += 1
    count = 1
    for i in range(row_len):
        if pixel[(row_len - 1) - i, (col_len - 1) - i] > density:
            out[2][row_len - 1] = count * (255 * 1 / (i + 1))
            break
        count += 1
    count = 0
    for i in range(col_len - 1):
        count = 1
        for j in range(i + 1):
            if pixel[(row_len - 1) - j, i - j] > density:
                out[2][row_len + i] = count * (255 * 1 / (i + 1))
                break
            count += 1
            
    # 1사분면
    for i in range(col_len - 1):
        count = 1
        for j in range(i + 1):
            if pixel[j, i - j] > density:
                out[3][i] = count * (255 * 1 / (i + 1))
                break
            count += 1
    count = 1
    for i in range(row_len):
        if pixel[i, (col_len - 1) - i] > density:
            out[3][row_len - 1] = count * (255 * 1 / (i + 1))
            break
        count += 1
    count = 0
    for i in range(col_len - 1):
        count = 1
        for j in range(i + 1):
            if pixel[(row_len - 1) - i + j, (col_len - 1) - j] > density:
                out[3][row_len + i] = count * (255 * 1 / (i + 1))
                break
            count += 1
    
    return out


def get_length_features(pixel, density=0):
    row_len = len(pixel)
    col_len = len(pixel[0])
    out = {x: np.zeros(row_len) for x in ("N", "W", "S", "E")}  # 북서남동 각각 28 개씩 있음
    transform_ratio = 255 * 1/28 * 2
    out_features = list()

    # North
    for i in range(col_len):
        count = 0
        for j in range(row_len):
            if pixel[j, i] > density:
                count += 1
            if count > 0 and pixel[j, i] <= density:
                break
        out["N"][i] = count * transform_ratio

    # West
    for i in range(row_len):
        count = 0
        for j in range(col_len):
            if pixel[i, j] > density:
                count += 1
            if count > 0 and pixel[i, j] <= density:
                break
        out["W"][i] = count * transform_ratio

    # South
    for i in range(col_len):
        count = 0
        for j in range(row_len):
            if pixel[(row_len - 1) - j, i] > density:
                count += 1
            if count > 0 and pixel[(row_len - 1) - j, i] <= density:
                break
        out["S"][i] = count * transform_ratio

    # East
    for i in range(row_len):
        count = 0
        for j in range(col_len):
            if pixel[i, (col_len - 1) - j] > density:
                count += 1
            if count > 0 and pixel[i, (col_len - 1) - j] <= density:
                break
        out["E"][i] = count * transform_ratio

    # processing data
    for f in out.values():
        out_features.append(f.tolist())

    return out_features


def get_diagonal_length_features(pixel, density=0):
    row_len = len(pixel)
    col_len = len(pixel[0])
    out = [[0 for _ in range(2 * row_len - 1)],  # 2
           [0 for _ in range(2 * row_len - 1)],  # 3
           [0 for _ in range(2 * row_len - 1)],  # 4
           [0 for _ in range(2 * row_len - 1)]]  # 1
    
    # transform_ratio = 255 * 1/28
    
    # 2사분면
    for i in range(row_len - 1):
        count = 0
        for j in range(i + 1):
            if pixel[(row_len - 1) - i + j, j] > density:
                count += 1
            if count > 0 and pixel[(row_len - 1) - i + j, j] <= density:
                break
        out[0][i] = count * (255 * 1 / (i + 1))
        
    count = 0
    for i in range(row_len):
        if pixel[i, i] > density:
            count += 1
        if count > 0 and pixel[i, i] <= density:
            break
    out[0][row_len - 1] = count * (255 * 1 / (i + 1))
    count = 0
    
    for i in range(row_len - 1):
        count = 0
        for j in range(i + 1):
            if pixel[j, (col_len - 1) - i + j] > density:
                count += 1
            if count > 0 and pixel[j, (col_len - 1) - i + j] <= density:
                break
        out[0][row_len + i] = count * (255 * 1 / (i + 1))
                
    # 3사분면
    for i in range(row_len - 1):
        count = 0
        for j in range(i + 1):
            if pixel[i - j, j] > density:
                count += 1
            if count > 0 and pixel[i - j, j] <= density:
                break
        out[1][i] = count * (255 * 1 / (i + 1))
        
    count = 0
    for i in range(row_len):
        if pixel[(row_len - 1) - i, i] > density:
            count += 1
        if count > 0 and pixel[(row_len - 1) - i, i] <= density:
            break
    out[1][row_len - 1] = count * (255 * 1 / (i + 1))
    count = 0
    
    for i in range(row_len - 1):
        count = 0
        for j in range(i + 1):
            if pixel[(row_len - 1) - j, (col_len - 1) - i + j] > density:
                count += 1
            if count > 0 and pixel[(row_len - 1) - j, (col_len - 1) - i + j] <= density:
                break
        out[1][row_len + i] = count * (255 * 1 / (i + 1))
                
    # 4사분면
    for i in range(col_len - 1):
        count = 0
        for j in range(i + 1):
            if pixel[i - j, (col_len - 1) - j] > density:
                count += 1
            if count > 0 and pixel[i - j, (col_len - 1) - j] <= density:
                break
        out[2][i] = count * (255 * 1 / (i + 1))
            
    count = 0
    for i in range(row_len):
        if pixel[(row_len - 1) - i, (col_len - 1) - i] > density:
            count += 1
        if count > 0 and pixel[(row_len - 1) - i, (col_len - 1) - i] <= density:
            break
    out[2][row_len - 1] = count * (255 * 1 / (i + 1))
    count = 0
    
    for i in range(col_len - 1):
        count = 0
        for j in range(i + 1):
            if pixel[(row_len - 1) - j, i - j] > density:
                count += 1
            if count > 0 and pixel[(row_len - 1) - j, i - j] <= density:
                break
        out[2][row_len + i] = count * (255 * 1 / (i + 1))
            
    # 1사분면
    for i in range(col_len - 1):
        count = 0
        for j in range(i + 1):
            if pixel[j, i - j] > density:
                count += 1
            if count > 0 and pixel[j, i - j] <= density:
                break
        out[3][i] = count * (255 * 1 / (i + 1))
            
    count = 0
    for i in range(row_len):
        if pixel[i, (col_len - 1) - i] > density:
            count += 1
        if count > 0 and pixel[i, (col_len - 1) - i] <= density:
            break
    out[3][row_len - 1] = count * (255 * 1 / (i + 1))
    count = 0
    
    for i in range(col_len - 1):
        count = 0
        for j in range(i + 1):
            if pixel[(row_len - 1) - i + j, (col_len - 1) - j] > density:
                count += 1
            if count > 0 and pixel[(row_len - 1) - i + j, (col_len - 1) - j] <= density:
                break
        out[3][row_len + i] = count * (255 * 1 / (i + 1))
    
    return out

	
filter = np.load('./filter.npy')
def extract_feature(image_input):
    image = copy.copy(image_input)
    image = image.flatten()
    image = np.asarray(image, dtype='float')
    new_feature=[]
    for number in range(0,10):
        new_feature .append(np.dot(image,filter[number]))
    
    return new_feature

	
def get_two_connected_features(pixel):
    row_len = len(pixel)
    col_len = len(pixel[0])
    out = []
    transform_ratio = 255 ** 2
    margin = 4
    
    for i in range(margin, row_len - margin + 1):
        for j in range(margin, col_len - margin + 1):
            out.append([pixel[i, j] * pixel[i - 1, j] / transform_ratio, pixel[i, j] * pixel[i - 1, j + 1] / transform_ratio, pixel[i, j] * pixel[i + 1, j + 1] / transform_ratio])
    
    return out



"""
Process
"""
train_pattern_path = sys.argv[1] # 학습 패턴
train_label_path = sys.argv[2] # 학습 라벨
test_pattern_path = sys.argv[3] # 결과를 낼 패턴
#test_label_path = sys.argv[4]

train_set = list(read(train_pattern_path, train_label_path))
test_set = list(read(test_pattern_path))

labels = {}
labels['train'] = []
labels['test'] = []

images = {}
images['train'] = []
images['test'] = []

for i in range(len(train_set)):
    images['train'].append(list(np.reshape(train_set[i][1], 28*28)))
    labels['train'].append(train_set[i][0])

test = {}
test['images'] = []
test['labels'] = []

for i in range(len(test_set)):
    images['test'].append(list(np.reshape(test_set[i], 28*28)))
    labels['test'].append(test_set[i])
    

###

n_feature = 30

end_to_train = [[] for i in range(len(images['train']))]

for i in range(len(images['train'])):
    end_to_train[i] += draw_to_end(np.reshape(images['train'][i], (28, 28)))
    
end_to_test = [[] for i in range(len(images['test']))]

for i in range(len(images['test'])):
    end_to_test[i] += draw_to_end(np.reshape(images['test'][i], (28, 28)))
    
pca2 = PCA(n_components=n_feature)
X_train_pca2 = pca2.fit_transform(end_to_train)
X_test_pca2 = pca2.transform(end_to_test)

end_to_train_by_col = [[] for i in range(len(images['train']))]

for i in range(len(images['train'])):
    end_to_train_by_col[i] += draw_to_end_by_col(np.reshape(images['train'][i], (28, 28)))
    
end_to_test_by_col = [[] for i in range(len(images['test']))]

for i in range(len(images['test'])):
    end_to_test_by_col[i] += draw_to_end_by_col(np.reshape(images['test'][i], (28, 28)))
    
pca3 = PCA(n_components=n_feature)
X_train_pca3 = pca3.fit_transform(end_to_train_by_col)
X_test_pca3 = pca3.transform(end_to_test_by_col)

result_images = {}
result_images['train'] = [[] for i in range(len(images['train']))]
result_images['test'] = [[] for i in range(len(images['test']))]

###

temp_images = {}
sc = StandardScaler()

n_feature_f = 50

pca1 = PCA(n_components=n_feature_f)
X_train_pca1 = pca1.fit_transform(images['train'])
X_test_pca1 = pca1.transform(images['test'])

temp_images['train'] = [[] for i in range(len(images['train']))]

for i in range(len(X_train_pca1)):
    temp_images['train'][i] += new_feature(X_train_pca1[i], n_feature_f)
    
temp_images['test'] = [[] for i in range(len(images['test']))]

for i in range(len(X_test_pca1)):
    temp_images['test'][i] += new_feature(X_test_pca1[i], n_feature_f)
    
sc.fit(temp_images['train'])
temp_images['train'] = sc.transform(temp_images['train'])
temp_images['test'] = sc.transform(temp_images['test'])


for i in range(len(images['train'])):
    result_images['train'][i] += list(temp_images['train'][i])
    
for i in range(len(images['test'])):
    result_images['test'][i] += list(temp_images['test'][i])


# type(temp_images['train'][i])


temp_images['train'] = [[] for i in range(len(images['train']))]

for i in range(len(X_train_pca1)):
    temp_images['train'][i] += new_feature2(X_train_pca1[i], n_feature_f)
    
temp_images['test'] = [[] for i in range(len(images['test']))]

for i in range(len(X_test_pca1)):
    temp_images['test'][i] += new_feature2(X_test_pca1[i], n_feature_f)
    
sc.fit(temp_images['train'])
temp_images['train'] = sc.transform(temp_images['train'])
temp_images['test'] = sc.transform(temp_images['test'])


for i in range(len(images['train'])):
    result_images['train'][i] += list(temp_images['train'][i])
    
for i in range(len(images['test'])):
    result_images['test'][i] += list(temp_images['test'][i])
    
# end-to 곱하기 연산
temp_images['train'] = [[] for i in range(len(images['train']))]

for i in range(len(X_train_pca2)):
    temp_images['train'][i] += new_feature(X_train_pca2[i], n_feature)
    
temp_images['test'] = [[] for i in range(len(images['test']))]

for i in range(len(X_test_pca2)):
    temp_images['test'][i] += new_feature(X_test_pca2[i], n_feature)
    
sc.fit(temp_images['train'])
temp_images['train'] = sc.transform(temp_images['train'])
temp_images['test'] = sc.transform(temp_images['test'])

for i in range(len(images['train'])):
    result_images['train'][i] += list(temp_images['train'][i])
    
for i in range(len(images['test'])):
    result_images['test'][i] += list(temp_images['test'][i])
    
# end-to 더하기 연산
temp_images['train'] = [[] for i in range(len(images['train']))]

for i in range(len(X_train_pca2)):
    temp_images['train'][i] += new_feature2(X_train_pca2[i], n_feature)
    
temp_images['test'] = [[] for i in range(len(images['test']))]

for i in range(len(X_test_pca2)):
    temp_images['test'][i] += new_feature2(X_test_pca2[i], n_feature)
    
sc.fit(temp_images['train'])
temp_images['train'] = sc.transform(temp_images['train'])
temp_images['test'] = sc.transform(temp_images['test'])

for i in range(len(images['train'])):
    result_images['train'][i] += list(temp_images['train'][i])
    
for i in range(len(images['test'])):
    result_images['test'][i] += list(temp_images['test'][i])
    
# end-to_by_col 곱하기 연산
temp_images['train'] = [[] for i in range(len(images['train']))]

for i in range(len(X_train_pca3)):
    temp_images['train'][i] += new_feature(X_train_pca3[i], n_feature)
    
temp_images['test'] = [[] for i in range(len(images['test']))]

for i in range(len(X_test_pca3)):
    temp_images['test'][i] += new_feature(X_test_pca3[i], n_feature)
    
sc.fit(temp_images['train'])
temp_images['train'] = sc.transform(temp_images['train'])
temp_images['test'] = sc.transform(temp_images['test'])

for i in range(len(images['train'])):
    result_images['train'][i] += list(temp_images['train'][i])
    
for i in range(len(images['test'])):
    result_images['test'][i] += list(temp_images['test'][i])

# end-to_by_col 더하기 연산
temp_images['train'] = [[] for i in range(len(images['train']))]

for i in range(len(X_train_pca3)):
    temp_images['train'][i] += new_feature2(X_train_pca3[i], n_feature)
    
temp_images['test'] = [[] for i in range(len(images['test']))]

for i in range(len(X_test_pca3)):
    temp_images['test'][i] += new_feature2(X_test_pca3[i], n_feature)
    
sc.fit(temp_images['train'])
temp_images['train'] = sc.transform(temp_images['train'])
temp_images['test'] = sc.transform(temp_images['test'])

for i in range(len(images['train'])):
    result_images['train'][i] += list(temp_images['train'][i])
    
for i in range(len(images['test'])):
    result_images['test'][i] += list(temp_images['test'][i])


del temp_images # 필요없는 것 지움

##
# 1
train_add = {}
train_add['images'] = []
train_add['labels'] = []
for i in range(len(train_set)):
    image = []
    
    diagonal_direction_features = get_diagonal_direction_features(train_set[i][1], density=150)
    add_new_features(image, diagonal_direction_features)
    diagonal_length_features = get_diagonal_length_features(train_set[i][1], density=20)
    add_new_features(image, diagonal_length_features)
    direction_features = get_four_direction_features(train_set[i][1], density=150)
    add_new_features(image, direction_features)
    length_features = get_length_features(train_set[i][1], density=20)
    add_new_features(image, length_features)

    
#     for line in train_set[i][1]:
#         image += list(line)
    
    train_add['images'].append(image)
    train_add['labels'].append(train_set[i][0])

test_add = {}
test_add['images'] = []
test_add['labels'] = []

for i in range(len(test_set)):
    image = []
    
    diagonal_direction_features = get_diagonal_direction_features(test_set[i], density=150)
    add_new_features(image, diagonal_direction_features)
    diagonal_length_features = get_diagonal_length_features(test_set[i], density=20)
    add_new_features(image, diagonal_length_features)
    direction_features = get_four_direction_features(test_set[i], density=150)
    add_new_features(image, direction_features)
    length_features = get_length_features(test_set[i], density=20)
    add_new_features(image, length_features)


#     for line in train_set[i][1]:
#         image += list(line)
    
    
    test_add['images'].append(image)
    #test_add['labels'].append(test_set[i][0])

sc = StandardScaler()
sc.fit(train_add['images'])
std_images = {}
std_images['train'] = sc.transform(train_add['images'])
std_images['test'] = sc.transform(test_add['images'])

for i in range(len(images['train'])):
    result_images['train'][i] += list(std_images['train'][i])
for i in range(len(images['test'])):
    result_images['test'][i] += list(std_images['test'][i])
	

# 2
train_add = {}
train_add['images'] = []
train_add['labels'] = []
for i in range(len(train_set)):
    image = []
    
    eeee = extract_feature(train_set[i][1])
    add_new_features(image, eeee)
    
    train_add['images'].append(image)
    train_add['labels'].append(train_set[i][0])

test_add = {}
test_add['images'] = []
test_add['labels'] = []

for i in range(len(test_set)):
    image = []
    
    eeee = extract_feature(test_set[i])
    add_new_features(image, eeee)
    
    test_add['images'].append(image)
    #test_add['labels'].append(test_set[i][0])

sc = StandardScaler()
sc.fit(train_add['images'])
std_images = {}
std_images['train'] = sc.transform(train_add['images'])
std_images['test'] = sc.transform(test_add['images'])

for i in range(len(images['train'])):
    result_images['train'][i] += list(std_images['train'][i])
for i in range(len(images['test'])):
    result_images['test'][i] += list(std_images['test'][i])


# 3
train_add = {}
train_add['images'] = []
train_add['labels'] = []
for i in range(len(train_set)):
    image = []
    
    demension = get_two_connected_features(train_set[i][1])
    add_new_features(image, demension)
    
    train_add['images'].append(image)
    train_add['labels'].append(train_set[i][0])

test_add = {}
test_add['images'] = []
test_add['labels'] = []

for i in range(len(test_set)):
    image = []
    
    demension = get_two_connected_features(test_set[i])
    add_new_features(image, demension)
    
    test_add['images'].append(image)
    #test_add['labels'].append(test_set[i][0])

sc = StandardScaler()
sc.fit(train_add['images'])
std_images = {}
std_images['train'] = sc.transform(train_add['images'])
std_images['test'] = sc.transform(test_add['images'])

for i in range(len(images['train'])):
    result_images['train'][i] += list(std_images['train'][i])
for i in range(len(images['test'])):
    result_images['test'][i] += list(std_images['test'][i])
    
	
##
result_images['train'] = np.array(result_images['train'])
result_images['test'] = np.array(result_images['test'])


print("image processing is end.")

ppn = StochasticClassifier(lamb=0.0005, eta=0.5, random_state=1, b_size=300, n_iter=30000)
ppn = ppn.fit(result_images["train"], labels['train'])
y_pred = ppn.predict(result_images['test'])

#from sklearn.metrics import accuracy_score
#print('accuray: %.2f' % accuracy_score(test['labels'], y_pred))

f = open('prediction.txt', 'a')
for i in y_pred:
    f.write('{0}\n'.format(i))
f.close()