#!/usr/bin/env python
# Author: Hardik Rakholiya

from __future__ import division
import pickle
import math


class Img:
    def __init__(self, img_data):
        self.name = img_data[0]
        self.label = int(img_data[1])
        self.pixels = map(int, img_data[2:])
        self.wgt = 1.0

    def __repr__(self):
        return '{}_{}'.format(self.name, self.label)


blue_filter = {
    'h': {0: 1, 1: 1, 6: -1, 7: -1, 8: 1, 9: 1, 14: -1, 15: -1, 16: 1, 17: 1, 22: -1, 23: -1, 24: 1, 25: 1, 30: -1,
          31: -1, 32: 1, 33: 1, 38: -1, 39: -1, 40: 1, 41: 1, 46: -1, 47: -1, 48: 1, 49: 1, 54: -1, 55: -1, 56: 1,
          57: 1, 62: -1, 63: -1}, 'colors': 'b',
    'v': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 48: -1,
          49: -1, 50: -1, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: -1, 58: -1, 59: -1, 60: -1, 61: -1,
          62: -1, 63: -1}}

light_filter = {
    'h': {0: 1, 1: 1, 2: 1, 5: -1, 6: -1, 7: -1, 8: 1, 9: 1, 10: 1, 13: -1, 14: -1, 15: -1, 16: 1, 17: 1, 18: 1, 21: -1,
          22: -1, 23: -1, 24: 1, 25: 1, 26: 1, 29: -1, 30: -1, 31: -1, 32: 1, 33: 1, 34: 1, 37: -1, 38: -1, 39: -1,
          40: 1, 41: 1, 42: 1, 45: -1, 46: -1, 47: -1, 48: 1, 49: 1, 50: 1, 53: -1, 54: -1, 55: -1, 56: 1, 57: 1, 58: 1,
          61: -1, 62: -1, 63: -1}, 'colors': 'rgb',
    'v': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1,
          17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 40: -1, 41: -1, 42: -1, 43: -1, 44: -1, 45: -1, 46: -1,
          47: -1, 48: -1, 49: -1, 50: -1, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: -1, 58: -1, 59: -1,
          60: -1, 61: -1, 62: -1, 63: -1}}

train_pile = []
features = []
classifiers = {}


def classify_filter(img, f):
    score_v = 0.0
    sum_v = 0.0
    score_h = 0.0
    sum_h = 0.0

    for cell, sign in f['v'].iteritems():
        if 'r' in f['colors']:
            score_v += img.pixels[3 * cell] * sign
            sum_v += img.pixels[3 * cell]
        if 'g' in f['colors']:
            score_v += img.pixels[3 * cell + 1] * sign
            sum_v += img.pixels[3 * cell + 1]
        if 'b' in f['colors']:
            score_v += img.pixels[3 * cell + 2] * sign
            sum_v += img.pixels[3 * cell + 2]

    for cell, sign in f['h'].iteritems():
        if 'r' in f['colors']:
            score_h += img.pixels[3 * cell] * sign
            sum_h += img.pixels[3 * cell]
        if 'g' in f['colors']:
            score_h += img.pixels[3 * cell + 1] * sign
            sum_h += img.pixels[3 * cell + 1]
        if 'b' in f['colors']:
            score_h += img.pixels[3 * cell + 2] * sign
            sum_h += img.pixels[3 * cell + 2]

    p_vert = abs(score_v / sum_v)
    p_horz = abs(score_h / sum_h)

    if p_vert >= p_horz:
        if score_v >= 0:
            return 0
        else:
            return 180
    else:
        if score_h >= 0:
            return 270
        else:
            return 90


def classify(img, f):
    global classifiers

    if f == 'blue_filter':
        return classify_filter(img, blue_filter)
    if f == 'light_filter':
        return classify_filter(img, light_filter)

    px1 = f[0]
    px2 = f[1]

    return max(classifiers[px1][px2], key=classifiers[px1][px2].get) if img.pixels[px1] < img.pixels[px2] \
        else min(classifiers[px1][px2], key=classifiers[px1][px2].get)


fs = []
alphas = []


def generate_classifiers():
    global classifiers, features, train_pile
    for px1 in range(0, 192):
        for px2 in range(0, 192):
            if px1 < px2:
                features.append([px1, px2])

    # initializing classifiers for every pixel pair
    for px1_px2 in features:
        px1 = px1_px2[0]
        px2 = px1_px2[1]
        if px1 not in classifiers:
            classifiers[px1] = {}
        if px1 < px2:
            classifiers[px1][px2] = {0: 0, 90: 0, 180: 0, 270: 0}

    # looking for classifiers
    for img in train_pile:
        for pair in features:
            px1 = pair[0]
            px2 = pair[1]
            if img.pixels[px1] < img.pixels[px2]:
                classifiers[px1][px2][img.label] += 1

    features.append('blue_filter')
    features.append('light_filter')


# https://web.stanford.edu/~hastie/Papers/samme.pdf
def run_adaboost():
    global features, train_pile

    T = 20
    for t in range(0, T):
        # normalize weights
        total_wgt = sum(img.wgt for img in train_pile)
        for img in train_pile:
            img.wgt = img.wgt / total_wgt

        least_e = float('inf')
        f_least_e = None

        for f in features:
            e = 0.0
            for img in train_pile:
                if classify(img, f) != img.label:
                    e += img.wgt
            if e < least_e:
                least_e = e
                f_least_e = f

        alpha = math.log((1 - least_e) * 3 / least_e)

        for img in train_pile:
            if img.label != classify(img, f_least_e):
                img.wgt = img.wgt * math.exp(alpha)

        fs.append(f_least_e)
        alphas.append(alpha)


def train_adaboost(train_file, model_txt):
    global classifiers, features, train_pile
    train_data = open(train_file, "r")
    for line in train_data:
        img_data = line.strip().split()
        img = Img(img_data)
        train_pile.append(img)
    train_data.close()

    generate_classifiers()
    run_adaboost()

    # writing the classifiers into adaboost_model.txt
    model = {'classifiers': classifiers, 'fs': fs, 'alphas': alphas}
    model_file = open(model_txt, "wb")
    pickle.dump(model, model_file)
    model_file.close()


def test_adaboost(test_file, model_txt):
    global classifiers
    model_file = open(model_txt, "rb")
    model = pickle.load(model_file)
    classifiers = model['classifiers']
    fs = model['fs']
    alphas = model['alphas']

    test_data = open(test_file, "r")
    test_pile = []

    for line in test_data:
        img_data = line.strip().split()
        img = Img(img_data)
        test_pile.append(img)

    correct = 0
    for img in test_pile:
        label_scores = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}

        for alpha, f in zip(alphas, fs):
            label_scores[classify(img, f)] += alpha

        predicted_label = max(label_scores, key=label_scores.get)
        if predicted_label == img.label:
            correct += 1

    print 'accuracy: {} %'.format(correct * 100.0 / len(test_pile))



