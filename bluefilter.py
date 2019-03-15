from __future__ import division


class Img:
    def __init__(self, img_data):
        self.name = img_data[0]
        self.label = int(img_data[1])
        self.pixels = map(int, img_data[2:])
        self.wgt = 1.0

    def __repr__(self):
        return '{}_{}'.format(self.name, self.label)


train_data = open("train-data.txt", "r")
img_dict = {}
for line in train_data:
    img_data = line.strip().split()
    img = Img(img_data)
    img_dict[repr(img)] = img
train_data.close()
N = len(img_dict)


def classify_blue(img):
    sum_v = 0
    sum_h = 0
    score = {0: 0, 90: 0, 180: 0, 270: 0}
    h = {2: 1, 5: 1, 140: -1, 143: -1, 146: 1, 20: -1, 149: 1, 23: -1, 26: 1, 29: 1, 164: -1, 167: -1, 170: 1, 44: -1,
         173: 1, 47: -1, 50: 1, 53: 1, 188: -1, 191: -1, 68: -1, 71: -1, 74: 1, 77: 1, 92: -1, 95: -1, 98: 1, 101: 1,
         116: -1, 119: -1, 122: 1, 125: 1}

    v = {2: 1, 5: 1, 8: 1, 11: 1, 14: 1, 17: 1, 146: -1, 20: 1, 149: -1, 23: 1, 152: -1, 26: 1, 155: -1, 29: 1, 158: -1,
         32: 1, 161: -1, 35: 1, 164: -1, 38: 1, 167: -1, 41: 1, 170: -1, 44: 1, 173: -1, 47: 1, 176: -1, 179: -1,
         182: -1, 185: -1, 188: -1, 191: -1}

    for blue_pixel, sign in v.iteritems():
        if sign > 0:
            score[0] += img.pixels[blue_pixel]
        else:
            score[180] += img.pixels[blue_pixel]
        sum_v += img.pixels[blue_pixel]

    for blue_pixel, sign in h.iteritems():
        if sign > 0:
            score[270] += img.pixels[blue_pixel]
        else:
            score[90] += img.pixels[blue_pixel]
        sum_h += img.pixels[blue_pixel]

    score[0] = score[0] / sum_v
    score[180] = score[180] / sum_v
    score[270] = score[270] / sum_h
    score[90] = score[90] / sum_h

    return max(score, key=score.get)


def predict():
    correct = 0
    for img in img_dict.values():
        label = classify_blue(img)
        if label == img.label:
            correct += 1

    print 'accuracy: {} %'.format(correct * 100 / N)


predict()
