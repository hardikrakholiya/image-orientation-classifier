from __future__ import division
import math
import pickle


class Img:
    def __init__(self, img_data):
        self.name = img_data[0]
        self.label = int(img_data[1])
        self.pixels = map(int, img_data[2:])
        self.wgt = 1.0

        for i in range(0, 192, 3):
            prominent = max(self.pixels[i:i + 3])
            if prominent == 0:
                continue
            self.pixels[i] = self.pixels[i] * 255 / prominent
            self.pixels[i + 1] = self.pixels[i + 1] * 255 / prominent
            self.pixels[i + 2] = self.pixels[i + 2] * 255 / prominent

    def __repr__(self):
        return '{}_{}'.format(self.name, self.label)


train_data = open("train-data-small.txt", "r")
img_dict = {}
for line in train_data:
    img_data = line.strip().split()
    img = Img(img_data)
    img_dict[repr(img)] = img

train_data.close()
N = len(img_dict)

pixel_pair = []
for px1 in range(0, 192):
    for px2 in range(0, 192):
        if px1 < px2:
            pixel_pair.append([px1, px2])

# initializing classifiers for every pixel pair
features = {}
for px1 in range(0, 192):
    features[px1] = {}
    for px2 in range(0, 192):
        if px1 < px2:
            features[px1][px2] = {0: 0, 90: 0, 180: 0, 270: 0}

# looking for classifiers
for img in img_dict.values():
    for pair in pixel_pair:
        px1 = pair[0]
        px2 = pair[1]
        if img.pixels[px1] < img.pixels[px2]:
            features[px1][px2]['<'][img.label] += 1
        elif img.pixels[px1] > img.pixels[px2]:
            features[px1][px2]['>'][img.label] += 1
        else:
            features[px1][px2]['='][img.label] += 1

# writing the classifiers into model.pkl
model = {'classifiers': features}
model_pkl = open("model-small.pkl", "wb")
pickle.dump(model, model_pkl)
model_pkl.close()
print 'wrote classifiers in model-small.pkl'

model_pkl = open("model-small.pkl", "rb")
model = pickle.load(model_pkl)
features = model['classifiers']


def classify(img, f):
    px1 = f[0]
    px2 = f[1]

    if img.pixels[px1] < img.pixels[px2]:
        return max(features[px1][px2]['<'], key=features[px1][px2]['<'].get)
    elif img.pixels[px1] > img.pixels[px2]:
        return max(features[px1][px2]['>'], key=features[px1][px2]['>'].get)
    else:
        return max(features[px1][px2]['='], key=features[px1][px2]['='].get)


# https://web.stanford.edu/~hastie/Papers/samme.pdf
def adaboost():
    T = 20
    fs = []
    alphas = []
    for t in range(0, T):
        # normalize weights
        total_wgt = sum(img.wgt for img in img_dict.values())
        for img in img_dict.values():
            img.wgt = img.wgt / total_wgt

        least_e = float('inf')
        f_least_e = None
        for f in pixel_pair:
            e = 0
            for img in img_dict.values():
                if classify(img, f) != img.label:
                    e += img.wgt
            if e < least_e:
                least_e = e
                f_least_e = f

        print '{}->{}'.format(f_least_e, least_e)
        alpha = math.log((1 - least_e) * 3 / least_e)

        for img in img_dict.values():
            if img.label != classify(img, f_least_e):
                img.wgt = img.wgt * math.exp(alpha)

        fs.append(f_least_e)
        alphas.append(alpha)
        print 'alpha {}'.format(alpha)
        print 'f_least_e {}'.format(f_least_e)
    print fs
    print alphas


# adaboost()


def predict():
    correct = 0
    for img in img_dict.values():
        label_scores = {0: 0, 90: 0, 180: 0, 270: 0}

        fs = [[2, 20], [23, 191], [173, 191], [26, 146], [69, 180], [9, 72], [11, 177], [96, 119], [74, 94], [14, 181],
              [10, 179], [93, 104], [78, 92], [55, 112], [9, 134], [145, 153], [114, 187], [4, 75], [127, 133],
              [82, 106]]
        alphas = [0.9443068460147002, 0.9725466399830442, 0.8144412168746995, 0.7668041187624565, 0.35930805340315386,
                  0.3735208695618053, 0.31061977670330704, 0.31329736331275776, 0.21779191109784932,
                  0.22309588628103452, 0.2825102706757905, 0.29741278411474253, 0.21053970009709938,
                  0.19369546399404836, 0.1821107236108465, 0.18642216546015056, 0.17913155197486494,
                  0.22376339838826548, 0.18016030589690551, 0.18443443639536758]

        # alphas = [0.747035061801198, 1.2030512384981453, 0.770070911905004, 0.5700331034936513, 0.4892162370447969,
        # 0.6827658585376826, 0.42698040003490284, 0.455400459980957, 0.4449953257931318, 0.48556308048045077]
        # , 0.38782810217905633, 0.32364533018353664, 0.29859391268480184, 0.3008447957413786, 0.4068070151193691,
        #       0.3491159695290017, 0.4173731715812956, 0.32810084276252527, 0.2978601494165457, 0.24344892815027233]
        # fs = [[8, 143], [50, 185], [32, 191], [2, 164], [68, 173], [20, 125], [46, 47], [4, 5], [145, 146], [187, 188]]
        # , [92, 179], [64, 95], [14, 54], [96, 97], [9, 10], [93, 94], [180, 181], [120, 146], [9, 23], [45, 65]]

        for alpha, f in zip(alphas, fs):
            label_scores[classify(img, f)] += alpha

        label = max(label_scores, key=label_scores.get)
        print '{} classified as {}'.format(img, label)
        if label == img.label:
            correct += 1

    print 'accuracy: {} %'.format(correct * 100 / N)


predict()


# correct = 0
# for img in img_dict.values():
#     predicted_label = classify_filter(img, blue_filter)
#     print '{} classified as {}'.format(img, predicted_label)
#     if predicted_label == img.label:
#         correct += 1
#
# print 'accuracy: {} %'.format(correct * 100.0 / N)

#  0  1  2  3  4  5  6  7
#  8  9 10 11 12 13 14 15
# 16 17 18 19 20 21 22 23
# 24 25 26 27 28 29 30 31
# 32 33 34 35 36 37 38 39
# 40 41 42 43 44 45 46 47
# 48 49 50 51 52 53 54 55
# 56 57 58 59 60 61 62 63

# blue_filter = {'colors': 'b', 'v': {}, 'h': {}}
# for r in range(8):
#     for c in range(8):
#         if r in (0, 1):
#             blue_filter['v'][r * 8 + c] = 1
#
#         if r in (6, 7):
#             blue_filter['v'][r * 8 + c] = -1
#
# for r in range(8):
#     for c in range(8):
#         if c in (0, 1):
#             blue_filter['h'][r * 8 + c] = 1
#
#         if c in (6, 7):
#             blue_filter['h'][r * 8 + c] = -1

# light_filter = {'colors': 'rgb', 'v': {}, 'h': {}}
# for r in range(8):
#     for c in range(8):
#         if r in (0, 1, 2):
#             light_filter['v'][r * 8 + c] = 1
#
#         if r in (6, 7, 5):
#             light_filter['v'][r * 8 + c] = -1
#
# for r in range(8):
#     for c in range(8):
#         if c in (0, 1, 2):
#             light_filter['h'][r * 8 + c] = 1
#
#         if c in (6, 7, 5):
#             light_filter['h'][r * 8 + c] = -1


