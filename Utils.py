import os
import numpy as np
import itertools
from keras import backend as K


def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for c in out_best:
        if c < len(letters):
            outstr += letters[c]
    return outstr

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

def preprocess(path, img_w, img_h):
    """ Pre-processing image for predicting """
    img = cv.imread(path)
    img = fix_size(img, img_w, img_h)

    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img = img.astype(np.float32)
    img /= 255
    return img
def predict_image(model_predict, path, is_word):
    if is_word:
        width = word_cfg['img_w']
    else:
        width = line_cfg['img_w']
    img = preprocess(path, width, 64)
    img = img.T
    if K.image_data_format() == 'channels_first':
        img = np.expand_dims(img, 0)
    else:
        img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)

    net_out_value = model_predict.predict(img)
    pred_texts = decode_label(net_out_value)
    return pred_texts
