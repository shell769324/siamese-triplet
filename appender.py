from matplotlib.image import imread
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage.feature import match_template
from skimage.transform import resize
import time
import multiprocessing as mp
import cv2
from PIL import Image
import os
import glob

def getSubImg(img, size, cx, cy):
    halfSize = int(size / 2)
    (lx, ly) = (cx - halfSize, cy - halfSize)
    (rx, ry) = (cx + halfSize + 1, cy + halfSize + 1)
    lxs = -lx if lx < 0 else 0
    lys = -ly if ly < 0 else 0
    (lx, ly) = (max(0, lx), max(0, ly))
    (rx, ry) = (min(len(img[0]), rx), min(len(img), ry))
    curr = img[ly:ry, lx:rx]
    res = np.random.rand(size, size)
    res[lys:lys + len(curr), lxs:lxs + len(curr[0])] = curr
    return res

def ac(img, size, cx, cy):
    patch = getSubImg(img, size, cx, cy)
    surround = getSubImg(img, size * 3 - 2, cx, cy)
    return match_template(surround, patch)


def acAvg(img, cx, cy, step = 8, start = 23, end = 50):
    accCorr = ac(img, start, cx, cy)

    #for i in range(start + step, end + 1, step):
    #for i in range(start + 4, start + 5, step):
    #    freq = np.ones((i * 2 - 1, i * 2 - 1))
    #    freq[step:(step + len(accFreq)), step:(step + len(accFreq))] += accFreq
    #    accFreq = freq
    #    corr = ac(img, i, cx, cy)
    #    corr[step:(step + len(accCorr)), step:(step + len(accCorr))] += accCorr
    #    accCorr = corr

    center = int(len(accCorr)/2)
    accCorr[center - 1:center + 2, center - 1:center + 2] = np.zeros((3, 3))
    return accCorr

def findPeaks(img, numPeaks, thresholdFrac=0.5, neighSpan=2):
    threshold = (thresholdFrac * (img.max() - img.min())) + img.min()

    rows, cols = np.array(np.where(img >= threshold))
    values = []
    for i, j in zip(rows, cols):
        values.append((i, j, img[i, j]))

    dtype = [('row', int), ('col', int), ('intensity', np.float64)]
    indices = np.array(values, dtype=dtype)

    indices[::-1].sort(order='intensity')
    res = []

    for idx in indices:
        intensity = idx[2]
        if intensity <= -1:
            continue

        x0 = idx[1] - neighSpan
        xend = idx[1] + neighSpan
        y0 = idx[0] - neighSpan
        yend = idx[0] + neighSpan

        toSuppress = np.where((indices['col'] >= x0) &
                                       (indices['col'] <= xend) &
                                       (indices['row'] >= y0) &
                                       (indices['row'] <= yend))
        if toSuppress:
            indices['intensity'][toSuppress] = -2
        idx[2] = intensity
        res.append([idx[0], idx[1]])
        if(len(res) >= numPeaks):
            break
    res = np.array(res)
    return res

def h(cpq, peaks, AC, converted, i, j):
    def frac(a):
        return abs(a - round(a))
    res = 0
    for k in range(len(peaks)):
        if k == i or k == j:
            continue
        abk = cpq.dot(converted[k])
        res += (1 - 2 * max(frac(abk[0]), frac(abk[1]))) * AC[peaks[k][0]][peaks[k][1]]
    return res

def score(peaks, AC, alpha = 2):
    def isBadAngle(v1, v2):
        return abs(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) > 0.6
    best = 0
    converted = (peaks - int(len(AC) / 2)).astype(float)
    for i in range(len(peaks)):
        cp = peaks[i]
        cpc = converted[i]
        for j in range(i + 1, len(peaks)):
            cq = peaks[j]
            cqc = converted[j]
            cpq = np.array([[cpc[0], cqc[0]],
                            [cpc[1], cqc[1]]])
            if isBadAngle(cpc, cqc):
                continue
            cpq = np.linalg.inv(cpq)
            local = h(cpq, peaks, AC, converted, i, j)
            local /= np.linalg.norm(cpc) + np.linalg.norm(cqc)
            local += alpha * (AC[cp[0]][cp[1]] + AC[cq[0]][cq[1]])
            if local > best:
                best = local
    return best

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.33333, 0.33333, 0.33333])


def getSampleWithMult(addr):
    img = imread(addr)
    if "png" in addr:
        img = img * 256
    img = np.uint8(rgb2gray(img))
    if addr[-2] != 'n':
        img = cv2.fastNlMeansDenoising(img)
    #(lx, rx) = (int(len(img[0])/4), int(len(img[0]) * 3/4))
    #(ly, ry) = (int(len(img)/4), int(len(img) * 3/4))
    (lx, rx) = (int(0), int(len(img[0])))
    (ly, ry) = (int(0), int(len(img)))
    #plt.imshow(img)
    #plt.show()
    manager = mp.Manager()
    heat = manager.list()
    lok = mp.Lock()
    for _ in range(ry - ly):
        heat.append([])

    def oneRow(lok, heat, cy):
        res = [0 for _ in range(rx - lx)]
        if abs(cy - ly) < 11 or abs(cy - ry) < 11:
            lok.acquire()
            heat[cy - ly] = res
            lok.release()
            return
        for cx in range(lx + 11, max(rx - 10, lx + 11)):
            ac = acAvg(img, cx, cy)
            peaks = findPeaks(ac, 7, 0.5, 6)
            sc = score(peaks, ac)
            res[cx] = sc
        lok.acquire()
        heat[cy - ly] = res
        lok.release()
    processes = []
    for cy in range(ly, ry):
        p = mp.Process(target = oneRow, args = (lok, heat, cy))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    heat = np.array(heat)
    best = np.max(heat)
    #plt.imshow(heat)
    #plt.show()
    return np.hstack((img[ly:ry, lx:rx], heat * 255/best))

def easyStack(addr):
    img = imread(addr)
    if "png" in addr:
        img = img * 256
    img = np.uint8(rgb2gray(img))
    if addr[-2] != 'n':
        img = cv2.fastNlMeansDenoising(img)
    zeros = np.zeros(tuple(img.shape))
    return np.hstack((img, zeros))

