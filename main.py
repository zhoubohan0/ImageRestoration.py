import cv2
import numpy as np
import os
from skimage import transform


def Variance_compute(center, contours):
    points = contours.reshape((-1, 2))
    distance = np.linalg.norm(points - center, axis=1)
    variance = np.var(distance)
    return variance


def Circle_detect(img, minVar, minRadius, maxRadius):
    result = []
    # 对二值图像生成边界
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        # 根据边界情况得到最小圆的圆心和半径
        center, radius = cv2.minEnclosingCircle(contours[i])
        # 计算边界上各点至最小圆的圆心的距离方差，当方差小于minVar使可以认为符合圆形检测
        variance = Variance_compute(center, contours[i])
        if variance < minVar and minRadius < radius < maxRadius:
            result.append((*center, radius))
    return result


def getspotposition(raw_img, variance=0.8, minRadius=5, maxRadius=12,display=False):
    # 提取红色区域并做开运算，使目标区域平滑
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([150, 43, 99])
    upper_red = np.array([180, 255, 255])
    thresh = cv2.inRange(img, lower_red, upper_red)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # 检测圆形区域，circles为圆心和半径的三元组列表
    circles = Circle_detect(thresh, variance, minRadius, maxRadius)
    if len(circles) == 0:
        return False, np.round(20 * np.random.rand(1, 3)).astype(np.int16)  # 找不到返回一个随机点和半径
    if display:
        displayimage = raw_img.copy()
        for x, y, r in circles:
            displayimage = cv2.circle(displayimage, np.array([x, y],dtype=np.int16), int(r), (0, 255, 0), thickness=-1)
        show(displayimage)
    maxr = 0
    for y, x, r in circles:
        if r>maxr:
            maxr = r
            circleinfo = np.array([x, y, r],dtype=np.int16)
    return True,circleinfo  # 三元组，坐标+半径


def show(img, filename=''):
    cv2.imshow('', img)
    cv2.waitKey()
    if filename != '':
        cv2.imwrite(filename, img)


def isred(color, adjust=200):
    return color[0] >= adjust or color[1] >= adjust or color[2] >= adjust


def getredspot(img, adjust=205):
    h, w = img.shape[:-1]
    outimg = np.zeros_like(img)
    avepos = np.zeros(2, dtype=np.float64)
    cnt = 0
    for i in range(h):
        for j in range(w):
            if isred(img[i, j], adjust=adjust) and j / w > 0.4:
                outimg[i, j] = img[i, j]
                avepos = avepos + np.array([i, j])
                cnt += 1

            else:
                outimg[i, j] = [0, 0, 0]
    avepos = avepos / cnt
    return outimg, avepos.astype(np.uint8)  # np.array([int(i)for i in avepos]),maxcolor,maxpos


def within(pos, img):
    h, w = img.shape[0], img.shape[1]
    return 0 <= pos[0] < h and 0 <= pos[1] < w


def gaussian(pos, target, sigma=1):
    sigmax = sigmay = sigma
    X = (pos[0] - target[0]) * (pos[0] - target[0]) / (2 * sigmax * sigmax)
    Y = (pos[1] - target[1]) * (pos[1] - target[1]) / (2 * sigmay * sigmay)
    return np.exp(-(X + Y))  # if np.linalg.norm(pos - target, ord=2) > 10 else 1


def addredspot(srcimg, targetpos, redspot, center_redspot, sigma=6, adjust=205):
    outimg = srcimg.copy()
    for i in range(srcimg.shape[0]):
        for j in range(srcimg.shape[1]):
            tmp_pos = np.array([i, j]) - targetpos + center_redspot
            if within(tmp_pos, redspot) and sum(redspot[tmp_pos[0], tmp_pos[1]]) > adjust:
                # outimg[i, j] = redspot[tmp_pos[0],tmp_pos[1]]
                theta = gaussian(tmp_pos, center_redspot, sigma=sigma)
                outimg[i, j] = theta * redspot[tmp_pos[0], tmp_pos[1]] + (1 - theta) * outimg[i, j]
    return outimg


def changescale(origin, scale=0.6):
    h, w, channel = origin.shape
    scale_h = scale_w = scale
    scaleimg = np.zeros((round(scale_h * h), round(scale_w * w), channel))
    for i in range(channel):
        scaleimg[:, :, i] = transform.rescale(origin[:, :, i],
                                              scale=scale)  # cv2.resize(origin[:,:,i],dsize=(int(scale_w * w), int(scale_h * h)),interpolation=cv2.INTER_CUBIC)
    scaleimg = 255 * scaleimg
    return scaleimg.astype(np.uint8)



def removeredspot(inputimg,circleinfo):
    # 获得掩膜(掩膜比检测出的圆的半径略大)
    delta = 2
    mask = np.zeros(inputimg.shape[:-1], dtype=np.uint8)
    try:
        c_x, c_y, radius = circleinfo
    except ValueError:
        return inputimg
    radius+=delta
    pos_redspot = np.array([c_x, c_y])

    for i in range(c_x - radius + 1,c_x + radius + 1):
        for j in range(c_y - radius + 1, c_y + radius + 1):
            temp = np.array([i,j])
            if within(temp, inputimg) and np.linalg.norm(temp-pos_redspot)<=radius:
                mask[i,j] = 1
    # for c_x, c_y, radius in circleinfo:
    # mask[max(c_x - radius + 1, 0):min(c_x + radius + 1, w_mask),
    # max(c_y - radius + 1, 0):min(h_mask, c_y + radius + 1)] = 1
    # 图像复原
    restoredimg = cv2.inpaint(src=inputimg, inpaintMask=mask, inpaintRadius=radius,flags=cv2.INPAINT_TELEA)  # cv2.INPAINT_NS
    return restoredimg



def main(sourcepath,datasetdirpath,outputdirpath):
    imgwithspotpath = './test_withspot'
    if not os.path.exists(imgwithspotpath):
        os.mkdir(imgwithspotpath)
    # 读入红点图片
    source = cv2.imread(sourcepath)
    filenames = os.listdir(datasetdirpath)
    for i,filename in enumerate(filenames):
        print(f'processing image {i+1}')
        # 红点缩放
        scale_source = changescale(source, scale=0.3 + np.random.rand() * 0.2)
        # show(scale_source,'scaleredspot.jpg')
        # 获得掩膜
        redspot, center_redspot = getredspot(scale_source)
        # show(redspot,'redspot.jpg')
        # 读入测试图片
        srcimg = cv2.imread(os.path.join(datasetdirpath,filename))
        # 为测试图片随机添加激光红点
        targetpos = np.int16(np.random.rand(1, 2) * np.array(srcimg.shape[:-1])).squeeze()
        imgwithredspot = addredspot(srcimg, targetpos, redspot, center_redspot)
        show(imgwithredspot,os.path.join(imgwithspotpath,filename))
        # show(imgwithredspot,'imgwithredspot.jpg')
        # 算法获取激光红点位置
        success, circleinfo = getspotposition(imgwithredspot)
        # 生成掩膜除去激光红点
        imgwithoutredspot = removeredspot(imgwithredspot, circleinfo)
        show(imgwithoutredspot, os.path.join(outputdirpath,filename))




if __name__ == '__main__':
    sourcepath = r'./src/redspot.jpg'
    datasetdirpath = './test'
    outputdirpath = './test_withoutspot'

    main(sourcepath,datasetdirpath,outputdirpath)