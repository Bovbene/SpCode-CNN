import numpy as np

def arraycompare(array1, array2, height, width):
    resultarray = np.zeros((height, width))
    for row in range(0, height):
        for col in range(0, width):
            resultarray[row, col] = max(array1[row, col], array2[row, col])
    return resultarray


def integralImgSqDiff2(paddedimg_val, Ds_val, t1_val, t2_val):

    lengthrow = len(paddedimg_val[:, 0])
    lengthcol = len(paddedimg_val[0, :])
    Dist2 = (paddedimg_val[Ds_val:-Ds_val, Ds_val:-Ds_val] -
             paddedimg_val[Ds_val + t1_val:lengthrow - Ds_val + t1_val,
                           Ds_val + t2_val:lengthcol - Ds_val + t2_val]) ** 2
    Sd_val = Dist2.cumsum(0)
    Sd_val = Sd_val.cumsum(1)
    return Sd_val


def nl_meansfilter(imagearray, h_=10, ds0=2, ds1=5):
    height, width = imagearray[:, :].shape
    length0 = height + 2 * ds1
    length1 = width + 2 * ds1
    h = (h_ ** 2)
    d = (2 * ds0 + 1) ** 2
    imagearray_NL = np.zeros(imagearray.shape).astype('uint8')
    for i in range(0, 1):
        paddedimg = np.pad(imagearray[:, :], ds0 + ds1 + 1, 'symmetric')
        paddedimg = paddedimg.astype('float64')
        paddedv = np.pad(imagearray[:, :], ds1, 'symmetric')
        paddedv = paddedv.astype('float64')
        average = np.zeros((height, width))
        sweight = np.zeros((height, width))
        wmax = np.zeros((height, width))
        for t1 in range(-ds1, ds1 + 1):
            for t2 in range(-ds1, ds1 + 1):
                if t1 == 0 and t2 == 0:
                    continue
                Sd = integralImgSqDiff2(paddedimg, ds1, t1, t2)
                SqDist2 = Sd[2 * ds0 + 1:-1, 2 * ds0 + 1:-1] + Sd[0:-2 * ds0 - 2, 0:-2 * ds0 - 2] - \
                          Sd[2 * ds0 + 1:-1, 0:-2 * ds0 - 2] - Sd[0:-2 * ds0 - 2, 2 * ds0 + 1:-1]
                SqDist2 /= d * h
                w = np.exp(-SqDist2)
                v = paddedv[ds1 + t1:length0 - ds1 + t1, ds1 + t2:length1 - ds1 + t2]
                average += w * v
                wmax = arraycompare(wmax, w, height, width)
                sweight += w
        average += wmax * imagearray[:, :]
        average /= wmax + sweight
        average_uint8 = average.astype('uint8')
        imagearray_NL[:, :] = average_uint8
    return imagearray_NL
	
	
	
	