
import cv2
import math
import numpy as np
import os

def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def atmospheric_light(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = max(math.floor(imsz / 1000), 1)

    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx:]

    atmsum = np.zeros([1, 3])
    for ind in range(numpx):
        atmsum += imvec[indices[ind]]

    A = atmsum / numpx
    return A

def transmission_estimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    transmission = 1 - omega * dark_channel(im3, sz)
    return transmission

def guided_filter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q

def refine_transmission(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = guided_filter(gray, et, r, eps)
    return t

def recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)
    for ind in range(3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
    return res

def dehaze_image(fn, sz=15, tx=0.1):
    src = cv2.imread(fn)
    if src is None:
        raise ValueError(f"Unable to load image: {fn}")

    I = src.astype('float64') / 255

    dark = dark_channel(I, sz)
    A = atmospheric_light(I, dark)
    te = transmission_estimate(I, A, sz)
    t = refine_transmission(src, te)
    J = recover(I, t, A, tx)

    return src, J

if __name__ == '__main__':
    import sys

    try:
        fn = sys.argv[1]
    except IndexError:
        fn = "/home/lachimolala/Documents/python/image.jpg"

    try:
        src, J = dehaze_image(fn)

        # Rescale dehazed image to uint8
        J_display = np.clip(J * 255, 0, 255).astype(np.uint8)

        # Create output folder
        output_dir = os.path.join(".", "output_images")
        os.makedirs(output_dir, exist_ok=True)

        # Save the images
        hazy_path = os.path.join(output_dir, "Hazy_Image.png")
        dehazed_path = os.path.join(output_dir, "Dehazed_Image.png")

        cv2.imwrite(hazy_path, src)
        cv2.imwrite(dehazed_path, J_display)

        print(f"Images saved successfully in '{output_dir}'.")

    except Exception as e:
        print(f"Error: {e}")
