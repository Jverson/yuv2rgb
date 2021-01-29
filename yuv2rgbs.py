import os
import cv2
import shutil

from glob import glob

import numpy as np


RK_VIDEO_DIR = "./data/yuv_videos"
RK_IMAGE_DIR = "./data/rgb_images"
IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_SIZE = int(IMG_WIDTH * IMG_HEIGHT * 3 / 2)

Y_WIDTH = IMG_WIDTH
Y_HEIGHT = IMG_HEIGHT
Y_SIZE = int(Y_WIDTH * Y_HEIGHT)

U_V_WIDTH = int(IMG_WIDTH / 2)
U_V_HEIGHT = int(IMG_HEIGHT / 2)
U_V_SIZE = int(U_V_WIDTH * U_V_HEIGHT)


if not os.path.exists(RK_IMAGE_DIR):
    os.mkdir(RK_IMAGE_DIR)


def from_I420(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_start = y_start + Y_SIZE
        v_start = u_start + U_V_SIZE
        v_end = v_start + U_V_SIZE

        Y[frame_idx, :, :] = yuv_data[y_start : u_start].reshape((Y_HEIGHT, Y_WIDTH))
        U[frame_idx, :, :] = yuv_data[u_start : v_start].reshape((U_V_HEIGHT, U_V_WIDTH))
        V[frame_idx, :, :] = yuv_data[v_start : v_end].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def from_YV12(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        v_start = y_start + Y_SIZE
        u_start = v_start + U_V_SIZE
        u_end = u_start + U_V_SIZE

        Y[frame_idx, :, :] = yuv_data[y_start : v_start].reshape((Y_HEIGHT, Y_WIDTH))
        V[frame_idx, :, :] = yuv_data[v_start : u_start].reshape((U_V_HEIGHT, U_V_WIDTH))
        U[frame_idx, :, :] = yuv_data[u_start : u_end].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def from_NV12(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_v_start = y_start + Y_SIZE
        u_v_end = u_v_start + (U_V_SIZE * 2)

        Y[frame_idx, :, :] = yuv_data[y_start : u_v_start].reshape((Y_HEIGHT, Y_WIDTH))
        U_V = yuv_data[u_v_start : u_v_end].reshape((U_V_SIZE, 2))
        U[frame_idx, :, :] = U_V[:, 0].reshape((U_V_HEIGHT, U_V_WIDTH))
        V[frame_idx, :, :] = U_V[:, 1].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V


def from_NV21(yuv_data, frames):
    Y = np.zeros((frames, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    U = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)
    V = np.zeros((frames, U_V_HEIGHT, U_V_WIDTH), dtype=np.uint8)

    for frame_idx in range(0, frames):
        y_start = frame_idx * IMG_SIZE
        u_v_start = y_start + Y_SIZE
        u_v_end = u_v_start + (U_V_SIZE * 2)

        Y[frame_idx, :, :] = yuv_data[y_start : u_v_start].reshape((Y_HEIGHT, Y_WIDTH))
        U_V = yuv_data[u_v_start : u_v_end].reshape((U_V_SIZE, 2))
        V[frame_idx, :, :] = U_V[:, 0].reshape((U_V_HEIGHT, U_V_WIDTH))
        U[frame_idx, :, :] = U_V[:, 1].reshape((U_V_HEIGHT, U_V_WIDTH))
    return Y, U, V

def yuv2rgb(Y, U, V):
    bgr_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    for h_idx in range(Y_HEIGHT):
        for w_idx in range(Y_WIDTH):
            y = Y[h_idx, w_idx]
            u = U[int(h_idx // 2), int(w_idx // 2)]
            v = V[int(h_idx // 2), int(w_idx // 2)]

            c = (y - 16) * 298
            d = u - 128
            e = v - 128
            r = (c + 409 * e + 128) // 256
            g = (c - 100 * d - 208 * e + 128) // 256
            b = (c + 516 * d + 128) // 256

            bgr_data[h_idx, w_idx, 2] = 0 if r < 0 else (255 if r > 255 else r)
            bgr_data[h_idx, w_idx, 1] = 0 if g < 0 else (255 if g > 255 else g)
            bgr_data[h_idx, w_idx, 0] = 0 if b < 0 else (255 if b > 255 else b)
            print(bgr_data[h_idx, w_idx, :])
    return bgr_data


if __name__ == '__main__':
    if not os.path.exists(RK_IMAGE_DIR):
        os.mkdir(RK_IMAGE_DIR)

    yuvs = glob(os.path.join(RK_VIDEO_DIR, "*.yuv"))
    yuvs.sort()
    total = len(yuvs)

    yuv_idx = 0
    for yuv in yuvs:
        print("{} / {} -- {}, size: {}".format(yuv_idx + 1, total, yuv, os.path.getsize(yuv)))

        yuv_path, yuv_name = os.path.split(yuv)
        img_out_dir = os.path.join(RK_IMAGE_DIR, yuv_name[:-4])
        #print(img_out_dir)
        if os.path.exists(img_out_dir):
            shutil.rmtree(img_out_dir)
        os.mkdir(img_out_dir)

        frames = int(os.path.getsize(yuv) / IMG_SIZE)
        #print("frames: {}".format(frames))

        with open(yuv, "rb") as yuv_f:
            yuv_bytes = yuv_f.read()
            yuv_data = np.frombuffer(yuv_bytes, np.uint8)
            #print(len(yuv_data))
            Y, U, V = from_I420(yuv_data, frames)
            #Y, U, V = from_YV12(yuv_data, frames)
            #Y, U, V = from_NV12(yuv_data, frames)
            #Y, U, V = from_NV21(yuv_data, frames)
            rgb_data = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            for frame_idx in range(frames):
                bgr_data = yuv2rgb(Y[frame_idx, :, :], U[frame_idx, :, :], V[frame_idx, :, :])
                if bgr_data is not None:
                    cv2.imwrite(os.path.join(img_out_dir, "frame_{}.jpg".format(frame_idx)), bgr_data)

        yuv_idx += 1
