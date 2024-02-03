import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def simple_diff(img1, img2):
    """単純なピクセル単位の差分を計算する"""
    return cv2.absdiff(img1, img2)

def gaussian_diff(img1, img2, kernel_size=5):
    """ガウシアンブラーを適用後の差分を計算する"""
    blur1 = cv2.GaussianBlur(img1, (kernel_size, kernel_size), 0)
    blur2 = cv2.GaussianBlur(img2, (kernel_size, kernel_size), 0)
    return cv2.absdiff(blur1, blur2)

def ssim_diff(img1, img2):
    """SSIMを計算し、差分画像を生成する"""
    (score, diff) = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    return diff

def gradient_diff(img1, img2):
    """勾配差分を計算する"""
    grad_x1 = cv2.Sobel(img1, cv2.CV_16S, 1, 0)
    grad_y1 = cv2.Sobel(img1, cv2.CV_16S, 0, 1)
    grad_x2 = cv2.Sobel(img2, cv2.CV_16S, 1, 0)
    grad_y2 = cv2.Sobel(img2, cv2.CV_16S, 0, 1)
    abs_grad_x1 = cv2.convertScaleAbs(grad_x1)
    abs_grad_y1 = cv2.convertScaleAbs(grad_y1)
    abs_grad_x2 = cv2.convertScaleAbs(grad_x2)
    abs_grad_y2 = cv2.convertScaleAbs(grad_y2)
    grad1 = cv2.addWeighted(abs_grad_x1, 0.5, abs_grad_y1, 0.5, 0)
    grad2 = cv2.addWeighted(abs_grad_x2, 0.5, abs_grad_y2, 0.5, 0)
    return cv2.absdiff(grad1, grad2)

# Streamlit UI
st.title("Image Diff Viewer")

# 画像アップローダー
img1_file = st.file_uploader("Upload the first Image", type=['png', 'jpg', 'jpeg'])
img2_file = st.file_uploader("Upload the second Image", type=['png', 'jpg', 'jpeg'])

if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    # OpenCVで扱える形式に変換
    img1 = np.array(img1)
    img2 = np.array(img2)

    # グレースケール変換
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 画像差分を計算
    simple_diff_img = simple_diff(img1_gray, img2_gray)
    gaussian_diff_img = gaussian_diff(img1_gray, img2_gray)
    ssim_diff_img = ssim_diff(img1_gray, img2_gray)
    gradient_diff_img = gradient_diff(img1_gray, img2_gray)

    # 結果を表示
    st.image([img1, img2, simple_diff_img, gaussian_diff_img, ssim_diff_img, gradient_diff_img],
             caption=["画像1", "画像2", "単純差分", "ガウシアン差分", "SSIM差分", "勾配差分"], use_column_width=True)
