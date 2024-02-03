import streamlit as st
from PIL import Image
import imageio
import os
from io import BytesIO

# 一時的な画像を保存するディレクトリの作成
tmp_dir = "tmp_images"
os.makedirs(tmp_dir, exist_ok=True)

def clear_tmp_dir():
    """一時ディレクトリ内のファイルを削除"""
    for filename in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

def create_gif(image_paths, duration, loop):
    """画像リストからGIFを作成し、BytesIOオブジェクトとして返す"""
    images = [imageio.imread(image_path) for image_path in image_paths]
    with BytesIO() as buffer:
        imageio.mimsave(buffer, images, format='GIF', duration=duration, loop=loop)
        buffer.seek(0)
        return buffer.getvalue()

st.title('GIF Creator')

# ファイルアップローダー
uploaded_files = st.file_uploader("Upload Images", type=['png', 'jpg'], accept_multiple_files=True)

# GIFパラメータ
duration = st.slider("フレーム間の遅延（秒）", min_value=0.05, max_value=2.0, value=0.1, step=0.05)
loop = st.slider("ループ回数（0は無限ループ）", min_value=0, max_value=10, value=0)

if st.button('GIFを作成'):
    if uploaded_files is not None:
        # 一時ディレクトリのクリア
        clear_tmp_dir()

        image_paths = []
        for uploaded_file in uploaded_files:
            # 画像を一時ディレクトリに保存
            img = Image.open(uploaded_file)
            img_path = os.path.join(tmp_dir, uploaded_file.name)
            img.save(img_path)
            image_paths.append(img_path)

        # GIFの作成
        gif_bytes = create_gif(image_paths, duration, loop)

        # GIFの表示
        st.image(gif_bytes)

        # GIFをダウンロード可能にする
        st.download_button(label="GIFをダウンロード", data=gif_bytes, file_name="output.gif", mime="image/gif")

        # 一時ディレクトリのクリア
        clear_tmp_dir()
