import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from modules.predict import predict_digit

def load_image(image_file):
	img = Image.open(image_file)
	return img


choice = "Image"

if choice == "Image":

    st.subheader("Image")
    st.caption('Trang web này giúp bạn dự đoán chữ số viết tay với độ chính xác tương đối :)))')
    st.caption('Bạn hãy vẽ một bức ảnh bằng paint hoặc bất kì app vẽ nào khác nhưng lưu ý ảnh chỉ có một chữ và nền thôi nhé, đừng vẽ thêm những thứ khác')
    st.caption('Sau đó lưu lại và upload vào đây nhé!!!')
    image_file = st.file_uploader("Upload Images",
                                  type=["png", "jpg", "jpeg"])

    if image_file is not None:
        # TO See details
        file_details = {"filename": image_file.name, "filetype": image_file.type,
                        "filesize": image_file.size}
        st.write(file_details)
        st.image(load_image(image_file), width=250)

        # Saving upload
        with open(os.path.join("fileDir", image_file.name), "wb") as f:
            f.write((image_file).getbuffer())

        st.success("File Saved")
        num, acc = predict_digit(os.path.join("fileDir", image_file.name))
        st.write("Số bạn đã vẽ là: ", num)
        st.write("Độ chính xác: ", acc)

