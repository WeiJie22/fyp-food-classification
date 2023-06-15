import streamlit as st
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import os


def load_model(weights_file, categories):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, out_features=len(categories))
    )
    model.load_state_dict(torch.load(weights_file))

    return model


def preprocess(image):
    mean = [0.554, 0.450, 0.343]
    std = [0.231, 0.241, 0.241]
    image_size = 224

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return image_transform(image)


def get_folder_items(folder_path):
    items = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            items.append(item)
    return items


def main():

    categories = ['Bread',
                  'Dairy product',
                  'Egg',
                  'Fried food',
                  'Meat',
                  'Nasi Lemak',
                  'Noodles-Pasta',
                  'Rice',
                  'Seafood',
                  'Soup',
                  'Vegetable-Fruit']

    weight_path = 'weights/with data aug'

    items = get_folder_items(weight_path)
    selected_item = st.selectbox(
        "Select a weight file", items, format_func=lambda x: x)

    if selected_item:
        model = load_model(
            f'{weight_path}/{selected_item}', categories=categories)

    st.title("Food Image Classification App")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Loading..."):
            model.eval()
            with torch.inference_mode():
                transformed_image = preprocess(image).unsqueeze(dim=0)

                image_pred = model(transformed_image)

        image_pred_probs = torch.softmax(image_pred, dim=1)

        _, sorted_idx = torch.sort(
            image_pred_probs, descending=True)

        for i in sorted_idx.squeeze().numpy():
            class_label = categories[i]
            probability = float(image_pred_probs.squeeze()[i])
            st.write(f"{class_label}: {probability:.2f}")
            st.progress(probability)


if __name__ == '__main__':
    main()
