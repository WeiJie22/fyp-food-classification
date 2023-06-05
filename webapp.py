import streamlit as st
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt


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


def main():

    categories = ['Bread',
                  'Dairy product',
                  'Dessert',
                  'Egg',
                  'Fried food',
                  'Meat',
                  'Noodles-Pasta',
                  'Rice',
                  'Seafood',
                  'Soup',
                  'Vegetable-Fruit']

    model = load_model(
        'epoch-29-train-0.84-test0.76.pth', categories=categories)

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

        sorted_probs, sorted_idx = torch.sort(
            image_pred_probs, descending=True)

        for i in sorted_idx.squeeze().numpy():
            class_label = categories[i]
            probability = float(image_pred_probs.squeeze()[i])
            st.write(f"{class_label}: {probability:.2f}")
            st.progress(probability)


if __name__ == '__main__':
    main()
