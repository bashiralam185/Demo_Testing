import streamlit as st
import numpy as np
import json
import pandas as pd
from PIL import Image
import torch
import cv2 
from torchvision import transforms
from skimage import measure
import matplotlib.pyplot as plt
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Get the absolute path to the Implementations directory
implementations_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Implementations'))
sys.path.append(implementations_dir)

from fpn_net import FPN, Bottleneck

# # Define available models and loss functions
# MODELS = [
#     "UNet", "LinkNet", "Deeplabv3Plus", "FCB Former", "FCN", "FPN", "HR-NET"
# ]
LOSS_FUNCTIONS = [
   'AsymmetricSimilarityLoss', 'BoundaryLoss', 'ComboLoss', 'DiceLoss'
]



def generate_heatmap_overlay(original_image, mask_image):
    # Convert images to numpy arrays
    original_image_np = np.array(original_image.convert("RGB"))
    mask_resized = mask_image.resize(original_image.size)
    mask_np = np.array(mask_resized).astype(np.float32)  # Ensure it's float for normalization

    # Normalize the mask (assuming binary or probability values)
    mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)  # Avoid division by zero

    # Apply 'jet' colormap using Matplotlib
    cmap = plt.get_cmap('jet')
    heatmap = cmap(mask_np)  # Returns RGBA (H, W, 4)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB

    # Ensure heatmap matches the original image size
    if heatmap.shape[:2] != original_image_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))

    # Blend the heatmap with the original image
    overlay = cv2.addWeighted(original_image_np, 0.7, heatmap, 0.5, 0)

    return Image.fromarray(overlay)



# Function to generate contour overlay
def generate_contour_overlay(original_image, mask_image):
    original_image_np = np.array(original_image.convert("RGB"))
    mask_resized = mask_image.resize(original_image.size)
    mask_np = np.array(mask_resized)
    contours = measure.find_contours(mask_np, 0.5)
    
    for contour in contours:
        contour = np.flip(contour, axis=1)
        for point in contour:
            x, y = point
            if x >= 0 and x < original_image_np.shape[1] and y >= 0 and y < original_image_np.shape[0]:
                original_image_np[int(y), int(x)] = [255, 0, 0]  

    return original_image_np




# Load model results from JSON
def load_model_results(loss_function_name):
    try:
        # Define the absolute path to the 'best-models' folder
        best_models_dir = os.path.join(os.path.dirname(__file__), 'results')
        # Construct the full path to the results file
        file_name = os.path.join(best_models_dir, f"FPN_{loss_function_name}_results.json")
        # Open and load the JSON file
        with open(file_name, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return None


# Load model from pickle
@st.cache_resource
def load_model(loss_function_name):
    # Define the absolute path to the 'best-models' folder
    best_models_dir = os.path.join(os.path.dirname(__file__), 'best-models')
    
    # Construct the full path to the model file for FPN
    model_path = os.path.join(best_models_dir, f"FPN{loss_function_name}.pth")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Clean up the checkpoint's state_dict (e.g., remove 'module.' if it's there)
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    
    # Initialize the FPN model and load the state_dict
    model = FPN(Bottleneck, [2, 2, 2, 2])  
    model.load_state_dict(checkpoint, strict=False)  # Allow missing keys if necessary
    
    # Move the model to the appropriate device (CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)



# Function to reset model when selection changes
def reset_model():
    if "model" in st.session_state:
        del st.session_state.model



# Make predictions
def make_prediction(model, image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    input_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_image)

    output_mask = torch.sigmoid(output) 
    output_mask = (output_mask > 0.5).float()

    output_mask_image = output_mask.squeeze().cpu().numpy()
    output_mask_image = Image.fromarray((output_mask_image * 255).astype(np.uint8))

    return output_mask_image




def generate_difference_overlay(gt_image, prediction):
    """
    Generates an overlay showing the difference between ground truth and prediction.
    
    - Green: True Positives (Correctly predicted tumor areas)
    - Red: False Negatives (Missed tumor areas where the ground truth indicates tumor but the model predicted non-tumor)
    - Blue: False Positives (Over-predicted areas where the ground truth says no tumor but the model predicted a tumor)
    """
    # Resize prediction to match the ground truth dimensions
    prediction_resized = prediction.resize(gt_image.size, Image.NEAREST)

    gt_array = np.array(gt_image.convert("L"))  # Convert to grayscale
    pred_array = np.array(prediction_resized.convert("L"))  # Convert resized prediction to grayscale

    # Normalize both to binary masks (assuming 255 is foreground, 0 is background)
    gt_binary = (gt_array > 127).astype(np.uint8)
    pred_binary = (pred_array > 127).astype(np.uint8)

    # Compute difference masks
    true_positive = (gt_binary & pred_binary)  # Green (Correctly predicted tumor areas)
    false_negative = ((gt_binary == 1) & (pred_binary == 0))  # Red (Missed tumor areas)
    false_positive = ((gt_binary == 0) & (pred_binary == 1))  # Blue (Over-predicted areas)

    # Calculate total pixels for each category
    total_tumor_area = np.sum(gt_binary)
    correct_tumor = np.sum(true_positive)
    over_predicted = np.sum(false_positive)
    missed_tumor = np.sum(false_negative)

    # Calculate percentages
    correct_percentage = (correct_tumor / total_tumor_area * 100) if total_tumor_area > 0 else 0
    over_predicted_percentage = (over_predicted / total_tumor_area * 100) if total_tumor_area > 0 else 0
    missed_percentage = (missed_tumor / total_tumor_area * 100) if total_tumor_area > 0 else 0

    # Create an RGB image for visualization
    difference_overlay = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
    difference_overlay[..., 1] = true_positive * 255  # Green
    difference_overlay[..., 0] = false_negative * 255  # Red
    difference_overlay[..., 2] = false_positive * 255  # Blue

    overlay_image = Image.fromarray(difference_overlay)
    
    # Display results in Streamlit
    st.write(f"✅ Correct Tumor (Green): {correct_tumor} pixels ({correct_percentage:.2f}%)")
    st.write(f"❌ Missed Tumor (Red): {missed_tumor} pixels ({missed_percentage:.2f}%)")
    st.write(f"🔵 Over-Predicted Tumor (Blue): {over_predicted} pixels ({over_predicted_percentage:.2f}%)")
    
    return overlay_image




# Streamlit UI
def main():
    uploaded_file = st.sidebar.file_uploader("Upload a Tumor Image (JPG Format)", type=["jpg", "jpeg", 'png'])

    st.title("Tumor Segmentation Tool")
    st.write("Select a model and loss function, upload a tumor image, and get the prediction.")

    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'overlay_image' not in st.session_state:
        st.session_state.overlay_image = None
    if 'show_overlap' not in st.session_state:
        st.session_state.show_overlap = False

    # Sidebar for model and loss function selection
    st.sidebar.header("Configuration")
    selected_model = 'FPN'
    selected_loss = st.sidebar.selectbox("Select a Loss Function", LOSS_FUNCTIONS)

    # Check if model needs to be reloaded
    if "prev_model" not in st.session_state or "prev_loss" not in st.session_state:
        st.session_state.prev_model = selected_model
        st.session_state.prev_loss = selected_loss

    # If user changes model or loss, reset model
    if st.session_state.prev_model != selected_model or st.session_state.prev_loss != selected_loss:
        reset_model()  # Clear model cache
        st.session_state.prev_model = selected_model
        st.session_state.prev_loss = selected_loss

    # Load model only if not cached
    if "model" not in st.session_state:
        st.session_state.model = load_model( selected_loss)

    # Load and display model performance results in a DataFrame based on selected model and loss function
    model_results = load_model_results(selected_loss)
    
    if model_results:
        df = pd.DataFrame([model_results])
        st.sidebar.subheader(f"{selected_model} - {selected_loss} Model Performance")
        st.sidebar.dataframe(df)

        options = st.sidebar.radio("Select an option", ["loss_history", "dsc_history"])
        if options == "loss_history":
            st.sidebar.line_chart(df["loss_history"][0])

        else:
            st.sidebar.line_chart(df["dsc_history"][0])

    else:
        st.sidebar.write("Model performance results not found for this combination.")


    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.image = image
        col1, col2 = st.columns(2)
        col1_3d, col2_3d = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load model
        if st.session_state.model is None:
            st.session_state.model = load_model(selected_loss)

        # Prediction and Visualization
        predict_button, overlap_button, colored_overlay_button = st.columns([1, 1, 1])

        with predict_button:
            if st.button("Predict"):
                if st.session_state.model is not None:
                    with st.spinner("Predicting..."):
                        prediction = make_prediction(st.session_state.model, st.session_state.image)
                    st.session_state.prediction = prediction
                    st.success("Prediction completed!")
                    with col2:
                        st.image(prediction, caption="Segmentation Output", use_column_width=True)

        with overlap_button:
            if st.button("Show Overlay"):
                if st.session_state.prediction is not None:
                    st.session_state.overlay_image = generate_heatmap_overlay(st.session_state.image, st.session_state.prediction)
                    with col2:
                        st.image(st.session_state.overlay_image, caption="Overlayed Tumor Region", use_column_width=True)
        with colored_overlay_button:
            if st.button("Show Contour Overlay"):
                if st.session_state.prediction is not None:
                    st.session_state.overlay_image = generate_contour_overlay(st.session_state.image, st.session_state.prediction)
                    with col2:
                        st.image(st.session_state.overlay_image, caption="Contour Overlay", use_column_width=True)

    # campare with ground truth
    ground_truth = st.sidebar.file_uploader("Upload Ground Truth Image (JPG Format)", type=["jpg", "jpeg", 'png'])

    if ground_truth is not None:
        gt_image = Image.open(ground_truth)
        st.session_state.gt_image = gt_image
        col3, col4 = st.columns(2)

        with col3:
            st.image(gt_image, caption="Ground Truth Image", use_column_width=True)

        if st.session_state.prediction is not None:
            overlap_button, colored_overlay_button, difference_overlay_button = st.columns([1, 1, 1])

            with overlap_button:
                if st.button("Show Overlay", key="overlay_button"):
                    st.session_state.overlay_image = generate_heatmap_overlay(gt_image, st.session_state.prediction)
                    with col4:
                        st.image(st.session_state.overlay_image, caption="Overlayed Tumor Region", use_column_width=True)

            with colored_overlay_button:
                if st.button("Show Contour Overlay", key="contour_overlay_button"):
                    st.session_state.overlay_image = generate_contour_overlay(gt_image, st.session_state.prediction)
                    with col4:
                        st.image(st.session_state.overlay_image, caption="Contour Overlay", use_column_width=True)

            with difference_overlay_button:
                if st.button("Show Difference Overlay", key="difference_overlay_button"):
                    st.session_state.overlay_image = generate_difference_overlay(gt_image, st.session_state.prediction)
                    with col4:
                        st.image(st.session_state.overlay_image, caption="Difference Overlay", use_column_width=True)

if __name__ == "__main__":
    main()