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
import zipfile
import tempfile
import io
from datetime import datetime
from fpdf import FPDF
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

# Path configurations
implementations_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Implementations'))
sys.path.append(implementations_dir)
from fpn_net import FPN, Bottleneck

LOSS_FUNCTIONS = [
   'AsymmetricSimilarityLoss', 'BoundaryLoss', 'ComboLoss', 'DiceLoss'
]

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



def generate_heatmap_overlay(original_image, mask_image):
    original_image_np = np.array(original_image.convert("RGB"))
    mask_resized = mask_image.resize(original_image.size)
    mask_np = np.array(mask_resized).astype(np.float32)
    mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)

    cmap = plt.get_cmap('jet')
    heatmap = cmap(mask_np)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)

    if heatmap.shape[:2] != original_image_np.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_image_np.shape[1], original_image_np.shape[0]))

    overlay = cv2.addWeighted(original_image_np, 0.7, heatmap, 0.5, 0)
    return Image.fromarray(overlay)

def generate_contour_overlay(original_image, mask_image):
    original_image_np = np.array(original_image.convert("RGB"))
    mask_resized = mask_image.resize(original_image.size)
    mask_np = np.array(mask_resized)
    contours = measure.find_contours(mask_np, 0.5)
    
    for contour in contours:
        contour = np.flip(contour, axis=1)
        for point in contour:
            x, y = point
            if 0 <= x < original_image_np.shape[1] and 0 <= y < original_image_np.shape[0]:
                original_image_np[int(y), int(x)] = [255, 0, 0]  
    return original_image_np

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

def load_model_results(loss_function_name):
    try:
        best_models_dir = os.path.join(os.path.dirname(__file__), 'results')
        file_name = os.path.join(best_models_dir, f"FPN_{loss_function_name}_results.json")
        with open(file_name, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model(loss_function_name):
    best_models_dir = os.path.join(os.path.dirname(__file__), 'best-models')
    model_path = os.path.join(best_models_dir, f"FPN{loss_function_name}.pth")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    
    model = FPN(Bottleneck, [2, 2, 2, 2])  
    model.load_state_dict(checkpoint, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

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
    return Image.fromarray((output_mask_image * 255).astype(np.uint8))

def process_single_image(model, image, mask=None):
    """Process a single image and return all visualizations"""
    viz_dict = {}
    
    # Make prediction
    prediction = make_prediction(model, image)
    viz_dict['prediction'] = prediction
    
    # Generate overlays
    viz_dict['heatmap_overlay'] = generate_heatmap_overlay(image, prediction)
    viz_dict['contour_overlay'] = Image.fromarray(generate_contour_overlay(image, prediction))
    
    # If mask provided, generate difference overlay
    if mask is not None:
        viz_dict['difference_overlay'] = generate_difference_overlay(mask, prediction)
    
    return viz_dict

def process_image_batch(model, image_files, mask_files, temp_dir):
    """Process batch and return paths to all generated visualizations"""
    viz_paths = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
        try:
            status_text.text(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_file.name)}")
            progress_bar.progress((i + 1) / len(image_files))
            
            # Load images
            image = Image.open(img_file)
            mask = Image.open(mask_file)
            base_name = os.path.splitext(os.path.basename(img_file.name))[0]
            
            # Generate all visualizations
            viz_dict = process_single_image(model, image, mask)
            viz_dict['original'] = image
            viz_dict['ground_truth'] = mask
            
            # Save each visualization
            for viz_name, viz_img in viz_dict.items():
                try:
                    if isinstance(viz_img, np.ndarray):
                        viz_img = Image.fromarray(viz_img)
                    viz_path = os.path.join(temp_dir, f"{base_name}_{viz_name}.png")
                    viz_img.save(viz_path)
                    viz_paths.append(viz_path)
                except Exception as e:
                    st.error(f"Error saving {viz_name}: {str(e)}")
                    
        except Exception as e:
            st.error(f"Error processing {img_file.name}: {str(e)}")
    
    return viz_paths

# def create_pdf_report(visualization_paths, output_path):
    """Create PDF report with matplotlib's PdfPages"""
    with PdfPages(output_path) as pdf:
        # Group visualizations by image
        image_groups = {}
        for path in visualization_paths:
            try:
                base = os.path.basename(path)
                image_name = '_'.join(base.split('_')[:-1])  # Remove visualization type
                viz_type = base.split('_')[-1].split('.')[0]
                
                if image_name not in image_groups:
                    image_groups[image_name] = {}
                image_groups[image_name][viz_type] = path
            except:
                continue
        
        # Create a page for each image
        for image_name, viz_dict in image_groups.items():
            # Skip if we don't have all required visualizations
            required_viz = ['original', 'ground_truth', 'prediction', 
                          'heatmap_overlay', 'contour_overlay', 'difference_overlay']
            if not all(viz in viz_dict for viz in required_viz):
                continue
            
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f"Image: {image_name}", fontsize=16)
            
            # Define the grid layout
            gs = fig.add_gridspec(2, 3)
            
            # Row 1: Original, Prediction, Heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(mpimg.imread(viz_dict['original']))
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(mpimg.imread(viz_dict['prediction']), cmap='gray')
            ax2.set_title("Prediction")
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(mpimg.imread(viz_dict['heatmap_overlay']))
            ax3.set_title("Heatmap Overlay")
            ax3.axis('off')
            
            # Row 2: Ground Truth, Contour, Difference
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(mpimg.imread(viz_dict['ground_truth']), cmap='gray')
            ax4.set_title("Ground Truth")
            ax4.axis('off')
            
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(mpimg.imread(viz_dict['contour_overlay']))
            ax5.set_title("Contour Overlay")
            ax5.axis('off')
            
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.imshow(mpimg.imread(viz_dict['difference_overlay']))
            ax6.set_title("Difference Overlay")
            ax6.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Add metadata
        metadata = pdf.infodict()
        metadata['Title'] = "Tumor Segmentation Report"
        metadata['Author'] = "Medical Imaging App"
        metadata['CreationDate'] = datetime.now()

def main():
    st.title("Tumor Segmentation Tool")
    st.write("Upload images to generate tumor segmentation visualizations")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None

    # Sidebar configuration
    st.sidebar.header("Configuration")
    selected_loss = st.sidebar.selectbox("Select Loss Function", LOSS_FUNCTIONS)
    
    # Load model
    if st.session_state.model is None:
        st.session_state.model = load_model(selected_loss)

    # Load and display model performance results
    model_results = load_model_results(selected_loss)
    if model_results:
        df = pd.DataFrame([model_results])
        st.sidebar.subheader(f"{selected_loss} Model Performance")
        st.sidebar.dataframe(df)

        options = st.sidebar.radio("Select an option", ["loss_history", "dsc_history"])
        if options == "loss_history":
            st.sidebar.line_chart(df["loss_history"][0])
        else:
            st.sidebar.line_chart(df["dsc_history"][0])
    else:
        st.sidebar.write("Model performance results not found for this combination.")

    # === TABS ===
    tab1, tab2 = st.tabs(["📷 Single Image", "🗂️ Batch ZIP Upload"])

    with tab1:
        st.header("Single Image Processing")
        uploaded_file = st.file_uploader("Upload Tumor Image", type=["jpg", "jpeg", "png"], key="single_img")
        ground_truth = st.file_uploader("Upload Ground Truth Mask (optional)", type=["jpg", "jpeg", "png"], key="single_gt")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            mask = Image.open(ground_truth) if ground_truth else None
            
            cols = st.columns(2)
            cols[0].image(image, caption="Input Image", use_column_width=True)
            
            if st.button("Process Image"):
                with st.spinner("Generating visualizations..."):
                    viz_dict = process_single_image(st.session_state.model, image, mask)
                    st.session_state.prediction = viz_dict['prediction']
                    
                    st.subheader("All Visualizations")

                    fig = plt.figure(figsize=(15, 10))
                    fig.suptitle("Tumor Segmentation Visualizations", fontsize=16)
                    gs = fig.add_gridspec(2, 3)

                    # Row 1
                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.imshow(image)
                    ax1.set_title("Original Image")
                    ax1.axis('off')

                    ax2 = fig.add_subplot(gs[0, 1])
                    ax2.imshow(viz_dict['prediction'], cmap='gray')
                    ax2.set_title("Prediction")
                    ax2.axis('off')

                    ax3 = fig.add_subplot(gs[0, 2])
                    ax3.imshow(viz_dict['heatmap_overlay'])
                    ax3.set_title("Heatmap Overlay")
                    ax3.axis('off')

                    # Row 2
                    ax4 = fig.add_subplot(gs[1, 0])
                    if mask:
                        ax4.imshow(mask, cmap='gray')
                        ax4.set_title("Ground Truth")
                    else:
                        ax4.set_title("Ground Truth (Not Provided)")
                    ax4.axis('off')

                    ax5 = fig.add_subplot(gs[1, 1])
                    ax5.imshow(viz_dict['contour_overlay'])
                    ax5.set_title("Contour Overlay")
                    ax5.axis('off')

                    ax6 = fig.add_subplot(gs[1, 2])
                    if 'difference_overlay' in viz_dict:
                        ax6.imshow(viz_dict['difference_overlay'])
                        ax6.set_title("Difference Overlay")
                    else:
                        ax6.set_title("Difference Overlay (N/A)")
                    ax6.axis('off')

                    st.pyplot(fig)

    with tab2:
        st.header("Batch Processing")
        batch_zip = st.file_uploader("Upload ZIP file with images and masks", type=["zip"], key="zip_upload")

        if batch_zip is not None:
            with st.spinner("Processing batch..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    with zipfile.ZipFile(batch_zip, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    all_files = []
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                all_files.append(os.path.join(root, file))
                    
                    image_files = []
                    mask_files = []

                    for file_path in all_files:
                        filename = os.path.basename(file_path).lower()
                        if '_mask' in filename:
                            base_filename = filename.split('_mask')[0]
                            for img_path in all_files:
                                img_name = os.path.basename(img_path).lower()
                                if img_name.startswith(base_filename) and '_mask' not in img_name:
                                    image_files.append(img_path)
                                    mask_files.append(file_path)
                                    break
                    
                    if not image_files:
                        for file_path in all_files:
                            filename = os.path.basename(file_path).lower()
                            if 'mask' in filename or 'seg' in filename or 'gt' in filename:
                                base_name = filename.replace('mask', '').replace('seg', '').replace('gt', '')
                                base_name = base_name.split('.')[0].strip('_')
                                for img_path in all_files:
                                    img_name = os.path.basename(img_path).lower().split('.')[0]
                                    if img_name == base_name and file_path != img_path:
                                        image_files.append(img_path)
                                        mask_files.append(file_path)
                                        break

                    if not image_files:
                        st.error("Could not pair images with masks. Please check naming conventions.")
                    else:
                        viz_paths = process_image_batch(
                            st.session_state.model,
                            [open(img, 'rb') for img in image_files],
                            [open(mask, 'rb') for mask in mask_files],
                            temp_dir
                        )
                        
                        # pdf_path = os.path.join(temp_dir, "tumor_segmentation_report.pdf")
                        # create_pdf_report(viz_paths, pdf_path)
                        
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # with open(pdf_path, 'rb') as f:
                            #     zip_file.writestr("tumor_segmentation_report.pdf", f.read())
                            for viz_path in viz_paths:
                                zip_file.write(viz_path, os.path.basename(viz_path))
                        zip_buffer.seek(0)
                        
                        st.success(f"Processed {len(image_files)} images successfully!")
                        
                        col1, col2 = st.columns(2)
                        # with col1:
                        #     with open(pdf_path, 'rb') as f:
                        #         st.download_button(
                        #             label="Download PDF Report",
                        #             data=f,
                        #             file_name="tumor_segmentation_report.pdf",
                        #             mime="application/pdf"
                        #         )
                        with col1:
                            st.download_button(
                                label="Download All Files (ZIP)",
                                data=zip_buffer,
                                file_name="tumor_visualizations.zip",
                                mime="application/zip"
                            )

                        st.subheader("First Image Results Preview")
                        first_image_base = os.path.splitext(os.path.basename(image_files[0]))[0]
                        first_image_viz = [p for p in viz_paths if os.path.basename(p).startswith(first_image_base)]
                        
                        preview_cols = st.columns(3)
                        for i, viz_path in enumerate(first_image_viz[:6]):
                            preview_cols[i%3].image(
                                Image.open(viz_path),
                                caption=os.path.basename(viz_path),
                                use_column_width=True
                            )

if __name__ == "__main__":
    main()