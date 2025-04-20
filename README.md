## Comprehensive Guide to Loss Functions used in Image Segmentation

In medical image segmentation, selecting an appropriate loss function is critical to achieving optimal model performance, particularly in tasks that involve challenging datasets with severe class imbalance and blurred anatomical boundaries. This tool is part of the research presented in our paper(PaperLink), where we perform a comprehensive theoretical review of 25 loss functions commonly used in medical image segmentation. The study emphasizes the impact of loss functions on model behavior, especially in the context of brain tumor segmentation.

To bridge theory and practice, we introduce an [interactive Streamlit-based](https://demotesting.streamlit.app/) platform that allows real-time experimentation with all 25 loss functions. The tool supports both single-image and batch processing modes, providing visual diagnostics, performance metrics, and overlays for comparing model predictions against ground truth data. The system is built upon a consistent Feature Pyramid Network (FPN) architecture, trained on multiple cancer imaging datasets to facilitate a fair comparison of the loss functions in segmentation tasks.

This tool is a valuable resource for researchers and clinicians, enabling the exploration, comparison, and fine-tuning of loss functions for various medical image segmentation applications. The tool is deployed and can be acces from this link: [Testing Loss Functions For Segmentation](https://demotesting.streamlit.app/).

#### Paper Reference
- Title: Comprehensive Guide to Loss Functions used in Image Segmentation
- Authors: Masa Cirkovic, Mete Harun Akcay, Bashir Alam, Md Kaf Shahrier, Sebastien Lafond, Hergys Rexha, Kurt Benke, Sepinoud Azimi, and Janan Arslan
- Conference/Journal: 
- DOI: 

### Key Features
- **Loss Function Comparison**: The tool supports 25 loss functions, allowing users to compare their performance across multiple medical imaging tasks, with a special focus on brain tumor segmentation.
- **Interactive Platform**: Built with Streamlit, the platform provides a user-friendly interface for experimenting with different loss functions in real time.
- Segmentation Visualization: The tool generates various visual outputs, including:
    - Segmentation masks (predictions)
    - Heatmap overlays to highlight tumor areas
    - Contour overlays to visualize tumor boundaries
    - Difference maps between predicted and ground truth masks, highlighting true positives, false positives, and false negatives.
- **Batch Processing**: Users can upload a ZIP file containing a batch of images and their corresponding ground truth masks for efficient processing.
- **Performance Metrics**: For each processed image, the tool provides performance metrics such as Dice Similarity Coefficient (DSC), Intersection-over-Union (IoU), and loss history graphs.
- **Pre-trained Models**: The system includes pre-trained models for each loss function, trained on four open-access cancer imaging datasets, allowing users to test the models out of the box.

### Technology Stack
```bash
- Programming Language: Python 3.x
- Libraries and Frameworks:
    - Streamlit
    - PyTorch
    - TorchVision
    - OpenCV
    - Matplotlib
    - NumPy and Pandas
    - json
    - zipfile
```
### Installation and Setup Locally

##### 1. Clone the Repository

```bash
https://github.com/bashiralam185/Loss_functions_guide.git
cd Loss_functions_guide
```
##### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
##### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```
##### 4. Run the Application

```bash
streamlit run user_interface.py
```
Once the app is running, you can access it through your browser at ***http://localhost:8501***.

### Input and output specifications

##### Input

- **Single Image**: Upload a single medical image (e.g., .png, .jpg, .jpeg) for segmentation. Optionally, provide the corresponding ground truth mask to compare predictions.
- **Batch Processing**: Upload a ZIP file containing multiple images and their corresponding ground truth masks. Ensure that filenames match (e.g., image1.png and image1_mask.png).

##### Output

For each processed image, the tool generates the following outputs:
- **Segmentation Masks**: Predicted tumor regions saved as image_prediction.png.
- **Heatmap Overlays**: Color-coded regions indicating areas of high model confidence for tumor presence.
- **Contour Overlays**: Visual boundaries that highlight tumor outlines in the original image.
- **Difference Overlay**: Visual comparison showing regions of true positives, false positives, and false negatives.
- **Performance Metrics**: Dice Similarity Coefficient (DSC) and Intersection over Union (IoU) scores are displayed for each image processed.

All outputs are compressed into a ZIP archive for easy download.

## Citation
If you use this tool in your research, please cite the following:
```bash
```
