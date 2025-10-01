import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, Concatenate, Conv2DTranspose
from tensorflow.keras.applications import VGG16
import time
import io
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# -----------------------------------------------------------
# 1. Configuration & Styling
# -----------------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Analysis (VGG & UNet)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def set_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .result-box {
            padding: 15px;
            border-radius: 12px;
            margin-top: 15px;
            text-align: center;
        }
        .result-tumor {
            background-color: rgba(255, 99, 132, 0.15);
            border: 2px solid #FF6384;
        }
        .result-no-tumor {
            background-color: rgba(75, 192, 192, 0.15);
            border: 2px solid #4BC0C0;
        }
        .stTitle {
            font-size: 3em;
            color: #1E40AF;
        }
        </style>
    """, unsafe_allow_html=True)

set_styles()

# -----------------------------------------------------------
# 2. Model Architecture Definitions
# -----------------------------------------------------------

def unet_model(input_size=(128,128,3)):
    inputs = Input(input_size)
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3,3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)
    c5 = Conv2D(1024, (3,3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3,3), activation='relu', padding='same')(c5)
    u6 = Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3,3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3,3), activation='relu', padding='same')(c6)
    u7 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3,3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3,3), activation='relu', padding='same')(c7)
    u8 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3,3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3,3), activation='relu', padding='same')(c8)
    u9 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3,3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3,3), activation='relu', padding='same')(c9)
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    unet_arch = Model(inputs=[inputs], outputs=[outputs]) 
    return unet_arch
    
class DummyClassifier:
    def predict(self, x):
        return np.array([[0.1]])

class DummySegmentor:
    def predict(self, x):
        return np.array([np.zeros((128, 128, 1), dtype=np.float32)])

@st.cache_resource
def load_deep_learning_models():
    clf_model = None
    seg_model = None
    
    # 1. Load UNet Model (Segmentation)
    try:
        seg_model = load_model("unet_model.h5", custom_objects={'unet_model': unet_model})
    except Exception as e:
        st.error(f"⚠️ Failed to load UNet model. Using a dummy segmentor. Error: {e}")
        seg_model = DummySegmentor()
    
    # 2. Load VGG Model (Classification)
    try:
        base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers:
             layer.trainable = False 
             
        for layer in base_model.layers[-4:]:
            layer.trainable = True

        clf_model_arch = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ])
        
        clf_model_arch.load_weights("vgg_model.h5", skip_mismatch=False, by_name=True)
        clf_model = clf_model_arch
        
    except Exception as e:
        st.error(f"🔴 VGG Model (vgg_model.h5) loading failed. Please ensure the file is valid. Error: {e}")
        clf_model = DummyClassifier()

    return clf_model, seg_model

clf_model, seg_model = load_deep_learning_models()

# -----------------------------------------------------------
# 3. Main Processing Functions
# -----------------------------------------------------------

CLASSIFICATION_CLASSES = ["No Tumor", "Tumor"]
VGG_INPUT_SHAPE = (224, 224)
UNET_INPUT_SHAPE = (128, 128)

def preprocess_for_vgg(img_data):
    img_resized = cv2.resize(img_data, VGG_INPUT_SHAPE)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def preprocess_for_unet(img_data):
    img_resized_seg = cv2.resize(img_data, UNET_INPUT_SHAPE)
    img_array_seg = image.img_to_array(img_resized_seg)
    img_array_seg = np.expand_dims(img_array_seg, axis=0)
    return img_array_seg / 255.0

def generate_segmentation_mask(original_img, mask_raw):
    mask = mask_raw[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    colored_mask = np.zeros_like(original_img, dtype=np.uint8)
    colored_mask[:, :, 0] = mask_resized
    colored_mask[:, :, 1] = 0
    colored_mask[:, :, 2] = mask_resized
    
    blended_img = cv2.addWeighted(original_img, 0.7, colored_mask, 0.3, 0)
    return blended_img

# -----------------------------------------------------------
# 4. Streamlit UI Implementation
# -----------------------------------------------------------

st.markdown('<h1 class="stTitle">🧠 Brain Tumor Deep Learning Analysis</h1>', unsafe_allow_html=True)
st.markdown("---")

# Check model loading status
is_vgg_loaded = not isinstance(clf_model, DummyClassifier)
is_unet_loaded = not isinstance(seg_model, DummySegmentor)

# Create Tabs
tab1, tab2 = st.tabs(["Analysis", "Statistics"])

with tab1:
    st.markdown("""
        An advanced tool for analyzing Magnetic Resonance Imaging (MRI) scans to perform **Classification** (tumor detection) and **Segmentation** (precise location and size boundary).
        <br><br>
    """, unsafe_allow_html=True)

    if not is_vgg_loaded:
        st.error("⚠️ VGG Model (Classification) loading failed. Classification results will be unreliable.")

    if not is_unet_loaded:
        st.warning("UNet Model (Segmentation) loading failed. Segmentation feature is unavailable.")
        
    if is_vgg_loaded or is_unet_loaded:
        uploaded_file = st.file_uploader(
            "Upload MRI Image for Analysis:",
            type=["jpg", "png", "jpeg"]
        )

        if uploaded_file is not None:
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
            
            st.info("Image uploaded successfully. Running analysis...")
            
            st.image(original_img_rgb, caption="Original MRI Image", use_column_width=True)

            with st.spinner('⏳ Running Classification and Segmentation...'):
                time.sleep(1)

                # -------------------
                # 4.1. Classification
                # -------------------
                if is_vgg_loaded:
                    img_array_clf = preprocess_for_vgg(original_img_bgr)
                    pred_clf = clf_model.predict(img_array_clf)
                    probability_tumor = pred_clf[0][0]
                    
                    if probability_tumor > 0.5:
                        result_class = "Tumor"
                        probability = probability_tumor * 100
                    else:
                        result_class = "No Tumor"
                        probability = (1 - probability_tumor) * 100
                else:
                    result_class = "No Tumor"
                    probability = 0.0 

                # -------------------
                # 4.2. Segmentation
                # -------------------
                if is_unet_loaded:
                    img_array_seg = preprocess_for_unet(original_img_bgr)
                    mask_raw = seg_model.predict(img_array_seg)
                    blended_image = generate_segmentation_mask(original_img_bgr, mask_raw)
                    blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
                else:
                    blended_image_rgb = original_img_rgb

            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### 🩺 Classification Output")
                
                if is_vgg_loaded:
                    if result_class == "Tumor":
                        st.markdown(f"""
                            <div class='result-box result-tumor'>
                                <h2>⚠️ TUMOR DETECTED</h2>
                                <p>Confidence: **{probability:.2f}%**</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.warning("Please consult a specialist for confirmation and appropriate treatment.", icon="🚨")
                    else:
                         st.markdown(f"""
                            <div class='result-box result-no-tumor'>
                                <h2>✅ NO TUMOR DETECTED</h2>
                                <p>Confidence: **{probability:.2f}%**</p>
                            </div>
                        """, unsafe_allow_html=True)
                         st.success("The model did not detect a tumor. Always consult medical professionals.", icon="✅")
                else:
                    st.markdown(f"""
                        <div class='result-box result-no-tumor'>
                            <h2>🚫 CLASSIFICATION UNAVAILABLE</h2>
                            <p>VGG Model failed to load. Check console for errors.</p>
                        </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("#### 🩻 Segmentation Output")
                
                if is_unet_loaded:
                    st.image(blended_image_rgb, caption="Blended Tumor Mask (Purple Overlay)", use_column_width=True)
                    
                    if result_class == "Tumor":
                        st.markdown("""
                            **Localization:** The blended mask highlights the suspected tumor location on the MRI scan.
                        """)
                    else:
                        st.markdown("""
                            **Note:** As no tumor was classified, the mask may show minimal or uniform output.
                        """)
                else:
                    st.markdown(f"""
                        <div class='result-box result-tumor'>
                            <h2>🚫 SEGMENTATION UNAVAILABLE</h2>
                            <p>UNet Model failed to load. Check console for errors.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(blended_image_rgb, caption="Original Image (Fallback)", use_column_width=True)

with tab2:
    st.header("Model Performance Statistics")
    st.markdown("---")
    st.markdown("""
        This tab shows simulated performance metrics from the model training and evaluation phases. 
        Note: These are illustrative metrics and should be replaced with actual model evaluation data (e.g., from your Jupyter Notebook).
    """)

    # Simulated Data for Classification Metrics
    clf_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'VGG-16': [0.82, 0.81, 0.83, 0.82],
        'EfficientNetB0 (Original Training)' : [0.38, 0.15, 1.00, 0.26] # Using original low EfficientNet score from notebook as a contrast
    }).set_index('Metric')

    st.subheader("Classification Performance Comparison")
    st.dataframe(clf_metrics)

    # Plotting Classification Accuracy
    st.bar_chart(clf_metrics[['VGG-16']], height=350)
    
    st.markdown("---")

    # Simulated Data for Segmentation Metrics
    seg_metrics = pd.DataFrame({
        'Metric': ['Overall Accuracy', 'Loss (Binary Crossentropy)', 'IoU (Jaccard Index)'],
        'UNet Model': [0.9902, 0.0266, 0.85] 
    }).set_index('Metric')

    st.subheader("Segmentation (UNet) Performance")
    st.dataframe(seg_metrics)

    # Plotting Segmentation Metrics
    fig, ax = plt.subplots(figsize=(8, 4))
    seg_metrics['UNet Model'].plot(kind='barh', ax=ax, color=['#4BC0C0', '#FF6384', '#FFCE56'])
    ax.set_title('UNet Model Key Metrics')
    ax.set_xlabel('Score')
    st.pyplot(fig)


# -----------------------------------------------------------
# 5. Technical Notes
# -----------------------------------------------------------

st.sidebar.title("Technical Information")
st.sidebar.info("""
    This application utilizes:
    - **Streamlit** for the interactive front-end.
    - **VGG-16** for **Classification**.
    - **UNet** for **Segmentation**.
    - **OpenCV** and **NumPy** for image processing.
""")
