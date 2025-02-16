# COVID-19-Grading-System-
# First Step Install Libraries
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization, ReLU, Input, Dense, Flatten
from tensorflow.keras.applications import ResNet18
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.stats import entropy
# Volumetric Analysis
import os
import cv2
import numpy as np

# Define the path where CT images are stored
CT_IMAGE_PATH = "/path/to/ct_images"  # Change this to the actual path of your dataset

def preprocess_ct_images(ct_path, img_size=(256, 256)):
    axial_images = []
    coronal_images = []
    sagittal_images = []
    
    for img_name in os.listdir(ct_path):
        img_path = os.path.join(ct_path, img_name)
        
        # Check if the file is an image
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.dcm')):
            continue 
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size) / 255.0  # Normalize
        
        if "axial" in img_name.lower():
            axial_images.append(img)
        elif "coronal" in img_name.lower():
            coronal_images.append(img)
        elif "sagittal" in img_name.lower():
            sagittal_images.append(img)

    return np.array(axial_images), np.array(coronal_images), np.array(sagittal_images)

def compute_volumetric_scores(images):
    scores = []
    for img in images:
        score = np.sum(img) / (img.shape[0] * img.shape[1])  # Compute intensity score
        scores.append(score)
    return np.array(scores)

def assign_severity_labels(scores):
    labels = []
    for score in scores:
        if score < 0.3:
            labels.append("Healthy")
        elif score < 0.6:
            labels.append("Mild")
        else:
            labels.append("Moderate")
    return np.array(labels)

# Load and process the CT images
axial_imgs, coronal_imgs, sagittal_imgs = preprocess_ct_images(CT_IMAGE_PATH)

# Compute volumetric scores
axial_scores = compute_volumetric_scores(axial_imgs)
coronal_scores = compute_volumetric_scores(coronal_imgs)
sagittal_scores = compute_volumetric_scores(sagittal_imgs)

# Assign severity labels
axial_labels = assign_severity_labels(axial_scores)
coronal_labels = assign_severity_labels(coronal_scores)
sagittal_labels = assign_severity_labels(sagittal_scores)

# Print results
print("Axial Labels:", axial_labels)
print("Coronal Labels:", coronal_labels)
print("Sagittal Labels:", sagittal_labels)
# Classification Model
# Path of Dataset
image_path = "/path/to/ct_slices"
axial_images, coronal_images, sagittal_images = preprocess_ct_images(image_path)

# Compute severity scores
scores = compute_volumetric_scores(axial_images)
labels = assign_severity_labels(scores)

# Extract features using ResNet-18
features = extract_features(axial_images)

# Select features using HHO
selected_features = hho_feature_selection(features)

# Train classifier
classifier = classify_features(selected_features, labels)
def extract_features(images):
    base_model = ResNet18(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
    
    images = np.expand_dims(images, axis=-1)
    features = model.predict(images)
    features = features.reshape(features.shape[0], -1)  # Flatten
    return features
    import numpy as np
from sklearn.preprocessing import StandardScaler
from hho import HarrisHawksOptimization  # Assuming you have an HHO implementation

def hho_feature_selection(features, num_selected=446):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    def fitness_function(selected_features):
        # Define a fitness function based on classification accuracy or variance explained
        return -np.var(selected_features)  # Example: Maximizing variance
    
    hho = HarrisHawksOptimization(fitness_function, dim=features.shape[1], num_agents=30, max_iter=100)
    best_solution = hho.run()
    
    selected_indices = np.argsort(best_solution)[:num_selected]
    selected_features = features_scaled[:, selected_indices]
    
    return selected_features
    def classify_features(features, labels):
    clf = GaussianNB()
    clf.fit(features, labels)
    return clf
    # Segmentation Model
    def build_unet(input_shape=(256, 256, 1), filters=16, dropout_rate=0.05):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(filters, (3, 3), activation=None, padding="same")(inputs)
    c1 = BatchNormalization()(c1)
    c1 = ReLU()(c1)
    c1 = Dropout(dropout_rate)(c1)
    c1 = Conv2D(filters, (3, 3), activation=None, padding="same")(c1)
    c1 = BatchNormalization()(c1)
    c1 = ReLU()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(filters * 2, (3, 3), activation=None, padding="same")(p1)
    c2 = BatchNormalization()(c2)
    c2 = ReLU()(c2)
    c2 = Dropout(dropout_rate)(c2)
    c2 = Conv2D(filters * 2, (3, 3), activation=None, padding="same")(c2)
    c2 = BatchNormalization()(c2)
    c2 = ReLU()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(filters * 4, (3, 3), activation=None, padding="same")(p2)
    c3 = BatchNormalization()(c3)
    c3 = ReLU()(c3)
    c3 = Dropout(dropout_rate)(c3)
    c3 = Conv2D(filters * 4, (3, 3), activation=None, padding="same")(c3)
    c3 = BatchNormalization()(c3)
    c3 = ReLU()(c3)

    # Decoder
    u4 = Conv2DTranspose(filters * 2, (2, 2), strides=(2, 2), padding="same")(c3)
    u4 = concatenate([u4, c2])
    c4 = Conv2D(filters * 2, (3, 3), activation=None, padding="same")(u4)
    c4 = BatchNormalization()(c4)
    c4 = ReLU()(c4)
    c4 = Dropout(dropout_rate)(c4)
    c4 = Conv2D(filters * 2, (3, 3), activation=None, padding="same")(c4)
    c4 = BatchNormalization()(c4)
    c4 = ReLU()(c4)

    u5 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(c4)
    u5 = concatenate([u5, c1])
    c5 = Conv2D(filters, (3, 3), activation=None, padding="same")(u5)
    c5 = BatchNormalization()(c5)
    c5 = ReLU()(c5)
    c5 = Dropout(dropout_rate)(c5)
    c5 = Conv2D(filters, (3, 3), activation=None, padding="same")(c5)
    c5 = BatchNormalization()(c5)
    c5 = ReLU()(c5)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c5)

    model = Model(inputs, outputs)
    return model

# ---------------- TRAINING SEGMENTATION MODEL ---------------- #
#Instert the Path of Segmentation Dataset
mask_path = "/path/to/masks"
segmentation_model = train_segmentation_model(image_path, mask_path)

def train_segmentation_model(image_path, mask_path):
    images, masks = preprocess_ct_images(image_path)

    train_datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
    val_datagen = ImageDataGenerator()

    unet_model = build_unet()
    unet_model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=["accuracy"])

    history = unet_model.fit(
        train_datagen.flow(images, masks, batch_size=8),
        validation_data=val_datagen.flow(images, masks, batch_size=8),
        epochs=100
    )

    unet_model.save("unet_ct_segmentation.h5")
    return unet_model


Validation of COVID-19 Grading System based on Harris Hawks Optimization (HHO) and Variational Quantum Classifier using JLDS-2024
A method is proposed for classifying and segmenting COVID-19 lesions. First, preprocessing is performed on axial, coronial, and sagittal views of CT slices using statistical methods based on volumetric analysis to compute the scores. These are used to assign the grades based on the severity level of each lung lobe such as mild, moderate, and healthy. 

![image](https://github.com/user-attachments/assets/dbcf9528-994c-473c-ac61-d1bbd2bfcdf5)


The graded slices are input to the proposed classification model, and N×1000 features are extracted at fully connected (FC-1000) layer of the pre-trained ResNet-18 model. Out of these N×446 features are selected through the Harris Hawks Optimization (HHO) model. Finally, classification is performed using variational quantum, neural network, and Naïve Bayes classifiers. To segment the classified images, fine-tuned U-net model is designed having 47 layers such as 1 input, 5 dropout, 11 conv, 9 ReLU, 9 Batch-normalization, 4 Conv2DTranspo, 4 Concatenate, and 4 max-pooling. The model is trained from scratch based on optimal hyperparameters such as binary cross-entropy loss, 16 filter size, 0.005 drop-out, 8 batch size, and Adam optimizer.
	The one major contribution is to prepare a local dataset, in which preprocessing is performed using volumetric analysis based on statistical methods such as mean and variance, and grades are assigned on the left/right lung lobes based on radiological scoring criteria.

	The second contribution is, the proposed classification model where features are extracted using the FC-1000 layer of the pre-trained ResNet-18 model, and the best features are selected using the HHO method. Finally based on the selected features and optimum parameters of variational quantum, NB and NN classifiers. 
![image](https://github.com/user-attachments/assets/212647e7-aaae-42d6-b5bc-2620bab8a463)


	The third contribution is that classified images are passed as input to the proposed U-net model containing selected 47 layers and trained from scratch on the optimal hyperparameters, such as the Adam optimizer, binary cross-entropy loss, 16 filter size, and 0.005 drop-out layers, to segment the infected region more accurately.
![image](https://github.com/user-attachments/assets/1bfd1566-1ec7-4030-a3d9-1b7ebabf02ad)

Benchmark Datasets
UCSD-AI4H is a public classification dataset and contains 812 pictures of 216 patients [42]. The segmentation dataset contains 100 slices of 100 patients with ground annotation masks [43]. 
[42]	X. Yang, X. He, J. Zhao, Y. Zhang, S. Zhang, and P. Xie, "COVID-CT-dataset: a CT scan dataset about COVID-19," arXiv preprint arXiv:2003.13865, 2020.
[43]	"http://medicalsegmentation.com/covid19/ accessed by 23/12/2022."
