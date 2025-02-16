# COVID-19-Grading-System-
Step 1 # First Step Install Libraries
1. import numpy as np
2. import os
3. import cv2
4. import tensorflow as tf
5. from tensorflow.keras.models import Model
6. from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization, ReLU, Input, Dense, Flatten
7. from tensorflow.keras.applications import ResNet18
8. from tensorflow.keras.optimizers import Adam
9. from tensorflow.keras.losses import BinaryCrossentropy
10. from sklearn.decomposition import PCA
11. from sklearn.naive_bayes import GaussianNB
12. from sklearn.preprocessing import StandardScaler
13. from tensorflow.keras.preprocessing.image import ImageDataGenerator
14. from scipy.stats import entropy

Step 2 # Volumetric Analysis
1. import os
2. import cv2
3. import numpy as np

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

axial_imgs, coronal_imgs, sagittal_imgs = preprocess_ct_images(CT_IMAGE_PATH)
axial_scores = compute_volumetric_scores(axial_imgs)
coronal_scores = compute_volumetric_scores(coronal_imgs)
sagittal_scores = compute_volumetric_scores(sagittal_imgs)
axial_labels = assign_severity_labels(axial_scores)
coronal_labels = assign_severity_labels(coronal_scores)
sagittal_labels = assign_severity_labels(sagittal_scores)
print("Axial Labels:", axial_labels)
print("Coronal Labels:", coronal_labels)
print("Sagittal Labels:", sagittal_labels)

# Classification Model
# Path of Dataset
image_path = "/path/to/ct_slices"

# Extract features using ResNet-18
def extract_features(images):
    base_model = ResNet18(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    model = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
    
    images = np.expand_dims(images, axis=-1)
    features = model.predict(images)
    features = features.reshape(features.shape[0], -1)  # Flatten
    return features

# Select features using HHO
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
# Train classifier
classifier = classify_features(selected_features, labels)

# Proposed Unet Segmentation Model
  import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# Insert the path of the segmentation dataset
image_path = "/path/to/images"  # Change this to the actual path
mask_path = "/path/to/masks"    # Change this to the actual path

# Function to compute Dice coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Function to preprocess images and masks
def preprocess_ct_images(image_path, mask_path, img_size=(256, 256)):
    images, masks = [], []

    for img_name in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size) / 255.0  # Normalize
        images.append(img)

        mask = cv2.imread(os.path.join(mask_path, img_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size) / 255.0  # Normalize
        masks.append(mask)

    return np.array(images).reshape(-1, img_size[0], img_size[1], 1), np.array(masks).reshape(-1, img_size[0], img_size[1], 1)

# Function to build a simple U-Net model
def build_unet(input_shape=(256, 256, 1)):
    inputs = tf.keras.layers.Input(input_shape)
    
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

    up1 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv3)
    concat1 = tf.keras.layers.Concatenate()([up1, conv2])
    conv4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat1)
    conv4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)

    up2 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv4)
    concat2 = tf.keras.layers.Concatenate()([up2, conv1])
    conv5 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(concat2)
    conv5 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv5)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = tf.keras.Model(inputs, outputs)
    return model

# Function to train the segmentation model
def train_segmentation_model(image_path, mask_path):
    images, masks = preprocess_ct_images(image_path, mask_path)

    train_datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
    val_datagen = ImageDataGenerator()

    unet_model = build_unet()
    unet_model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=[dice_coefficient])

    history = unet_model.fit(
        train_datagen.flow(images, masks, batch_size=8),
        validation_data=val_datagen.flow(images, masks, batch_size=8),
        epochs=100
    )

    return unet_model, history

# Train the segmentation model
segmentation_model, training_history = train_segmentation_model(image_path, mask_path)


    unet_model.save("unet_ct_segmentation.h5")
    return unet_model

#Description 

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
