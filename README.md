# COVID-19-Grading-System-
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
