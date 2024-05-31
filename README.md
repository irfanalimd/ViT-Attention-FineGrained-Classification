# ViT-Attention-FineGrained-Classification
This project focuses on fine-grained image classification using the Vision Transformer (ViT) architecture enhanced with self-attention and hierarchical attention mechanisms. 
The model is trained and evaluated on the CUB-200-2011 dataset, achieving a testing accuracy of 77%.

Fine-grained image classification is a challenging task that requires distinguishing between visually similar subcategories within a broader category. This project leverages the power of the Vision Transformer (ViT) architecture, along with attention mechanisms, to tackle this problem effectively.

The model is trained and evaluated on the CUB-200-2011 dataset, which consists of 11,788 images belonging to 200 bird species. The dataset is widely used for fine-grained image classification tasks and provides a challenging benchmark for evaluating the performance of our model.

## Model Architecture
The core of the model is based on the Vision Transformer (ViT) architecture, which has shown impressive performance in various computer vision tasks. The ViT model is initialized with pre-trained weights from the 'vit_base_patch16_224' model, leveraging transfer learning to benefit from the knowledge gained from large-scale datasets.
To enhance the model's ability to capture fine-grained details, we incorporate self-attention and hierarchical attention mechanisms. These attention modules allow the model to focus on relevant regions and capture hierarchical relationships within the images, leading to improved classification performance.

## Training Techniques
Several training techniques are employed to optimize the model's performance and generalization ability:

* Dropout layers with a rate of 0.5 are added after the attention modules to prevent overfitting.
* L2 regularization is applied in the loss function, and weight decay is used in the Adam optimizer to regularize the model further.
* A learning rate scheduler (ReduceLROnPlateau) is utilized to adaptively adjust the learning rate based on the validation loss, ensuring optimal convergence.
* Gradient clipping with a max norm of 1.0 is implemented to stabilize the training process and prevent exploding gradients.
* Data parallelism is employed to distribute the model across multiple GPUs, enabling efficient training on large-scale datasets.

## Results
The trained model achieves impressive results on the CUB-200-2011 dataset:

* Testing Accuracy: 77%
*Validation Precision: 0.7632
* Validation Recall: 0.7581
* Validation F1 Score: 0.7606

These metrics demonstrate the effectiveness of the ViT-based model with attention mechanisms in capturing fine-grained details and accurately classifying bird species.
Installation
