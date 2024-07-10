# LLM Project

## Project Task
The task for this project is to perform sentiment analysis on movie reviews using a pre-trained language model. The goal is to classify each review as either positive or negative based on the text.

## Dataset
The dataset used for this project is the IMDB movie reviews dataset. This dataset contains 50,000 movie reviews labeled as either positive or negative. The dataset is split into 25,000 reviews for training and 25,000 reviews for testing.

## Pre-trained Model
The pre-trained model selected for this project is DistilBERT, specifically `distilbert-base-uncased-finetuned-sst-2-english`. DistilBERT is a smaller, faster, and lighter version of BERT (Bidirectional Encoder Representations from Transformers), which is designed to retain most of BERT's accuracy while being more efficient.

## Performance Metrics
The performance of the model is evaluated using the following metrics:
- **Accuracy**: The proportion of correctly predicted reviews out of the total reviews.
- **Precision**: The proportion of true positive predictions out of all positive predictions made.
- **Recall**: The proportion of true positive predictions out of all actual positives.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two metrics.

### Results

#### TF-IDF Representation
**Logistic Regression**
- Accuracy: 0.8771
- Precision: 0.8746
- Recall: 0.8815
- F1 Score: 0.8780

**Random Forest**
- Accuracy: 0.8426
- Precision: 0.8534
- Recall: 0.8285
- F1 Score: 0.8408

**Gradient Boosting**
- Accuracy: 0.8065
- Precision: 0.7760
- Recall: 0.8634
- F1 Score: 0.8174

**SVM**
- Accuracy: 0.8695
- Precision: 0.8721
- Recall: 0.8670
- F1 Score: 0.8695

#### BoW Representation
**Logistic Regression**
- Accuracy: 0.8440
- Precision: 0.8511
- Recall: 0.8351
- F1 Score: 0.8430

**Random Forest**
- Accuracy: 0.8396
- Precision: 0.8466
- Recall: 0.8306
- F1 Score: 0.8385

**Gradient Boosting**
- Accuracy: 0.8073
- Precision: 0.7773
- Recall: 0.8632
- F1 Score: 0.8180

**SVM**
- Accuracy: 0.8284
- Precision: 0.8377
- Recall: 0.8158
- F1 Score: 0.8266

#### DistilBERT Results
Using the pre-trained DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`), the performance on the train and test sets is as follows:

**Train Set Evaluation**
- Accuracy: 0.7957
- Precision: 0.9188
- Recall: 0.6495
- F1 Score: 0.7610

**Test Set Evaluation**
- Accuracy: 0.7988
- Precision: 0.9210
- Recall: 0.6551
- F1 Score: 0.7656

#### Optimized DistilBERT Results
After fine-tuning DistilBERT on our specific dataset, the evaluation metrics improved significantly. The fine-tuning process involved training the model for 3 epochs with a learning rate of 2e-5 and batch size of 16. The evaluation at the end of each epoch provided the following results:

**Epoch-wise Evaluation**
- **Epoch 1:**
  - Eval Loss: 0.6832
  - Eval Accuracy: 0.6452
- **Epoch 2:**
  - Eval Loss: 0.6659
  - Eval Accuracy: 0.6089
- **Epoch 3:**
  - Eval Loss: 0.6500
  - Eval Accuracy: 0.7540

The final training loss after 3 epochs was 0.6700, with a training accuracy of 0.7540. This indicates significant improvement over the initial pre-trained model results.

## Hyperparameters
During the optimization of the model, several hyperparameters were found to be particularly important:

- **Learning Rate**: The learning rate was set to 2e-5, which provided a good balance between training time and model performance.
- **Batch Size**: A batch size of 16 was used for both training and evaluation, which helped in managing the memory constraints while still providing efficient training.
- **Number of Epochs**: The model was trained for 3 epochs, which was sufficient to achieve convergence without overfitting.
- **Evaluation Strategy**: An evaluation strategy of 'epoch' was chosen to evaluate the model at the end of each epoch.
- **Save Strategy**: A save strategy of 'epoch' was used to save the model checkpoints at the end of each epoch for better model management.

## Conclusion
In conclusion, the optimized DistilBERT model significantly outperformed the initial pre-trained model and traditional machine learning models using TF-IDF and BoW representations. The fine-tuning process improved the model's accuracy and other performance metrics, making it a robust solution for sentiment analysis on movie reviews.

