# Model Comparison

## Scope

These three experiments are related but are **not a perfectly apples-to-apples benchmark** because the first and third perform binary sentiment classification, while the Random Forest experiment performs five-class emotion classification using heuristic labels.

## Comparison

### Logistic Regression
- Representation: TF-IDF
- Feature limit: 5,000
- Classifier: Logistic Regression (`max_iter=1000`)
- Evaluation: classification report and confusion matrix
- Additional capability: custom/live review prediction

### Random Forest
- Representation: TF-IDF with unigram and bigram features
- Feature limit: 5,000
- Classifier: Random Forest (`n_estimators=200`)
- Training strategy: RandomOverSampler
- Labels: Joy, Sadness, Neutral, Contemplation, Concern
- Evaluation: classification report and confusion matrix

The notebook reports approximately 0.91 accuracy, but its macro F1 is substantially lower because the generated emotion labels are highly imbalanced. For that reason, the accuracy figure should not be presented as evidence that Random Forest outperformed the other models.

### BERT
- Base model: `bert-base-uncased`
- Task: binary sentiment classification
- Training subset: 5,000 IMDb reviews
- Test subset: 2,000 IMDb reviews
- Maximum sequence length: 128
- Framework: Hugging Face Transformers / PyTorch
- Evaluation: accuracy, binary precision, recall, F1, loss curves, confusion matrix, and sample predictions

## Main Takeaway

The experiments illustrate three different NLP strategies: sparse TF-IDF features with a linear classifier, TF-IDF features with a non-linear ensemble classifier and class balancing, and contextual representations learned by a pretrained transformer. Evaluation should account for the difference in tasks and, for the emotion experiment, the quality and distribution of the heuristic labels.