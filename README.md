# Machine Learning Fall 2022 Final Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jJd_9gNRrQ3GDV-UTFUyZuaFPxgGeTXp?usp=sharing)

## Project Description
We are trying to solve the problem of automating hate speech and offensive language detection.

Hate speeches are common on social media, and it would be easier for such speeches to be regulated if some program can automatically detect them. The problem is similar to the language recognition in  hw3  lab in that we take a natural language input as a sequence, and train a model to predict some labels associated with such input sequence. The unique part of this task is that hate speech/offensive language is sometimes hard to detect because it really depends on the context the language is used. By automating hate speech and offensive language detection, we could contribute to making a more healthy internet environment.

## Install Dependencies
### On MacOS/Linux
```
pip install -r requirements.txt
```

## Quickstart
`ML_project_3labels.ipynb` is a jupyter notebook file contains the 3 labels classifier model we wrote.


`ML_project_2labels.ipynb` is a jupyter notebook file in which we combined the "Offensive" and "Hate" languages in the dataset together to make binary classification

### Methods Documentation
```python
create_train_and_test_set_balanced(X, y, train_ratio=0.8)
```
> Parameters
  - `X`: array of sentence embeddings
  - `y`: labels
  - `train_ratio`: proportion of size of training set to
> Returns
  - `X_train`: Training data
  - `X_rem`: Testing data
  - `y_train`: Training labels
  - `y_rem`: Testing labels
---
```python
model.fit(train_loader, epochs=300, lr=1e-5, interval=100)
```
> Parameters
  - `train_loader`: Dataloader for the training dataset
  - `epochs`: number of epochs in training
  - `lr`: learning rate of optimizer
  - `interval`: frequency to output loss information
---
```python
model.validate(valid_loader)
```
> Parameters
  - `valid_loader`: Dataloader for the validation dataset
> Returns
  - The average validation loss
---
```python
model.accuracy(test_loader)
```
> Parameters
  - `test_loader`: Dataloader for the testing dataset
> Returns
  - Accuracy score of the model on the testing dataset
---
```python
model.predict(sentence)
```
> Parameters
  - `sentence`: Input sentence to predict its category
> Returns
  - `Hate`, `Offensive` or `neither`
---
```python
model.metrics(test_loader)
```
> Parameters
  - `test_loader`: Dataloader for the testing dataset
> Returns
  - Evaluation metrics including accuracy, precision, recall, f1 score and a ROC graph


