# <center><b>Real Anot</b>: identifying COVID-19-related fake news</center>
Original implementation by Barry YAP, Kelvin SOH, Kenny CHUA and Zhong Hao NEO (AI Apprentices, Batch 6, AI Singapore)

Modified implementation by Kenny CHUA

## Introduction
<b>Real Anot</b> is a web app that uses neural networks to predict the probability that a given piece of text is real/fake news.

## Dataset
The dataset is a subset of the [CoAID](https://github.com/cuilimeng/CoAID/) dataset which containts a set of diverse COVID-19 healthcare misinformation. This subset has a total of 1,127 real and 266 fake news samples.

## Preprocessing
First the dataset was split into training and validation sets with a ratio of 75% to 25%.

Next, artefactual content was removed from the dataset. The dataset was collated from web sources, and therefore contained invalid content such as:
```
you must log in to continue ..
```

After removing artefactual content, we then concatenated the tile and content columns in the dataset.

Finally, the data was tokenised.

## Model
The current implementation of Real Anot uses the [DistilBERT](https://arxiv.org/abs/1910.01108) transformer language model. The data was used for fine-tuning the DistilBERT model using the [`ktrain`](https://github.com/amaiya/ktrain) library.

To handle class imbalance in our dataset, class weights were set as `n_samples / (n_classes * np.bincount(y))`.

Fine-tuning was performed with a batch size of 32, with a max learning rate of 0.0001. Fine-tuning was automatically stopped when validation accuracy did not improve for five consecutive epochs.

## Evaluation
Evaluation was performed on the 25% validation dataset:
```
              precision    recall  f1-score   support

        real       0.92      0.77      0.84        75
        fake       0.94      0.98      0.96       274

    accuracy                           0.94       349
   macro avg       0.93      0.88      0.90       349
weighted avg       0.94      0.94      0.93       349
```
```
|             | Predicted fake | Predicted real |
|-------------|----------------|----------------|
| Actual fake |              58|              17|
| Actual real |               5|             269|
```

## Unit testing
Unit tests for preprocessing and inference functions can be initialised by running:
```
$ pytest
```

## Demo app
- Instructions for launching the demo app locally
```
$ python -m app
```
After running the above command, the app will be accessible at http://localhost:8000. 

An earlier version of Real Anot (not using DistilBERT) is available [here](https://real-anot.herokuapp.com). As this version was deployed on Heroku, the first load might require more time.

## Disclaimer
Real Anot is for reference purposes only, and does not substitute advice from relevant professionals.