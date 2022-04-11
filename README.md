# congressional_tweets_project_code_istanbul

This is a README file that explains the steps that we followed for the Kaggle project in order of that the notebooks ran.

## Method 1) Support Vector Machines (SVM)

### svm.ipynb

This is the notebook wehere we process the **hashtags** column with **tfidf** method and then, apply classification using **SVM**. The accuracy returned by Kaggle is **0.85501**.

### svm_tfidf.ipynb

This is the notebook wehere we process the **full_text** column with **tfidf** method and then, apply classification using **SVM**. The accuracy returned by Kaggle is **0.89122**.

## Method 2) Logistic Regression with ElasticNet

### prepare_data.ipynb

This is the notebook that prepares X_train.csv, y_train.csv, X_test.csv and Id.csv files to be fed into the notebooks running the main classification methods. To preprocess the data, we clean it using the tools like tokenizing, lemmatizing, etc. in the **nltk** package, and then apply the **bag of words** method to vectorize the cleaned text data and the cleaned hashtags data. It could not complete running because of memory issues. Therefore, there are also .py versions of the same file that were run on BlueHive. One example is **prepare_data1.py** where we choose 1% of the training data using **stratification** and preprocess 1/22 of the test data. However, in order to have predictions, all of the test data has to be preprocessed the same way and even preprocessing a single chunk of BlueHive took more than 21 hours, so we canceled all such runs. 

This notebook was the hardest one to complete because cleaning large text data, vectorizing it with bag of words, and having dimensionality reduction are all computationally expensive operations. Therefore, there are also simplified versions of this file, for instance, where we apply that to hashtags, however, we did not include all the relevant code for such cases here, because even those simplified versions had technical limitations and could not complete.

### logistic_regression_elasticnet.ipynb

The notebook where we apply **logistic regression without cross validation with ElasticNet**. It reads the training and test data sets and applies the model. Although it is simple, it could not be run because no preprocessing run involving bag of words or dimensionality reduction could be applied to the test data due to time and memory limits.

### logistic_regression_tfidf

Regarding the computational burdens of bag of words, and the main purpose of Word2vec being to consider the similarity in the text, we use **tfidf** as a faster method to vectorize the text data. Then, we apply **logistic regression with no cross validation but with ElasticNet** to fit a model on the training data set and have predictions for the test data set. 

One better version of this code could be to clean the text here better as we do in prepare_data.ipynb before applying tfidf, however, even the presented version of the notebook here took almost a day to complete the run. Such a version of the code could be submitted over BlueHive to be run for a few days if we had more time.

## Method 3) BERT

## Method 4) Naive Bayes

## Older versions of the codes

### kaggle_preprocess.py

Earlier version of prepare_data.py code being run on BlueHive.

### congressional_tweets_project_code_istanbul.ipynb

Early version of the code that combines preprocessing and logistic regression with ElasticNet.
