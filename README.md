# congressional_tweets_project_code_istanbul

This is a README file that explains the steps that we followed for the Kaggle project in order of that the notebooks ran.

## Method 1) Support Vector Machines (SVM)

### svm.ipynb

This is the notebook wehere we process the **hashtags** column with **tfidf** method and then, apply classification using **SVM**. The accuracy returned by Kaggle is **0.85501**.

### svm_tfidf.ipynb

This is the notebook wehere we process the **full_text** column with **tfidf** method and then, apply classification using **SVM**. The accuracy returned by Kaggle is **0.89122**.

## Method 2) Logistic Regression with ElasticNet

## prepare_data.ipynb

This is the notebook that prepares X_train.csv, y_train.csv, X_test.csv and Id.csv files to be fed into the notebooks running the main classification methods. To preprocess the data, we clean it using the tools like tokenizing, lemmatizing, etc. in the **nltk** package, and then apply the **bag of words** method to vectorize the cleaned text data and the cleaned hashtags data. It could not complete running because of memory issues. Therefore, there are also .py versions of the same file that were run on BlueHive. One example is **prepare_data1.py** where we choose 1% of the training data using **stratification** and preprocess 1/22 of the test data. However, in order to have predictions, all of the test data has to be preprocessed the same way and even preprocessing a single chunk of BlueHive took more than 21 hours, so we canceled all such runs. 

## logistic_regression_elasticnet.ipynb

The notebook where we apply **logistic regression without cross validation with ElasticNet**. It reads the training and test data sets and applies the model. Although it is simple, it could not be run 

## Method 3) BERT

## Older versions of the codes

### kaggle_preprocess.py

Earlier version of prepare_data.py code being run on BlueHive.

### congressional_tweets_project_code_istanbul.ipynb

Early version of the code that combines preprocessing and logistic regression with ElasticNet.
