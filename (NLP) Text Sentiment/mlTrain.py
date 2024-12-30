import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump, load

def loadData():
    #load in data from csv and label
    dataCSV = pd.read_csv("data.csv", encoding='ISO-8859-1', header=None, usecols=[0, 5])
    dataCSV.columns = ['sentiment', 'comment']
    return dataCSV

def cleanData(data):
    # clean up data (blank spaces, nonalphanumerics, remove useless words)
    data = re.sub(r"http\S+", "", data)  # Remove URLs
    data = re.sub(r"@\w+", "", data)    # Remove mentions
    data = re.sub(r"#", "", data)       # Remove hashtags symbol
    data = re.sub(r"[^a-zA-Z\s]", "", data)  # Remove non-alphanumeric characters
    return data.lower().strip()

def preprocessData(data):
    # clean up text and store in cleaned_text col
    data['cleaned_text'] = data['comment'].apply(cleanData)
    return data

def extractFeatures(data):
    # vectorizer to convert to numerical features
    vectorizer = TfidfVectorizer(max_features=5000)  # limit for features
    X = vectorizer.fit_transform(data['cleaned_text'])  # text data to numerical features
    return X, vectorizer


def trainModel(X, y):
    # made random forest model and fit training data
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    print("TRAINING MODEL")
    model.fit(X, y)
    return model


def evalModel(model, X_test, y_test):
    # make a report
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    return report

def predictData(new_data, model, vectorizer):
    # clean and vectorize
    cleaned_data = [cleanData(new_data)]
    new_data_vectorized = vectorizer.transform(cleaned_data)  # Vectorize using the same vectorizer
    
    # predict
    predictions = model.predict(new_data_vectorized)
    
    return predictions

def main():
    # laod and process data
    data = loadData()
    print("DATA LOADED")
    data = preprocessData(data)
    print("DATA CLEANED")

    # vectorization of text

    X, vectorizer = extractFeatures(data)
    y = data['sentiment']  # 'sentiment' column as target labels
    print("DATA VECTORIZED")
    

    # split data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("DATA SPLIT")
    
    
    # train model

    model = trainModel(X_train, y_train)
    dump(model, 'rfmodel.joblib')
    print("MODEL TRAINED")

    
    # model report

    report = evalModel(model, X_test, y_test)
    print("Model Evaluation Report:\n", report)

'''
    # test on new data

    new_data = [
        "I love the new design of the iPhone!",
        "This is the worst experience I have ever had.",
        "Great product, highly recommend it!"
    ]
    
    predictions = predictData(new_data, model, vectorizer)
    
    print("Predictions for New Data:")
    for text, sentiment in zip(new_data, predictions):
        # replace w/ dictionary later
        if sentiment == 4:
            sentiment = "GOOD"
        else:
            sentiment = "BAD"
        print(f"Text: {text} -> Predicted Sentiment: {sentiment}")
        '''
        
if __name__ == "__main__":
    main()