import mlTrain
from joblib import load


def main():
    translate = {
        0:"Bad",
        4:"Good"
    }

    # laod and process data

    data = mlTrain.loadData()
    print("DATA LOADED")
    data = mlTrain.preprocessData(data)
    print("DATA CLEANED")


    # vectorization of text

    X, vectorizer = mlTrain.extractFeatures(data)
    y = data['sentiment']  # 'sentiment' column as target labels
    print("DATA VECTORIZED")


    # load in model

    model = load('rfmodel.joblib')
    print("MODEL LOADED")


    while True:
        print("Enter Input: ")
        newData = input()
        predictions = mlTrain.predictData(newData, model, vectorizer)
    
        print("Predictions for New Data:")
        print(translate[predictions[0]])
    

if __name__ == "__main__":
    main()