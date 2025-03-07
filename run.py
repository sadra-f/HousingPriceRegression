import pandas as pd
import numpy as np
from Visualize.plot import plot_multi_col as plot
from MLModel.LinearRegression import LinearRegression as LR
from sklearn.linear_model import LinearRegression as SKLLR

def main():

    DATASET_PATH = "dataset/kc_house_data.csv"
    CLEAN_DATASET_PATH = "dataset/clean_kc_house_data.csv"

    dataset = pd.read_csv(CLEAN_DATASET_PATH)
    """['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long']
    """

    train = dataset.sample(frac=0.7,random_state=200)
    train_Y = np.array(train['price'])
    train_X = np.array(train.drop(['price'], axis=1))
    plot(train_X, train_Y, 5, 3, "Train Dataset")

    test = dataset.drop(train.index)

    test_Y = np.array(test['price'])
    test_X = np.array(test.drop(['price'], axis=1))
    
    slrm = SKLLR(copy_X=True)
    slrm = slrm.fit(train_X, train_Y)
    res2 = slrm.predict(test_X)
    
    regression_model = LR()
    regression_model.train(train_X, train_Y)
    regression_model.save_model("weights.txt")

    regression_model = LR()
    regression_model.load_model("weights.txt")
    res = regression_model.predict(test_X)

    print("Mean of Squared Error for Custom Model=> ", np.mean(np.abs(np.subtract(test_Y, res))**2))
    print("Mean of Squared Error for SKLEARN => ", np.mean(np.abs(np.subtract(test_Y, res2))**2))
    print("Root Mean Squared Percentage Error for Custom Model=> ", np.sqrt(np.mean(((res - test_Y) / test_Y) ** 2)) * 100)
    print("Root Mean Squared Percentage Error for SKLEARN => ", np.sqrt(np.mean(((res2 - test_Y) / test_Y) ** 2)) * 100)
    print()



if __name__ == '__main__':
    main()