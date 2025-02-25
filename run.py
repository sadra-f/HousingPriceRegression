import pandas as pd
import numpy as np
from Visualize.plot import plot_multi_column_on_process as plot
from MLModel.LinearRegression import LinearRegression as LR


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
    
    regression_model = LR()
    regression_model.train(train_X, train_Y)
    regression_model.save_model("weights.txt")

    regression_model = LR()
    regression_model.load_model("weights.txt")
    res = regression_model.predict(test_X)
    regression_model.calc_loss(res, test_Y)



if __name__ == '__main__':
    main()