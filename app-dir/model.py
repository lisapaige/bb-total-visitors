import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle

class JellyBean():
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        gb = GradientBoostingRegressor()
        gb.fit(X_train, y_train)
        rf = RandomForestRegressor(n_estimators=2000)
        rf.fit(X_train, y_train)
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        return gb, rf, lm
    
    def predict(self, X_test):
        gb, rf, lm = self.fit(self.X_train, self.y_train)
        gbhat = int(gb.predict(X_test)[0])
        rfhat = int(rf.predict(X_test)[0])
        lmhat = int(lm.predict(X_test)[0])
        return int((gbhat+rfhat+lmhat)/3)

if __name__=="__main__":
    
    # # read 4x4 dataset
    df = pd.read_csv('data/app_df.csv')
    # print(df)

    # make training data
    X_train, y_train = df.loc[:,('snow_bin', 'net_sales')], df.iloc[:,-1]
    
    # new datapoint for testing
    x0 = pd.DataFrame([[-1,8.4]], columns=['snow_bin', 'net_sales'])

    gb = GradientBoostingRegressor()
    rf = RandomForestRegressor(n_estimators=2000, random_state=1111)
    lm = LinearRegression()

    gb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    lm.fit(X_train, y_train)

    pickle.dump(gb, open('model_gb.pkl', 'wb'))
    gb = pickle.load(open('model_gb.pkl', 'rb'))
    pickle.dump(rf, open('model_rf.pkl', 'wb'))
    rf = pickle.load(open('model_rf.pkl', 'rb'))
    pickle.dump(lm, open('model_lm.pkl', 'wb'))
    lm = pickle.load(open('model_lm.pkl', 'rb'))

    for snow in [-1, 0, 1]:
        newx = [[snow, 8.4]]
        gbhat = gb.predict(newx)
        rfhat = rf.predict(newx)
        lmhat = lm.predict(newx)
        print(snow)
        print(gbhat, rfhat, lmhat)
        print((gbhat+rfhat+lmhat)/3)

    # # BLOCKER how to get requirements.txt to see my classes vs sklearn?
    # # model to pickle: rf
    # jb_mod = JellyBean()
    # jb_mod.fit(X_train, y_train)
    # print('jb:')
    # print(jb_mod.predict(x0))
    # # pickle model
    # pickle.dump(jb_mod, open('model_2.pkl', 'wb'))
    # model_2 = pickle.load(open('model_2.pkl', 'rb'))
    # # test pickled model
    # print(model_2.predict([[-1, 8.4]]))
    # print(model_2.predict([[0, 8.4]]))
    # print(model_2.predict([[1, 8.4]]))
