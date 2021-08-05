import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

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

    x0_ = pd.DataFrame([[-1,8.4]], columns=['snow_bin', 'net_sales'])
    rf_mod = RandomForestRegressor(n_estimators=2000)
    rf_mod.fit(X_train, y_train)
    print('rf:')
    print(rf_mod.predict(x0_))
    
    # choosing a number of estimators for stability in rf predictions
    rf_stds = []
    x = np.arange(500, 3501, 500)
    for n in x:
        rf_vals = []
        for _ in np.arange(30):
            rf_mod = RandomForestRegressor(n_estimators=n)
            rf_mod.fit(X_train, y_train)
            rf_vals.append(rf_mod.predict(x0))
        # print(rf_vals)
        print(n, np.std(rf_vals))
        rf_stds.append(np.std(rf_vals))

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(x, rf_stds, color='crimson')
    # ax.axvline(x=2500, label='choose: 2000')
    ax.set_ylabel('std dev in predictions')
    ax.set_xlabel('number of estimators')
    ax.set_title('Stabilize Random Forest Predictions')
    # fig.savefig('../images/n_est.png')
    plt.show()

    feature_names = X_train.columns
    importances = rf_mod.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)[::-1]
    std = np.std([
        tree.feature_importances_ for tree in rf_mod.estimators_], axis=0)
    fig, ax = plt.subplots()
    forest_importances.plot.barh(yerr=std, ax=ax, color='darkslateblue')
    ax.set_title("Feature Importances")
    fig.tight_layout()
    # fig.savefig('../images/feat_imp3.png')
    fig.savefig('../images/feat_imp2.png')
    plt.show()

    
    result = permutation_importance(rf_mod, X_train, y_train, n_repeats=100,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=X_train.columns[sorted_idx])
    ax.set_title("Permutation Importance")
    ax.axvline(x=0, color='crimson', linestyle='--')
    fig.tight_layout()
    # fig.savefig('../images/perm_imp3.png')
    fig.savefig('../images/perm_imp2.png')
    plt.show()
