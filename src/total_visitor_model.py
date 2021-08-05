# basic
from os import spawnlp
import numpy as np
import pandas as pd

# modeling
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.cluster import KMeans

def hard_snow():
    '''hard-coded snowfall data from OnTheSnow'''
    d = {}
    d['total_snowfall'] = np.array([None, 59, 90, 70, 168, 241, 132, 202, 140, 164])
    d['total_snowfall_days'] = np.array([None, 26, 38, 25, 52, 49, 34, 55, 43, 49])
    d['biggest_snowfall'] = np.array([None, 5, 5, 6, 11, 21, 11, 10, 14, 8])
    d['max_base_depth'] = np.array([58, 44, 67, 39, 66, 96, 60, 101, 65, 85])
    d['average_base_depth'] = np.array([51, 32, 37, 33, 48, 61, 39, 58, 44, 53])
    d['max_summit_depth'] = np.array([82, 44, 67, 39, 69, 11, 66, 105, 76, 90]) # 769 changed to 76
    d['average_summit_depth'] = np.array([66, 32, 38, 33, 52, 65, 41, 62, 55, 58])

    snowfall = pd.DataFrame.from_dict(d, 
                                    orient='index', 
                                    columns = ['2011-2012', '2012-2013', '2013-2014', '2014-2015', '2015-2016', '2016-2017',
                                               '2017-2018', '2018-2019', '2019-2020', '2020-2021'])
    snowfall = snowfall.T
    return snowfall

def hard_rev():
    '''hard-coded total revenue from IRS filing line 12'''
    d = {}
    d['tot_rev_line_12'] = np.array([7.3, 7.6, 7.1, 6.6, 8.8, 11.0, 16.9, 14.4, 16.1])
    tot_rev = pd.DataFrame.from_dict(d, 
                                    orient='index', 
                                    columns = ['2011-2012', '2012-2013', '2013-2014', '2014-2015', '2015-2016', '2016-2017',
                                               '2017-2018', '2018-2019', '2019-2020'])
    tot_rev = tot_rev.T
    return tot_rev

def snow_bin(df, col_name):
    km = KMeans(n_clusters=3, random_state=42).fit(df[[col_name]])
    lab, cen = km.labels_, sorted(km.cluster_centers_)
    break1, break2 = ((cen[0]+cen[1])/2), ((cen[1]+cen[2])/2)
    x = pd.cut(df.average_summit_depth, [0, int(break1), int(break2), 100], labels=[-1, 0, 1])
    df['snow_bin'] = pd.to_numeric(x)
    return df

def rev_fit(df):
    n = df.shape[0]
    betas = np.polyfit(x=np.arange(n-1), y=df.tot_rev_line_12[:-1], deg=1)
    X = np.append(np.arange(n), np.ones(n)).reshape(2,-1)
    return betas.dot(X)

def new_x_trend(df, snow=0, tot_rev=16.4, net_sales=8.4):
    '''make one new exog entry to predict on'''
    temp = pd.DataFrame([snow], index=['2021-2022'], columns=['snow_bin'])
    temp['tot_rev_trend'] = tot_rev
    temp['net_sales'] = net_sales
    return temp[['snow_bin', 'tot_rev_trend', 'net_sales']]

def new_x(df, snow=0, tot_rev=16.4, net_sales=8.4):
    '''make one new exog entry to predict on'''
    temp = pd.DataFrame([snow], index=['2021-2022'], columns=['snow_bin'])
    temp['tot_rev'] = tot_rev
    temp['net_sales'] = net_sales
    return temp[['snow_bin', 'tot_rev', 'net_sales']]

class ResultPipeLine():
    def __init__(self):
        pass

    def xgb_fit(self, X_train, y_train):
        xgb=XGBRegressor(objective='reg:squarederror')
        xgb.fit(X_train, y_train)
        return xgb
    def xgb_predict(self, X_test):
        xgb = self.xgb_fit(X_train, y_train)
        yhat = xgb.predict(X_test)
        return yhat

    def gb_fit(self, X_train, y_train, alpha=0.5):
        gb=GradientBoostingRegressor(loss='quantile', alpha=alpha)
        gb.fit(X_train, y_train)
        return gb
    def gb_predict(self, X_test):
        gb = self.gb_fit(X_train, y_train)
        yhat = gb.predict(X_test)
        return yhat

    def gb_fit_low(self, X_train, y_train, alpha=0.025):
        gb=GradientBoostingRegressor(loss='quantile', alpha=alpha)
        gb.fit(X_train, y_train)
        return gb
    def gb_predict_low(self, X_test):
        gb = self.gb_fit_low(X_train, y_train)
        yhat = gb.predict(X_test)
        return yhat

    def gb_fit_high(self, X_train, y_train, alpha=.975):
        gb=GradientBoostingRegressor(loss='quantile', alpha=alpha)
        gb.fit(X_train, y_train)
        return gb
    def gb_predict_high(self, X_test):
        gb = self.gb_fit_high(X_train, y_train)
        yhat = gb.predict(X_test)
        return yhat

    def dt_fit(self, X_train, y_train):
        dt=DecisionTreeRegressor()
        dt.fit(X_train, y_train)
        return dt
    def dt_predict(self, X_test):
        dt = self.dt_fit(X_train, y_train)
        yhat = dt.predict(X_test)
        return yhat

    def lmod_fit(self, X_train, y_train):
        lmod = LinearRegression()
        lmod.fit(X_train, y_train)
        return lmod
    def lmod_predict(self, X_test):
        lmod = self.lmod_fit(X_train, y_train)
        yhat = lmod.predict(X_test)
        return yhat

if __name__=="__main__":
    # read in data
    ski_df = pd.read_csv('../data/ski_resort_data.csv', parse_dates=['ymd', 'date'])
    sp = pd.read_csv('../data/season_pass_sale.csv')

    # get response totals for each season
    targets = ['tot_alpine', 'tot_day', 'tot_sp_est',
               'day_sp_est', 'tot_tix', 'day_tix',
               'tot_night', 'night_sp_est', 'night_tix']
    data = ski_df[['season']+targets].groupby('season').sum()

    # change year of season pass sale to season of season pass sale
    season_lst = ['2016-2017']
    season_lst+=list(ski_df.season.unique())
    season_lst.append('2021-2022')
    sp_year_lst = list(sp.year.unique()[::-1])
    season_feature=[]
    for yr in sp.year:
        season_feature.append(season_lst[sp_year_lst.index(yr)])
    sp['season'] = season_feature
    # get net early bird window sales by season
    sp = sp[['season', 'net_sales']].groupby('season').sum()
    sp['net_sales'] = sp['net_sales']/1_000_000

    print(sp)

    # add hard coded snow data from OnTheSnow +
    # hard coded total revenue data from IRS filings
    df = hard_snow().join(hard_rev(), how='left').join(sp, how='left').join(data, how='left')
    # bin snow data into -1,0,1
    df = snow_bin(df, 'average_summit_depth')
    # true values from IRS + projected value for 2020-21
    df['tot_rev_trend'] = rev_fit(df)

    response = 'tot_alpine'
    temp = df[['snow_bin', response]].iloc[6:]
    temp['tot_rev'] = np.array([16.9, 14.4, 16.1, 17.2])
    temp['net_sales'] = sp.net_sales.values[1:-1]
    temp = temp[['snow_bin', 'tot_rev', 'net_sales', 'tot_alpine']]
    temp.to_csv('../data/app_df.csv', index=False)
    
    print(temp)

    # validation on 2020-21
    X_train = df[[response, 'snow_bin', 'tot_rev_trend']].iloc[6:-1,:]
    X_test = pd.DataFrame(df[[response, 'snow_bin', 'tot_rev_trend']].iloc[-1,:]).T
    y_train, y_test = X_train.pop(response), X_test.pop(response)

    # no data for holdout
    X = df[[response, 'snow_bin', 'tot_rev_trend']].iloc[6:,:]
    y = X.pop(response)

    # next season test data
    snow_value = 0
    x0 = new_x(df, snow=snow_value)
    x0 = x0.iloc[:,:-1]

    x0_trend = new_x_trend(df, snow=snow_value)
    x0_trend = x0_trend.iloc[:,:-1]


    # test set from before 2017-18 season
    X_test_pre = df[['snow_bin', 'tot_rev_trend']].iloc[:6,:]

    rpl_val = ResultPipeLine()
    rpl_new = ResultPipeLine()
    rpl = ResultPipeLine()

    # xgboost
    rpl_val.xgb_fit(X_train, y_train)
    yhat_train_xgb = rpl_val.xgb_predict(X_train)
    yhat_xgb = rpl_val.xgb_predict(X_test)

    rpl_new.xgb_fit(X,y)
    yhat0_xgb = rpl_new.xgb_predict(x0_trend)
    yhat_xgb_pre = rpl_new.xgb_predict(X_test_pre)

    # gb
    rpl.gb_fit(X_train, y_train)
    yhat_train_gb = rpl.gb_predict(X_train)
    yhat_gb = rpl.gb_predict(X_test)
    yhat0_gb = rpl.gb_predict(x0_trend)
    # gb_low
    rpl.gb_fit_low(X_train, y_train)
    yhat_train_gb_low = rpl.gb_predict_low(X_train)
    yhat_gb_low = rpl.gb_predict_low(X_test)
    yhat0_gb_low = rpl.gb_predict_low(x0_trend)
    # gb_high
    rpl.gb_fit_high(X_train, y_train)
    yhat_train_gb_high = rpl.gb_predict_high(X_train)
    yhat_gb_high = rpl.gb_predict_high(X_test)
    yhat0_gb_high = rpl.gb_predict_high(x0_trend)

    # dt
    rpl.dt_fit(X_train, y_train)
    yhat_train_df = rpl.dt_predict(X_train)
    yhat_dt = rpl.dt_predict(X_test)
    yhat0_dt = rpl.dt_predict(x0_trend)

    # lmod
    rpl_val.lmod_fit(X_train, y_train)
    yhat_train_lmod = rpl_val.lmod_predict(X_train)
    yhat_lmod = rpl_val.lmod_predict(X_test)

    rpl_new.lmod_fit(X,y)
    yhat0_lmod = rpl_new.lmod_predict(x0_trend)
    yhat_lmod_pre = rpl_new.lmod_predict(X_test_pre)

    # xgb
    print(f'snow value: {snow_value}')
    print('XGB: train, val, new_x, old_X')
    print(yhat_train_xgb)
    print(yhat_xgb)
    print(yhat0_xgb)
    print(yhat_xgb_pre)

    # lmod
    print(" ")
    print('lmod: train, val, new_x, old_X')
    print(yhat_train_lmod)
    print(yhat_lmod)
    print(yhat0_lmod)
    print(yhat_lmod_pre)

    print(" ")
    print('gb: new_x_low, new_x, new_x_high')
    print(yhat0_gb_low)
    print(yhat0_gb)
    print(yhat0_gb_high)

    print(" ")
    print('dt: test, new_x')
    print(yhat_dt)
    print(yhat0_dt)
    
    res = pd.DataFrame(np.array([[192, 192, 285, 320], [195, 250,304,359], [253, 243, 285, 285], [237, 234, 285, 298], [285, 285, 285, 320], [192, 285, 285, 320]]).reshape(6,4), index=['xgb', 'lmod', 'gb_low', 'gb', 'gb_high', 'dt'], columns=[0,1,2,3])
    jelly_beans = res.mean(axis=0)
    print(jelly_beans)

    print(x0)
