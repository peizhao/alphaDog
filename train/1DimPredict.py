from utils.logger import *
import matplotlib
import datetime as dt, itertools, pandas as pd, matplotlib.pyplot as plt, numpy as np
import math

csvFile = '../data/IBM.csv'
log.info("start")

logRet = lambda x,y : log(y/x)
zscore = lambda x : (x - x.mean())/x.std()

D = pd.read_csv(csvFile)
# print(D['Adj Close'])

def Ret(x, y):
    return math.log(y/x)

def make_inputs(filepath):
    D = pd.read_csv(filepath)
    #print(D)
    Res = pd.DataFrame()

    x = D['Open'].astype(float)
    y = D['Close'].astype(float)

    print(zscore(D['Open']))
    #logRet(D['Open'].astype(float), D['Adj Close'].astype(float))
    #Res['c_2_o'] = logRet(D['Open'], D['Adj Close'])
    #print(Res)

make_inputs(csvFile)