from cvxpy import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pyKstest as kstest2
from scipy import stats

# This function returns the matrix of returns
# Expected input is a pandas dataframe of prices
def returnsMatrix(rawData):
    rets = pd.DataFrame()
    columnNames = list(rawData)
    newColumns = []

    for i in columnNames:
        if (i != 'Date'):
            newColumns.append(i)

    rets = rawData[newColumns].copy()

    for i in list(rets):
        pd.to_numeric(rets[i], errors='ignore')

    temp1 = rets[0:rawData.shape[0] - 1]
    temp2 = temp1.shift(1)
    rets = temp1 / temp2 - 1

    return rets

# Takes a list of prices and returns a list of returns
def returnsMatrix2(a):
    return np.divide(a[1:], a[0:len(a)-1])-1

modelData = pd.read_csv('Data/MarketData.csv')
market = pd.read_csv('Data/Market.csv')
rate = pd.read_csv('Data/Rate3m.csv')

rets = returnsMatrix(modelData)
rets = rets[1:len(rets)-1]

#### Mean Variance Optimization #######
# mu is the mean return vector as a list
# sigma is the covariance matrix of returns
# gamma is the risk aversion scalar
def mvo(mu, sigma, gamma):
    # Long only portfolio optimization.
    w = Variable(len(mu))
    ret = mu.T*w
    risk = quad_form(w, sigma)
    prob = Problem(Maximize(ret - gamma*risk),
                   [sum_entries(w) == 1,
                    w >= 0])
    prob.solve()

    return w.value

'''print ('MVO')

tempRes = 0
g = 2000
counter = 0

while (tempRes != 1 or counter >= 100):
    mvoResult = mvo(np.array(rets.mean()*252), np.matrix(rets.cov()*252), g)
    mvoResult2 = mvo(np.array(rets.mean()*252), np.matrix(rets.cov()*252), g+1)

    ds1 = []
    ds2 = []
    for i in range (len(mvoResult)):
        ds1.append(float(mvoResult[i]))
        ds2.append(float(mvoResult2[i]))

    tempRes = (stats.ks_2samp(ds1,ds2))[1] < 0.05
    if (tempRes):
        print('Not The Same', g)
    else:
        print ('Same', g)

    g += 1'''
#mvoResult = mvo(np.array(rets.mean()*252), np.matrix(rets.cov()*252), 5)
#print (mvoResult)
#print (mvo(np.array(rets.mean()*252), np.matrix(rets.cov()*252), 5))
#################

########## Risk Parity Portfolio ##########
# sigma is the covariance matrix of returns
def riskParity(sigma):
    # Long only portfolio optimization.
    w = Variable(len(sigma))
    risk = quad_form(w, sigma)
    lweight = sum_entries(log(w))
    prob = Problem(Minimize(risk - lweight),
                   [w >= 0])
    prob.solve()
    return (w.value/sum_entries(w).value)

#print ('Risk Parity')
#rpResult = riskParity(np.matrix(rets.cov()*252))
#print (rpResult)
#print (rpResult.T.dot(np.array(rets.mean()*252)))

##############

########### CVaR Optimization ##########
# rets is the returns matrix of all assets
# (direct output of the returnsMatrix function)
def cvar(rets):
    s = len(rets)
    z = Variable (s)
    x = Variable(rets.shape[1])
    r = Variable(1)
    cons = [z >=0, sum_entries(x) == 1, x >=0, x<= 0.08]

    for i in range (s):
        cons += [z[i,0]>= -rets[i]*x - r]

    prob = Problem(Minimize(r + sum_entries(z)/((1-0.95)*s)), cons)
    prob.solve(solver = 'CVXOPT')
    return x.value

#cvarResult = cvar(np.array(rets))
#print (cvarResult)
#print (cvarResult.T.dot(np.array(rets.mean()*252)))

##################


#### Utility Maximization ####
# mu is the mean return vector as a list
# sigma is the covariance matrix of returns
# lam is the risk aversion scalar
def utilopt(mu, sigma, lam):
    x = Variable (len(mu))
    ret = mu.T*x
    risk = quad_form(x, sigma)
    utility = -exp(-lam*(1+ret))
    prob = Problem(Maximize(utility - risk),
                   [x>=0,
                    sum_entries(x) == 1])
    prob.solve(solver = 'CVXOPT')
    return x.value

#utilResult = utilopt(np.array(rets.mean()*252), np.matrix(rets.cov()*252), 2)
#print (utilResult)
#print (utilResult.T.dot(np.array(rets.mean()*252)))
##############################


'''
#### SSD-DP ######
m = 10
x = np.random.randn(m, rets.shape[1])

def phi (v, p):
    return ((1/v)*(1-p)**((1-v)/v))

def phi2 (v, j):
    N = 1000
    ub = j
    lb = j-1
    dp = (ub-lb)/N
    counter = lb
    result = 0
    for i in range (N):
        counter+= dp
        result+= (phi(v, counter) + phi(v, counter-dp))/2*dp
    return result/N

def deltaPhi (v, j, m):
    if j >=1 and j <= m-1:
        return phi2(v, j+1) - phi(v, j)
    elif j == m:
        return -phi2(v, m)
    else:
        return 0

def ssd(v, m):
    a = Variable (m , m)
    p = Variable (m)
    lam = Variable (rets.shape[1])

#print (phi(0.3, 1))
##################
'''

# Produces the weights in risk-free asset and a risky fund
# rf - risk free rate
# mp - expected return from the risky fund
# ret - desired return
def benchmark(rf, mp, ret):
    w1 = (ret-mp)/(rf-mp)
    w1 = max(w1, -0.5) # maximum short at risk-free of 50%
    w1 = min (w1, 1.5)
    return [w1, 1-w1]

#print (benchmark(float(rate.iloc[len(rate)-1]['Rate']), float(returnsMatrix(market).mean())*252, 0.2))


########### Back tester Function ##################
def backtest(desiredRet, start, end, trainingPeriod, period, gamma, lam, critical, showPlot, modelData, market, rate):
    rets = returnsMatrix(modelData)
    rets = rets[1:len(rets)-1]

    dates = []
    benchmarkPerformance = []
    mvoPerformance = []
    rpPerformance = []
    cvarPerformance = []
    utilPerformance = []

    benchmarkPerformanceul = []
    mvoPerformanceul = []
    rpPerformanceul = []
    cvarPerformanceul = []
    utilPerformanceul = []

    benchalpha = []
    mvoalpha = []
    rpalpha = []
    cvaralpha = []
    utilalpha = []

    trainingDataIndex = start*period - trainingPeriod*period
    benchmarkWeights = [0]*2
    benchmarkBudget = 1

    mvoWeights = [0]*2
    mvoBudget = 1
    mvoResult = 0

    rpWeights = [0]*2
    rpBudget = 1
    rpResult = 0

    cvarWeights = [0]*2
    cvarBudget = 1
    cvarResult = 0

    utilWeights = [0]*2
    utilBudget = 1
    utilResult = 0

    rateLock = 0
    rebalanceDates = []

    for i in range (start*period, end*period):
        if (i % period == 0):
            rebalanceDates.append(dt.datetime.strptime(modelData.iloc[i]['Date'], '%Y-%m-%d').date())
            rateLock = float(rate.iloc[i]['Rate'])
            temp = returnsMatrix(market.iloc[trainingDataIndex:i])
            temp2 = returnsMatrix(modelData.iloc[trainingDataIndex:i])

            benchmarkWeights = benchmark(float(rate.iloc[i]['Rate']),
                                         float(temp.mean())*252, desiredRet)

            mvoResult = mvo(np.array(temp2.mean()), np.matrix(temp2.cov()*252), gamma)
            mvoWeights = benchmark(float(rate.iloc[i]['Rate']),
                                         float(mvoResult.T.dot(np.array(temp2.mean()*252))), desiredRet)

            rpResult = riskParity(np.matrix(temp2.cov() * 252))
            rpWeights = benchmark(float(rate.iloc[i]['Rate']),
                                   float(rpResult.T.dot(np.array(temp2.mean() * 252))), desiredRet)

            cvarResult = cvar(np.matrix(temp2.mean() * 252))
            cvarWeights = benchmark(float(rate.iloc[i]['Rate']),
                                  float(cvarResult.T.dot(np.array(temp2.mean() * 252))), desiredRet)

            utilResult = utilopt(np.array(temp2.mean() * 252), np.matrix(temp2.cov() * 252), lam)
            utilWeights = benchmark(float(rate.iloc[i]['Rate']),
                                    float(utilResult.T.dot(np.array(temp2.mean() * 252))), desiredRet)

            initialBench =  float(market.iloc[i]['Market'])
            initialMVO = float(mvoResult.T.dot(modelData.iloc[i][1:]))
            initialrp = float(rpResult.T.dot(modelData.iloc[i][1:]))
            initialcvar = float(cvarResult.T.dot(modelData.iloc[i][1:]))
            initialutil = float(utilResult.T.dot(modelData.iloc[i][1:]))

            if i == start*period:
                benchmarkBudget = 1
                mvoBudget = 1
                rpBudget = 1
                cvarBudget = 1
                utilBudget = 1

            else:
                benchmarkBudget = max(benchmarkValue, 0)
                mvoBudget = max(mvoValue, 0)
                rpBudget = max(rpValue, 0)
                cvarBudget = max(cvarValue, 0)
                utilBudget = max(utilValue, 0)

        dates.append(dt.datetime.strptime(modelData.iloc[i]['Date'], '%Y-%m-%d').date())

        benchmarkValue = benchmarkWeights[0]*benchmarkBudget*(1 + rateLock)**((i%period)/252) + \
                         benchmarkWeights[1]*benchmarkBudget*float(market.iloc[i]['Market'])/initialBench

        benchmarkValueul = benchmarkBudget*float(market.iloc[i]['Market'])/initialBench

        mvoMarketValue = float(mvoResult.T.dot(modelData.iloc[i][1:]))

        mvoValue = mvoWeights[0] * mvoBudget * (1 + rateLock) ** ((i % period) / 252) + \
                         mvoWeights[1] * mvoBudget * float(mvoMarketValue) / initialMVO

        mvoValueul = mvoBudget * float(mvoMarketValue) / initialMVO

        rpMarketValue = float(rpResult.T.dot(modelData.iloc[i][1:]))

        rpValue = rpWeights[0] * rpBudget * (1 + rateLock) ** ((i % period) / 252) + \
                   rpWeights[1] * rpBudget * float(rpMarketValue) / initialrp

        rpValueul = rpBudget * float(rpMarketValue) / initialrp

        cvarMarketValue = float(cvarResult.T.dot(modelData.iloc[i][1:]))

        cvarValue = cvarWeights[0] * cvarBudget * (1 + rateLock) ** ((i % period) / 252) + \
                  cvarWeights[1] * cvarBudget * float(cvarMarketValue) / initialcvar

        cvarValueul = cvarBudget * float(cvarMarketValue) / initialcvar

        utilMarketValue = float(utilResult.T.dot(modelData.iloc[i][1:]))

        utilValue = utilWeights[0] * utilBudget * (1 + rateLock) ** ((i % period) / 252) + \
                    utilWeights[1] * utilBudget * float(utilMarketValue) / initialutil

        utilValueul = utilBudget * float(utilMarketValue) / initialutil

        benchmarkPerformance.append(benchmarkValue)
        mvoPerformance.append(mvoValue)
        rpPerformance.append(rpValue)
        cvarPerformance.append(cvarValue)
        utilPerformance.append(utilValue)

        benchmarkPerformanceul.append(benchmarkValueul)
        mvoPerformanceul.append(mvoValueul)
        rpPerformanceul.append(rpValueul)
        cvarPerformanceul.append(cvarValueul)
        utilPerformanceul.append(utilValueul)

        if (i > start*period + 1):
            totalPeriod = i - start * period

            benchTemp = returnsMatrix2(benchmarkPerformance)
            mvoTemp = returnsMatrix2(mvoPerformance)
            cvarTemp = returnsMatrix2(cvarPerformance)
            utilTemp = returnsMatrix2(utilPerformance)
            rpTemp = returnsMatrix2(rpPerformance)

            marketTemp = returnsMatrix(market.iloc[start * period:i + 2])
            marketTemp = marketTemp[1:len(marketTemp)]
            marketRet = (market.iloc[end * period]['Market'] / market.iloc[start * period]['Market']) ** (
            252 / totalPeriod) - 1


            cov = np.cov(marketTemp['Market'], benchTemp)
            beta = cov[1, 0] / cov[0, 0]
            benchalpha.append(benchmarkPerformance[len(benchmarkPerformance) - 1] ** (252 / totalPeriod) - 1 - (
            rateLock + beta * (marketRet - rateLock)))

            cov = np.cov(marketTemp['Market'], mvoTemp)
            beta = cov[1, 0] / cov[0, 0]
            mvoalpha.append(mvoPerformance[len(mvoPerformance) - 1] ** (252 / totalPeriod) - 1 - (
                rateLock + beta * (marketRet - rateLock)))

            cov = np.cov(marketTemp['Market'], cvarTemp)
            beta = cov[1, 0] / cov[0, 0]
            cvaralpha.append(cvarPerformance[len(cvarPerformance) - 1] ** (252 / totalPeriod) - 1 - (
                rateLock + beta * (marketRet - rateLock)))

            cov = np.cov(marketTemp['Market'], utilTemp)
            beta = cov[1, 0] / cov[0, 0]
            utilalpha.append(utilPerformance[len(utilPerformance) - 1] ** (252 / totalPeriod) - 1 - (
                rateLock + beta * (marketRet - rateLock)))

            cov = np.cov(marketTemp['Market'], rpTemp)
            beta = cov[1, 0] / cov[0, 0]
            rpalpha.append(rpPerformance[len(rpPerformance) - 1] ** (252 / totalPeriod) - 1 - (
                rateLock + beta * (marketRet - rateLock)))

    benchalpha = np.array(benchalpha[int(len(benchalpha)*3/4):len(benchalpha)])
    mvoalpha = np.array(mvoalpha[int(len(mvoalpha) * 3 / 4):len(mvoalpha)])
    cvaralpha = np.array(cvaralpha[int(len(cvaralpha) * 3 / 4):len(cvaralpha)])
    utilalpha = np.array(utilalpha[int(len(utilalpha) * 3 / 4):len(utilalpha)])
    rpalpha = np.array(rpalpha[int(len(rpalpha) * 3 / 4):len(rpalpha)])

    benchmarkPerformance = np.array(benchmarkPerformance)
    mvoPerformance = np.array(mvoPerformance)
    rpPerformance = np.array(rpPerformance)
    cvarPerformance = np.array(cvarPerformance)
    utilPerformance = np.array(utilPerformance)

    benchmarkPerformanceul = np.array(benchmarkPerformanceul)
    mvoPerformanceul = np.array(mvoPerformanceul)
    rpPerformanceul = np.array(rpPerformanceul)
    cvarPerformanceul = np.array(cvarPerformanceul)
    utilPerformanceul = np.array(utilPerformanceul)

    rawMarket = market.iloc[start*period: end * period]['Market'] /market.iloc[start*period]['Market']

    if (showPlot):
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.plot(dates, benchmarkPerformance, c='g')
        plt.plot(dates, mvoPerformance, c='r')
        plt.plot(dates, rpPerformance, c='y')
        plt.plot(dates, cvarPerformance, c='b')
        plt.plot(dates, utilPerformance, c='m')
        plt.plot(dates, rawMarket, c = 'k')
        plt.gcf().autofmt_xdate()

        for xc in rebalanceDates:
            plt.axvline(x=xc, c="r", alpha=0.2)
        # plt.scatter(rebalanceDates, rebalanceValues, c="r", alpha=0.5)
        plt.gcf().autofmt_xdate()

        plt.legend(['Benchmark', 'MVO', 'Risk Parity', 'CVaR', 'Utility', 'Market Index'])

        plt.show()

    totalPeriod = (end - start)*period

    results = {}

    p = np.argmax(np.maximum.accumulate(benchmarkPerformance) - benchmarkPerformance)  # end of the period
    q = np.argmax(benchmarkPerformance[:p])  # start of period
    marketTemp = returnsMatrix(market.iloc[start * period:end * period+1])
    marketTemp = marketTemp[1:len(marketTemp)]
    marketRet = (market.iloc[end * period]['Market'] / market.iloc[start * period]['Market']) ** (252 / totalPeriod) - 1

    benchTemp = returnsMatrix2(benchmarkPerformance)
    cov = np.cov(marketTemp['Market'], benchTemp)
    beta = cov[1, 0] / cov[0, 0]
    alpha = benchmarkPerformance[len(benchmarkPerformance) - 1] ** (252 / totalPeriod) - 1 - (rateLock + beta * (
        marketRet - rateLock)) #* benchmarkWeights[1] - rateLock * benchmarkWeights[0]
    sharpe = ((benchmarkPerformance  [len(benchmarkPerformance)-1]** (252 / totalPeriod) - 1)-rateLock)/\
             (benchmarkPerformance.std()*252**0.5)*100

    results ['Benchmark'] = {'beat': int(alpha > 0), 'alpha': alpha, 'Sharpe': sharpe,
                       'Return': (benchmarkPerformance[len(benchmarkPerformance)-1])**(252/totalPeriod)-1,
                       'Drawdown': ((benchmarkPerformance[q] - benchmarkPerformance[p])
                                          /benchmarkPerformance[q])**(252/(p-q)), 'Beta': beta,
                             'ConAlpha': int(stats.ttest_1samp(benchalpha,0.0)[0]>0.05)}

    benchExcess = mvoPerformance[len(mvoPerformance)-1] - benchmarkPerformance[len(benchmarkPerformance)-1]

    mvoTemp = returnsMatrix2(mvoPerformance)
    cov = np.cov(marketTemp['Market'], mvoTemp)
    beta = cov[1, 0] / cov[0, 0]
    alpha = mvoPerformance[len(mvoPerformance)-1]**(252/totalPeriod)-1 - (rateLock + beta*(
        marketRet - rateLock))#* mvoWeights[1] - rateLock * mvoWeights[0]
    sharpe = ((mvoPerformance  [len(mvoPerformance)-1]** (252 / totalPeriod) - 1)-rateLock)/\
             (mvoPerformance.std()*252**0.5)*100

    p = np.argmax(np.maximum.accumulate(mvoPerformance) - mvoPerformance)  # end of the period
    q = np.argmax(mvoPerformance[:p])  # start of period
    ks = int(kstest2.main(mvoPerformance,benchmarkPerformance,critical, 'larger')[1] > critical)

    results ['MVO'] = {'beat': int(alpha > 0), 'alpha': alpha, 'KSbench': ks, 'Sharpe': sharpe,
                       'Return': (mvoPerformance[len(mvoPerformance)-1])**(252/totalPeriod)-1,
                       'Drawdown': ((mvoPerformance[q] - mvoPerformance[p])
                                          /mvoPerformance[q])**(252/(p-q)), 'Beta': beta, 'benchExcess': benchExcess,
                       'ConAlpha': int(stats.ttest_1samp(mvoalpha, 0.0)[0] > 0.05)}

    benchExcess = rpPerformance[len(rpPerformance) - 1] - benchmarkPerformance[len(benchmarkPerformance) - 1]
    cov = np.cov(marketTemp['Market'], returnsMatrix2(rpPerformance))
    beta = cov[1, 0] / cov[0, 0]
    alpha = rpPerformance[len(rpPerformance) - 1] ** (252 / totalPeriod) - 1 - (rateLock + beta * (
        marketRet - rateLock))# * rpWeights[1] - rateLock * rpWeights[0]
    sharpe = ((rpPerformance[len(rpPerformance) - 1] ** (252 / totalPeriod) - 1) - rateLock) /\
             (rpPerformance.std()*252**0.5)*100

    p = np.argmax(np.maximum.accumulate(rpPerformance) - rpPerformance)  # end of the period
    q = np.argmax(rpPerformance[:p])  # start of period
    ks = int(kstest2.main(rpPerformance, benchmarkPerformance, critical, 'larger')[1] > critical)
    results ['Risk Parity'] = {'beat': int(alpha > 0), 'alpha': alpha, 'KS': ks, 'Sharpe': sharpe,
                               'Return': (rpPerformance[len(rpPerformance)-1])**(252/totalPeriod)-1,
                               'Drawdown': ((rpPerformance[q] - rpPerformance[p])
                                            /rpPerformance[q])**(252/(p-q)), 'Beta': beta, 'benchExcess': benchExcess,
                               'ConAlpha': int(stats.ttest_1samp(rpalpha, 0.0)[0] > 0.05)}

    benchExcess = cvarPerformance[len(cvarPerformance) - 1] - benchmarkPerformance[len(benchmarkPerformance) - 1]

    cov = np.cov(marketTemp['Market'], returnsMatrix2(cvarPerformance))
    beta = cov[1, 0] / cov[0, 0]
    alpha = cvarPerformance[len(cvarPerformance) - 1] ** (252 / totalPeriod) - 1 - (rateLock + beta * (
        marketRet - rateLock))# * cvarWeights[1] - rateLock * cvarWeights[0]
    sharpe = ((cvarPerformance[len(cvarPerformance) - 1] ** (252 / totalPeriod) - 1) - rateLock) / \
             (cvarPerformance.std()*252**0.5)*100

    p = np.argmax(np.maximum.accumulate(cvarPerformance) - cvarPerformance)  # end of the period
    q = np.argmax(cvarPerformance[:p])  # start of period
    ks = int(kstest2.main(cvarPerformance, benchmarkPerformance, critical, 'larger')[1] > critical)
    results ['CVaR'] = {'beat': int(alpha > 0), 'alpha': alpha, 'KS': ks, 'Sharpe': sharpe,
                       'Return': (cvarPerformance[len(cvarPerformance)-1])**(252/totalPeriod)-1,
                       'Drawdown': ((cvarPerformance[q] - cvarPerformance[p])
                                          /cvarPerformance[q])**(252/(p-q)), 'Beta': beta, 'benchExcess': benchExcess,
                        'ConAlpha': int(stats.ttest_1samp(cvaralpha, 0.0)[0] > 0.05)}

    benchExcess = utilPerformance[len(utilPerformance) - 1] - benchmarkPerformance[len(benchmarkPerformance) - 1]

    cov = np.cov(marketTemp['Market'], returnsMatrix2(utilPerformance))
    beta = cov[1, 0] / cov[0, 0]
    alpha = utilPerformance[len(utilPerformance) - 1] ** (252 / totalPeriod) - 1 - (rateLock + beta * (
        marketRet - rateLock))# * utilWeights[1] - rateLock * utilWeights[0]
    sharpe = ((utilPerformance[len(utilPerformance) - 1] ** (252 / totalPeriod) - 1) - rateLock) / \
             (utilPerformance.std()*252**0.5)*100

    p = np.argmax(np.maximum.accumulate(utilPerformance) - utilPerformance)  # end of the period
    q = np.argmax(utilPerformance[:p])  # start of period
    ks = int(kstest2.main(utilPerformance, benchmarkPerformance, critical, 'larger')[1] > critical)
    results ['Utility'] = {'beat': int(alpha > 0), 'alpha': alpha, 'KS': ks, 'Sharpe': sharpe,
                           'Return': (utilPerformance[len(utilPerformance)-1])**(252/totalPeriod)-1,
                           'Drawdown': ((utilPerformance[q] - utilPerformance[p])
                                        /utilPerformance[q])**(252/(p-q)), 'Beta': beta, 'benchExcess': benchExcess,
                           'ConAlpha': int(stats.ttest_1samp(utilalpha, 0.0)[0] > 0.05)}

    return results
##########################


########### Backtesting ##################

btestlogBench = pd.DataFrame()
btestlogMVO = pd.DataFrame()
btestlogRP = pd.DataFrame()
btestlogCVaR = pd.DataFrame()
btestlogUtil = pd.DataFrame()

desiredRet = [0.05, 0.25]
start = [20, 32]
end = [8, 28]
trainingPeriod = [4, 16]
period = [63, 252]
gamma = [1, 5] # lam assumed to be the same

modelData = pd.read_csv('Data/MarketData.csv')
market = pd.read_csv('Data/Market.csv')

#btest = backtest(0.25, 20, 32, 8, 63, 5, 5, 0.05, 0, modelData, market, rate)
#print (btest)

# Parameters: desiredRet, start, end, trainingPeriod, period, gamma, lam, critical, showPlot, modelData, market, rate
for p in period:
    if p == 63:
        rate = pd.read_csv('Data/Rate3m.csv')
    else:
        rate = pd.read_csv('Data/Rate1y.csv')
        start[:] = [round(x / 4) for x in start]
        end[:] = [round(x / 4) for x in end]
        trainingPeriod[:] = [round(x / 4) for x in trainingPeriod]
    for dr in desiredRet:
        for s in start:
            for e in end:
                for tp in trainingPeriod:
                        for g in gamma:
                            bresult = backtest(dr, s, s+e, tp, p, g, g, 0.05, 0, modelData, market, rate)
                            config = {'desiredRet': dr, 'start': s, 'end': s+e, 'trainingPeriod': tp, 'period': p,
                                      'gamma': g, 'lam': g}
                            print (config)
                            btestlogBench = btestlogBench.append({**config, **bresult['Benchmark']}, ignore_index=True)
                            btestlogMVO = btestlogMVO.append({**config, **bresult['MVO']}, ignore_index=True)
                            btestlogRP = btestlogRP.append({**config, **bresult['Risk Parity']}, ignore_index=True)
                            btestlogCVaR = btestlogCVaR.append({**config, **bresult['CVaR']}, ignore_index=True)
                            btestlogUtil = btestlogUtil.append({**config, **bresult['Utility']}, ignore_index=True)

btestlogBench.to_csv('BacktestLogBench.csv', sep=',')
btestlogMVO.to_csv('BacktestLogMVO.csv', sep=',')
btestlogRP.to_csv('BacktestLogRP.csv', sep=',')
btestlogCVaR.to_csv('BacktestLogCVaR.csv', sep=',')
btestlogUtil.to_csv('BacktestLogUtil.csv', sep=',')

############################

