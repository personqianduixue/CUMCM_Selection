import math
import numpy
import numpy.random as nrand
import pulp
from pandas_datareader import data


"""
Note - for some of the metrics the absolute value is returns. This is because if the risk (loss) is higher we want to
discount the expected excess return from the portfolio by a higher amount. Therefore risk should be positive.
"""


def vol(returns):
    # Return the standard deviation of returns
    return numpy.std(returns)


def beta(returns, market):
    # Create a matrix of [returns, market]
    m = numpy.matrix([returns, market])
    # Return the covariance of m divided by the standard deviation of the market returns
    return numpy.cov(m)[0][1] / numpy.std(market)


def lpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = numpy.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return numpy.sum(diff ** order) / len(returns)


def hpm(returns, threshold, order):
    # This method returns a higher partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = numpy.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the returns and the threshold
    diff = returns - threshold_array
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return numpy.sum(diff ** order) / len(returns)


def var(returns, alpha):
    # This method calculates the historical simulation var of the returns
    sorted_returns = numpy.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index])


def cvar(returns, alpha):
    # This method calculates the condition VaR of the returns
    sorted_returns = numpy.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    # Return the average VaR
    # CVaR should be positive
    return abs(sum_var / index)


def prices(returns, base):
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return numpy.array(s)


def dd(returns, tau):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)


def max_dd(returns):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)


def average_dd(returns, periods):
    # Returns the average maximum drawdown over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def average_dd_squared(returns, periods):
    # Returns the average maximum drawdown squared over n periods
    drawdowns = []
    for i in range(0, len(returns)):
        drawdown_i = math.pow(dd(returns, i), 2.0)
        drawdowns.append(drawdown_i)
    drawdowns = sorted(drawdowns)
    total_dd = abs(drawdowns[0])
    for i in range(1, periods):
        total_dd += abs(drawdowns[i])
    return total_dd / periods


def treynor_ratio(er, returns, market, rf):
    return (er - rf) / beta(returns, market)


def sharpe_ratio(er, returns, rf):
    return (er - rf) / vol(returns)


def information_ratio(returns, benchmark):
    diff = returns - benchmark
    return numpy.mean(diff) / vol(diff)


def modigliani_ratio(er, returns, benchmark, rf):
    np_rf = numpy.empty(len(returns))
    np_rf.fill(rf)
    rdiff = returns - np_rf
    bdiff = benchmark - np_rf
    return (er - rf) * (vol(rdiff) / vol(bdiff)) + rf


def excess_var(er, returns, rf, alpha):
    return (er - rf) / var(returns, alpha)


def conditional_sharpe_ratio(er, returns, rf, alpha):
    return (er - rf) / cvar(returns, alpha)


def omega_ratio(er, returns, rf, target=0):
    return (er - rf) / lpm(returns, target, 1)


def sortino_ratio(er, returns, rf, target=0):
    return (er - rf) / math.sqrt(lpm(returns, target, 2))


def kappa_three_ratio(er, returns, rf, target=0):
    return (er - rf) / math.pow(lpm(returns, target, 3), float(1/3))


def gain_loss_ratio(returns, target=0):
    return hpm(returns, target, 1) / lpm(returns, target, 1)


def upside_potential_ratio(returns, target=0):
    return hpm(returns, target, 1) / math.sqrt(lpm(returns, target, 2))


def calmar_ratio(er, returns, rf):
    return (er - rf) / max_dd(returns)


def sterling_ration(er, returns, rf, periods):
    return (er - rf) / average_dd(returns, periods)


def burke_ratio(er, returns, rf, periods):
    return (er - rf) / math.sqrt(average_dd_squared(returns, periods))


def test_risk_metrics():
    # This is just a testing method
    r = nrand.uniform(-1, 1, 50)
    m = nrand.uniform(-1, 1, 50)
    print("vol =", vol(r))
    print("beta =", beta(r, m))
    print("hpm(0.0)_1 =", hpm(r, 0.0, 1))
    print("lpm(0.0)_1 =", lpm(r, 0.0, 1))
    print("VaR(0.05) =", var(r, 0.05))
    print("CVaR(0.05) =", cvar(r, 0.05))
    print("Drawdown(5) =", dd(r, 5))
    print("Max Drawdown =", max_dd(r))


def test_risk_adjusted_metrics():
    # Returns from the portfolio (r) and market (m)
    r = nrand.uniform(-1, 1, 50)
    m = nrand.uniform(-1, 1, 50)
    # Expected return
    e = numpy.mean(r)
    # Risk free rate
    f = 0.06
    # Risk-adjusted return based on Volatility
    print("Treynor Ratio =", treynor_ratio(e, r, m, f))
    print("Sharpe Ratio =", sharpe_ratio(e, r, f))
    print("Information Ratio =", information_ratio(r, m))
    # Risk-adjusted return based on Value at Risk
    print("Excess VaR =", excess_var(e, r, f, 0.05))
    print("Conditional Sharpe Ratio =", conditional_sharpe_ratio(e, r, f, 0.05))
    # Risk-adjusted return based on Lower Partial Moments
    print("Omega Ratio =", omega_ratio(e, r, f))
    print("Sortino Ratio =", sortino_ratio(e, r, f))
    print("Kappa 3 Ratio =", kappa_three_ratio(e, r, f))
    print("Gain Loss Ratio =", gain_loss_ratio(r))
    print("Upside Potential Ratio =", upside_potential_ratio(r))
    # Risk-adjusted return based on Drawdown risk
    print("Calmar Ratio =", calmar_ratio(e, r, f))
    print("Sterling Ratio =", sterling_ration(e, r, f, 5))
    print("Burke Ratio =", burke_ratio(e, r, f, 5))


def PortfolioRiskTarget(mu, scen, CVaR_target=1, lamb=1, max_weight=1, min_weight=None, cvar_alpha=0.05):
    """ This function finds the optimal enhanced index portfolio according to some benchmark. The portfolio corresponds to the tangency portfolio where risk is evaluated according to the CVaR of the tracking error. The model is formulated using fractional programming.

    Parameters
    ----------
    mu : pandas.Series with float values
        asset point forecast
    scen : pandas.DataFrame with float values
        Asset scenarios
    max_weight : float
        Maximum allowed weight
    cvar_alpha : float
        Alpha value used to evaluate Value-at-Risk one

    Returns
    -------
    float
        Asset weights in an optimal portfolio

    """

    # define index
    i_idx = mu.index
    j_idx = scen.index

    # number of scenarios
    N = scen.shape[0]

    # define variables
    x = pulp.LpVariable.dicts("x", ((i) for i in i_idx),
                              lowBound=0,
                              cat='Continuous')

    # loss deviation
    VarDev = pulp.LpVariable.dicts("VarDev", ((t) for t in j_idx),
                                   lowBound=0,
                                   cat='Continuous')

    # value at risk
    VaR = pulp.LpVariable("VaR", lowBound=0,
                          cat='Continuous')
    CVaR = pulp.LpVariable("CVaR", lowBound=0,
                           cat='Continuous')

    # binary variable connected to cardinality constraints
    b_z = pulp.LpVariable.dicts("b_z", ((i) for i in i_idx),
                                cat='Binary')

    #####################################
    ## define model
    model = pulp.LpProblem("Mean-CVaR Optimization", pulp.LpMaximize)

    #####################################
    ## Objective Function

    model += lamb * (pulp.lpSum([mu[i] * x[i] for i in i_idx])) - (1 - lamb) * CVaR

    #####################################
    # constraint

    # calculate CVaR
    for t in j_idx:
        model += -pulp.lpSum([scen.loc[t, i] * x[i] for i in i_idx]) - VaR <= VarDev[t]

    model += VaR + 1 / (N * cvar_alpha) * pulp.lpSum([VarDev[t] for t in j_idx]) == CVaR

    model += CVaR <= CVaR_target

    ### price*number of products cannot exceed budget
    model += pulp.lpSum([x[i] for i in i_idx]) == 1

    ### Concentration limits
    # set max limits so it cannot not be larger than a fixed value
    ###
    for i in i_idx:
        model += x[i] <= max_weight

    ### Add minimum weight constraint, either zero or atleast minimum weight
    if min_weight is not None:

        for i in i_idx:
            model += x[i] >= min_weight * b_z[i]
            model += x[i] <= b_z[i]

    # solve model
    model.solve()

    # print an error if the model is not optimal
    if pulp.LpStatus[model.status] != 'Optimal':
        print("Whoops! There is an error! The model has error status:" + pulp.LpStatus[model.status])

    # Get positions
    if pulp.LpStatus[model.status] == 'Optimal':

        # print variables
        var_model = dict()
        for variable in model.variables():
            var_model[variable.name] = variable.varValue

        # solution with variable names
        var_model = pd.Series(var_model, index=var_model.keys())

        long_pos = [i for i in var_model.keys() if i.startswith("x")]

        # total portfolio with negative values as short positions
        port_total = pd.Series(var_model[long_pos].values, index=[t[2:] for t in var_model[long_pos].index])

        opt_port = port_total

    # set flooting data points to zero and normalize
    opt_port[opt_port < 0.000001] = 0
    opt_port = opt_port / sum(opt_port)

    # return portfolio, CVaR, and alpha
    return opt_port, var_model["CVaR"], sum(mu * port_total)


def PortfolioLambda(mu, scen, max_weight=1, min_weight=None, cvar_alpha=0.05, ft_points=15):
    # asset names
    assets = mu.index

    # column names
    col_names = mu.index.values.tolist()
    col_names.extend(["Mu", "CVaR", "STAR"])
    # number of frontier points

    # store portfolios
    portfolio_ft = pd.DataFrame(columns=col_names, index=list(range(ft_points)))

    # maximum risk portfolio
    lamb = 0.99999
    max_risk_port, max_risk_CVaR, max_risk_mu = PortfolioRiskTarget(mu=mu, scen=scen, CVaR_target=100, lamb=lamb,
                                                                    max_weight=max_weight, min_weight=min_weight,
                                                                    cvar_alpha=cvar_alpha)
    portfolio_ft.loc[ft_points - 1, assets] = max_risk_port
    portfolio_ft.loc[ft_points - 1, "Mu"] = max_risk_mu
    portfolio_ft.loc[ft_points - 1, "CVaR"] = max_risk_CVaR
    portfolio_ft.loc[ft_points - 1, "STAR"] = max_risk_mu / max_risk_CVaR

    # minimum risk portfolio
    lamb = 0.00001
    min_risk_port, min_risk_CVaR, min_risk_mu = PortfolioRiskTarget(mu=mu, scen=scen, CVaR_target=100, lamb=lamb,
                                                                    max_weight=max_weight, min_weight=min_weight,
                                                                    cvar_alpha=cvar_alpha)
    portfolio_ft.loc[0, assets] = min_risk_port
    portfolio_ft.loc[0, "Mu"] = min_risk_mu
    portfolio_ft.loc[0, "CVaR"] = min_risk_CVaR
    portfolio_ft.loc[0, "STAR"] = min_risk_mu / min_risk_CVaR

    # CVaR step size
    step_size = (max_risk_CVaR - min_risk_CVaR) / ft_points  # CVaR step size

    # calculate all frontier portfolios
    for i in range(1, ft_points - 1):
        CVaR_target = min_risk_CVaR + step_size * i
        i_risk_port, i_risk_CVaR, i_risk_mu = PortfolioRiskTarget(mu=mu, scen=scen, CVaR_target=CVaR_target, lamb=1,
                                                                  max_weight=max_weight, min_weight=min_weight,
                                                                  cvar_alpha=cvar_alpha)
        portfolio_ft.loc[i, assets] = i_risk_port
        portfolio_ft.loc[i, "Mu"] = i_risk_mu
        portfolio_ft.loc[i, "CVaR"] = i_risk_CVaR
        portfolio_ft.loc[i, "STAR"] = i_risk_mu / i_risk_CVaR

    return portfolio_ft
if __name__ == "__main__":
    test_risk_metrics()
    test_risk_adjusted_metrics()
