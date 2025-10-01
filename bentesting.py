import pandas as pd
from scipy.special import softmax
import numpy as np

def calc_score(momentum,vol,vvol,vvol_scalar=2): # input is momentum, vol, and vol of volatility for a day and output is the score.
    return (momentum/(vol + (vvol_scalar*vvol))).__round__(4)

def generate_allocations(score_dict,max_allocation=0.1): # input is score dictionary for all assets for a day and output is reccomended percentages allocated based on softmax normalizer
    symbols = list(score_dict.keys())
    raw_scores = np.array(list(score_dict.values()))
    weights = softmax(raw_scores)

    weights_dict = dict(zip(symbols, weights))
    for key,value in weights_dict.items():
        if value < 0.01:
            weights_dict[key] = 0
        elif value > max_allocation:
            weights_dict[key] = max_allocation
        else:
            weights_dict[key] = weights_dict[key].__round__(4)
    return weights_dict


def backtest(): # general purpose backtesting function for now
    vol_metric = "sd10"
    mom_metric = "m20"
    df = pd.read_csv("data_testing.csv")

    price_cols = [c for c in df.columns if "_" not in c]
    mom_cols = [c for c in df.columns if (mom_metric in c) or ("date" in c)]
    vol_cols = [c for c in df.columns if (vol_metric in c) or ("date" in c)]

    prices = df.loc[:,price_cols]
    moms = df.loc[:,mom_cols]
    vols = df.loc[:,vol_cols]

    vols_indexed = vols.set_index("date").sort_index()

    # compute vol-of-vol for all assets
    vvols = vols_indexed[vols_indexed.columns.drop("date", errors="ignore")].rolling(20).std()
    allocations = {}
    for d in pd.date_range("2011-01-01", "2025-08-01",freq="B"):
        d = d.strftime("%Y-%m-%d")
        day_data_prices = prices[prices["date"] == d]
        if len(day_data_prices) == 0: # skips over days where no data exists
            continue
        non_nan_cols = day_data_prices.iloc[0].index[(~day_data_prices.iloc[0].isna()) & (day_data_prices.iloc[0].index != "date")].tolist()
        if len(non_nan_cols) < 10: # gets rid of days where most assets arent trading
            continue

        daily_scores = {}
        for symbol in non_nan_cols: # for each symbol that has values for a given day, pull the momentum, vol, and vvol and compute score from that
            momentum = moms[moms["date"] == d][symbol+"_"+mom_metric].values[0]
            vol = vols[vols["date"] == d][symbol+"_"+vol_metric].values[0]
            vvol = vvols[vvols.index == d][symbol+"_"+vol_metric].values[0]
            score = calc_score(momentum,vol,vvol)
            daily_scores[symbol] = score
        alloc = generate_allocations(daily_scores)
        allocations[d] = alloc
    return allocations
backtest()
