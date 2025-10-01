import pandas as pd
import numpy as np

from bentesting import generate_allocations, backtest

allocations = backtest()

def backtest_with_allocations(price_file="data_testing.csv", allocs=None, max_allocation=0.1):
    """
    price_file: CSV with 'date' column and asset prices
    allocs: dict of {date: {asset: weight}}, recommended allocations
    """
    df = pd.read_csv(price_file)
    df = df.sort_values("date").reset_index(drop=True)

    # compute simple daily returns for each asset
    returns = df.set_index("date").pct_change().dropna()

    equity_curve = [1.0]  # start at 1.0
    for i, d in enumerate(returns.index):
        if d not in allocs:
            equity_curve.append(equity_curve[-1])  # no allocation this day
            continue

        weights = pd.Series(allocs[d])
        # normalize so weights sum to 1
        weights = weights / weights.sum()

        day_ret = (returns.loc[d] * weights).sum()
        equity_curve.append(equity_curve[-1] * (1 + day_ret))

    equity_curve = pd.Series(equity_curve[1:], index=returns.index)

    # metrics
    total_return = equity_curve.iloc[-1] - 1
    ann_return = (equity_curve.iloc[-1])**(252/len(equity_curve)) - 1
    ann_vol = equity_curve.pct_change().std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    return {
        "total_return": round(total_return, 4),
        "annualized_return": round(ann_return, 4),
        "annualized_vol": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4),
        "equity_curve": equity_curve
    }



def backtest_top_decile(prices: pd.DataFrame, allocations: dict, top_frac=0.1, start_equity=1_000_000):
    """
    prices: DataFrame with dates as index, columns = tickers
    allocations: {date: {symbol: score}} where scores >= 0
    """
    equity = start_equity
    curve = []
    returns = prices.pct_change().fillna(0)

    for d in returns.index:
        if d not in allocations:
            curve.append((d, equity))
            continue

        # Convert dict to Series and sort by score
        scores = pd.Series(allocations[d])
        scores = scores.reindex(prices.columns).fillna(0.0)
        n_keep = max(1, int(len(scores) * top_frac))  # top 10%

        top_assets = scores.nlargest(n_keep)

        # Equal-weight or score-weight
        weights = top_assets / top_assets.sum()

        # Compute portfolio return
        day_ret = (returns.loc[d] * weights).sum()
        equity *= (1 + day_ret)
        curve.append((d, equity))

    curve = pd.Series(dict(curve)).sort_index()
    total_return = curve.iloc[-1] / start_equity - 1
    sharpe = (curve.pct_change().mean() / curve.pct_change().std()) * np.sqrt(252)

    return {
        "equity_curve": curve,
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe, 2)
    }