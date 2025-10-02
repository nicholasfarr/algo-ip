import pandas as pd
import numpy as np
from scipy.special import softmax

def calc_score(momentum, vol, vvol, vvol_scalar=2):
    return (momentum / (vol + (vvol_scalar * vvol))).__round__(4)

def generate_allocations(score_dict, max_allocation=0.1):
    symbols = list(score_dict.keys())
    raw_scores = np.array(list(score_dict.values()))
    weights = softmax(raw_scores)

    weights_dict = dict(zip(symbols, weights))
    for key, value in weights_dict.items():
        if value < 0.01:
            weights_dict[key] = 0
        elif value > max_allocation:
            weights_dict[key] = max_allocation
        else:
            weights_dict[key] = weights_dict[key].__round__(4)
    return weights_dict

def backtest_fixed_momentum(price_file="data_testing.csv", mom_metric="m20", vol_metric="sd10"):
    """
    Fixed backtest: shift momentum metrics BACKWARD by 20 days to eliminate look-ahead bias
    """
    df = pd.read_csv(price_file)

    price_cols = ["date"] + [c for c in df.columns if "_" not in c and c != "date"]
    mom_cols = [c for c in df.columns if (mom_metric in c) or ("date" in c)]
    vol_cols = [c for c in df.columns if (vol_metric in c) or ("date" in c)]

    prices = df.loc[:, price_cols]
    moms = df.loc[:, mom_cols]
    vols = df.loc[:, vol_cols]

    # SHIFT MOMENTUM BACKWARD BY 20 DAYS
    for col in moms.columns:
        if col != "date":
            moms[col] = moms[col].shift(20)

    vols_indexed = vols.set_index("date").sort_index()
    vvols = vols_indexed[vols_indexed.columns.drop("date", errors="ignore")].rolling(20).std()

    allocations = {}
    for d in pd.date_range("2011-01-01", "2025-08-01", freq="B"):
        d = d.strftime("%Y-%m-%d")
        day_data_prices = prices[prices["date"] == d]
        if len(day_data_prices) == 0:
            continue
        non_nan_cols = day_data_prices.iloc[0].index[(~day_data_prices.iloc[0].isna()) & (day_data_prices.iloc[0].index != "date")].tolist()
        if len(non_nan_cols) < 10:
            continue

        daily_scores = {}
        for symbol in non_nan_cols:
            mom_val = moms[moms["date"] == d][symbol + "_" + mom_metric].values
            vol_val = vols[vols["date"] == d][symbol + "_" + vol_metric].values
            vvol_val = vvols[vvols.index == d][symbol + "_" + vol_metric].values

            if len(mom_val) > 0 and len(vol_val) > 0 and len(vvol_val) > 0:
                if pd.notna(mom_val[0]) and pd.notna(vol_val[0]) and pd.notna(vvol_val[0]):
                    momentum = mom_val[0]
                    vol = vol_val[0]
                    vvol = vvol_val[0]
                    score = calc_score(momentum, vol, vvol)
                    daily_scores[symbol] = score

        if len(daily_scores) > 0:
            alloc = generate_allocations(daily_scores)
            allocations[d] = alloc

    return allocations


def backtest_contracts(allocations, price_file="data_testing.csv", initial_capital=100000, top_pct=0.25):
    """Run backtest with top X% of assets"""
    df = pd.read_csv(price_file)
    price_cols = [c for c in df.columns if "_" not in c and c != "date"]
    prices = df[["date"] + price_cols].set_index("date").sort_index()

    cash = initial_capital
    positions = {}
    equity_curve = []
    dates = sorted(allocations.keys())

    for i, date in enumerate(dates):
        if date not in prices.index:
            continue

        current_prices = prices.loc[date]
        position_value = sum(
            positions.get(sym, 0) * current_prices.get(sym, 0)
            for sym in positions
            if pd.notna(current_prices.get(sym))
        )
        portfolio_value = cash + position_value

        target_allocs = allocations[date]
        sorted_assets = sorted(target_allocs.items(), key=lambda x: x[1], reverse=True)
        valid_assets = [(sym, weight) for sym, weight in sorted_assets
                       if sym in current_prices and pd.notna(current_prices[sym]) and current_prices[sym] > 0]

        n_assets = len(valid_assets)
        if n_assets == 0:
            equity_curve.append(portfolio_value)
            continue

        n_select = max(1, int(n_assets * top_pct))
        top_assets = valid_assets[:n_select]

        # Sell all
        for symbol, contracts in positions.items():
            if pd.notna(current_prices.get(symbol)) and current_prices[symbol] > 0:
                cash += contracts * current_prices[symbol]
        positions = {}

        # Buy top assets equal weight
        allocation_per_asset = portfolio_value / n_select
        for symbol, _ in top_assets:
            price = current_prices[symbol]
            if pd.notna(price) and price > 0:
                num_contracts = int(allocation_per_asset / price)
                if num_contracts > 0:
                    cash -= num_contracts * price
                    positions[symbol] = num_contracts

        position_value = sum(
            positions.get(sym, 0) * current_prices.get(sym, 0)
            for sym in positions
            if pd.notna(current_prices.get(sym))
        )
        portfolio_value = cash + position_value
        equity_curve.append(portfolio_value)

    equity_series = pd.Series(equity_curve, index=dates[:len(equity_curve)])
    total_return = (equity_curve[-1] / initial_capital - 1) * 100
    daily_returns = equity_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
    max_dd = ((equity_series.cummax() - equity_series) / equity_series.cummax()).max() * 100

    return {
        "equity_curve": equity_series,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "final_value": round(equity_curve[-1], 2),
    }


if __name__ == "__main__":
    print("Running FIXED momentum backtest (shifted backward by 20 days)...")
    allocations = backtest_fixed_momentum()
    results = backtest_contracts(allocations, top_pct=0.25)

    print(f"\nFixed Momentum (Top 25%) Results:")
    print(f"Total Return: {results['total_return_pct']}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']}")
    print(f"Max Drawdown: {results['max_drawdown_pct']}%")
    print(f"Final Value: ${results['final_value']:,.2f}")
