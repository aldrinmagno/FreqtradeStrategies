# Freqtrade Scalping Strategies

A collection of open-source scalping strategies for [Freqtrade](https://www.freqtrade.io/), configured for **Bybit spot trading** on 5-minute timeframes with medium risk settings.

## Strategy Overview

| Strategy | Timeframe | Complexity | Best Pairs | Description |
|---|---|---|---|---|
| **BbandRsi** | 5m | Beginner | BTC, ETH, majors | Bollinger Band + RSI mean-reversion scalper |
| **ClucHAnix** | 5m + 1h | Intermediate | BTC, ETH, SOL | Heikin-Ashi noise-filtered BB scalper with progressive trailing stop |
| **CombinedBinHAndClucV8Hyper** | 5m + 1h | Advanced | All USDT pairs | Hyperopt-ready combined dip-buyer with 4 togglable entry conditions |
| **E0V1E** | 5m + 1h | Intermediate | BTC, ETH, altcoins | Live-validated Bollinger squeeze scalper (ClucHAnix derivative) |
| **NostalgiaForInfinityX6** | 5m + 15m/1h/1d | Advanced | All USDT pairs | Multi-signal system with 8 buy conditions, derisking, and multi-TF analysis |

## Installation

### 1. Install Freqtrade

```bash
pip install freqtrade
# or use the official install script:
# git clone https://github.com/freqtrade/freqtrade.git
# cd freqtrade && ./setup.sh -i
```

### 2. Install dependencies

```bash
pip install pandas_ta
```

### 3. Clone this repository

```bash
git clone <this-repo-url>
cd FreqtradeStrategies
```

### 4. Configure Bybit API keys

Edit `user_data/config/config_bybit_medium_risk.json` and fill in your Bybit API credentials:

```json
"exchange": {
    "name": "bybit",
    "key": "YOUR_API_KEY",
    "secret": "YOUR_API_SECRET"
}
```

**Important:** Also update `jwt_secret_key`, `username`, and `password` in the `api_server` section.

## Running Backtests

Download historical data first:

```bash
freqtrade download-data --exchange bybit --pairs BTC/USDT ETH/USDT SOL/USDT \
    --timeframes 5m 15m 1h 1d --days 90 \
    -c user_data/config/config_bybit_medium_risk.json
```

### Backtest each strategy

```bash
# BbandRsi
freqtrade backtesting --strategy BbandRsi \
    -c user_data/config/config_bybit_medium_risk.json \
    --timerange 20250101-

# ClucHAnix
freqtrade backtesting --strategy ClucHAnix \
    -c user_data/config/config_bybit_medium_risk.json \
    --timerange 20250101-

# CombinedBinHAndClucV8Hyper
freqtrade backtesting --strategy CombinedBinHAndClucV8Hyper \
    -c user_data/config/config_bybit_medium_risk.json \
    --timerange 20250101-

# E0V1E
freqtrade backtesting --strategy E0V1E \
    -c user_data/config/config_bybit_medium_risk.json \
    --timerange 20250101-

# NostalgiaForInfinityX6
freqtrade backtesting --strategy NostalgiaForInfinityX6 \
    -c user_data/config/config_bybit_medium_risk.json \
    --timerange 20250101-
```

## Running Hyperopt

Both **E0V1E** and **CombinedBinHAndClucV8Hyper** are fully hyperopt-ready. Use `SortinoHyperOptLossDaily` for best results:

```bash
# Hyperopt E0V1E (recommended: 500+ epochs)
freqtrade hyperopt --strategy E0V1E \
    --hyperopt-loss SortinoHyperOptLossDaily \
    -c user_data/config/config_bybit_medium_risk.json \
    --timerange 20250101- \
    -e 500 \
    --spaces buy sell

# Hyperopt CombinedBinHAndClucV8Hyper
freqtrade hyperopt --strategy CombinedBinHAndClucV8Hyper \
    --hyperopt-loss SortinoHyperOptLossDaily \
    -c user_data/config/config_bybit_medium_risk.json \
    --timerange 20250101- \
    -e 500 \
    --spaces buy sell
```

## Dry-Run (Paper Trading)

Launch a dry-run to test strategies without risking real funds:

```bash
freqtrade trade --strategy BbandRsi \
    -c user_data/config/config_bybit_medium_risk.json
```

The config has `"dry_run": true` by default with a $1000 paper wallet.

## Project Structure

```
user_data/
├── config/
│   └── config_bybit_medium_risk.json   # Bybit spot config (medium risk)
└── strategies/
    ├── BbandRsi.py                      # Beginner BB + RSI scalper
    ├── ClucHAnix.py                     # Heikin-Ashi BB scalper
    ├── CombinedBinHAndClucV8Hyper.py    # Hyperopt-ready dip buyer
    ├── E0V1E.py                         # Live-validated BB squeeze
    └── NostalgiaForInfinityX6.py        # Multi-signal institutional system
```

## Warnings

- **Always dry-run first.** Test every strategy in paper trading mode before using real funds.
- **Past performance does not guarantee future results.** Backtesting results may not reflect live trading performance due to slippage, liquidity, and market conditions.
- **Bybit API key setup:** Create API keys at [Bybit](https://www.bybit.com/) with spot trading permissions only. Never enable withdrawal permissions for bot API keys.
- **Risk management:** The config uses `"stake_amount": "unlimited"` which distributes your balance across `max_open_trades`. Adjust `tradable_balance_ratio` to control capital exposure.
- **Do not share your API keys.** Keep your `.env` file and config credentials private.
