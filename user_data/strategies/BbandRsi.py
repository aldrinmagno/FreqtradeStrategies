"""
BbandRsi — Bollinger Band + RSI Mean-Reversion Scalper

Source: Custom beginner-friendly mean-reversion strategy
Timeframe: 5m
Description: Enters when RSI is oversold and price dips below the lower Bollinger Band,
             exits when RSI reaches overbought territory.
"""

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import pandas_ta as pta


class BbandRsi(IStrategy):

    INTERFACE_VERSION: int = 3

    timeframe: str = "5m"

    # ROI table — quick scalp targets
    minimal_roi: dict = {
        "0": 0.02,
        "10": 0.01,
        "30": 0.005,
    }

    stoploss: float = -0.04

    # Trailing stop configuration
    trailing_stop: bool = True
    trailing_stop_positive: float = 0.01
    trailing_stop_positive_offset: float = 0.015
    trailing_only_offset_is_reached: bool = True

    # Trade management
    max_open_trades: int = 5
    use_exit_signal: bool = True
    exit_profit_only: bool = False

    # -----------------------------------------------------------------------
    # Buy hyperopt parameters
    # -----------------------------------------------------------------------
    buy_rsi = IntParameter(15, 40, default=30, space="buy")
    buy_bb_factor = DecimalParameter(0.97, 1.0, default=1.0, space="buy",
                                     help="Entry when close < bb_lower * this factor")

    # -----------------------------------------------------------------------
    # Sell hyperopt parameters
    # -----------------------------------------------------------------------
    sell_rsi = IntParameter(60, 85, default=70, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # RSI (14)
        dataframe["rsi"] = pta.rsi(dataframe["close"], length=14)

        # Bollinger Bands (20-period, 2 std dev)
        bbands = pta.bbands(dataframe["close"], length=20, std=2.0)
        dataframe["bb_lowerband"] = bbands[f"BBL_20_2.0"]
        dataframe["bb_middleband"] = bbands[f"BBM_20_2.0"]
        dataframe["bb_upperband"] = bbands[f"BBU_20_2.0"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] < self.buy_rsi.value)
                & (dataframe["close"] < dataframe["bb_lowerband"] * self.buy_bb_factor.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] > self.sell_rsi.value)
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe
