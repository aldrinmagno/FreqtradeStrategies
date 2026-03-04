"""
E0V1E — Live-Validated Bollinger Squeeze Scalper

Source: https://github.com/ssssi/freqtrade-strategies (by ssssi)
Timeframe: 5m (with 1h informative)
Description: Simplified ClucHAnix derivative optimized for live trading. Uses Bollinger Band
             squeeze with Heikin-Ashi confirmation and 1h ROCR trend filter.

Recommended hyperopt loss function: SortinoHyperOptLossDaily
    freqtrade hyperopt --strategy E0V1E --hyperopt-loss SortinoHyperOptLossDaily -e 500
"""

from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    IntParameter,
    merge_informative_pair,
)
from pandas import DataFrame
import numpy as np
import pandas_ta as pta


class E0V1E(IStrategy):

    INTERFACE_VERSION: int = 3

    timeframe: str = "5m"
    informative_timeframe: str = "1h"

    # ROI disabled — all exits via signals
    minimal_roi: dict = {"0": 100}

    # Base stoploss disabled — custom stoploss handles exits
    stoploss: float = -0.99

    use_exit_signal: bool = True
    exit_profit_only: bool = False

    # -----------------------------------------------------------------------
    # Buy hyperopt parameters
    # -----------------------------------------------------------------------
    buy_bbdelta_close = DecimalParameter(0.008, 0.02, default=0.012, space="buy")
    buy_closedelta_factor = DecimalParameter(0.6, 1.2, default=0.9, space="buy")
    buy_tail_factor = DecimalParameter(0.2, 0.6, default=0.35, space="buy")
    buy_rocr_1h = DecimalParameter(1.0, 1.1, default=1.04, space="buy")

    # -----------------------------------------------------------------------
    # Sell hyperopt parameters
    # -----------------------------------------------------------------------
    sell_fisher_rsi = DecimalParameter(0.0, 0.5, default=0.2, space="sell")
    sell_rsi = IntParameter(50, 80, default=65, space="sell")

    def informative_pairs(self) -> list:
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_timeframe) for pair in pairs]

    def _compute_heikin_ashi(self, dataframe: DataFrame) -> DataFrame:
        ha_close = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4

        ha_open = ha_close.copy()
        ha_open.iloc[0] = (dataframe["open"].iloc[0] + dataframe["close"].iloc[0]) / 2
        for i in range(1, len(dataframe)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

        dataframe["ha_open"] = ha_open
        dataframe["ha_close"] = ha_close
        dataframe["ha_high"] = dataframe[["high", "ha_open", "ha_close"]].max(axis=1)
        dataframe["ha_low"] = dataframe[["low", "ha_open", "ha_close"]].min(axis=1)
        return dataframe

    def _compute_fisher_rsi(self, dataframe: DataFrame, length: int = 5, smoothing: int = 5) -> DataFrame:
        rsi = pta.rsi(dataframe["close"], length=length)
        rsi_norm = 0.1 * (rsi - 50)
        rsi_norm = rsi_norm.clip(-0.999, 0.999)
        fisher = (np.log((1 + rsi_norm) / (1 - rsi_norm))).rolling(window=smoothing).mean()
        dataframe["fisher_rsi"] = fisher
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- 1h informative ---
        inf_tf = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
        inf_tf["rocr"] = pta.roc(inf_tf["close"], length=1) / 100 + 1
        dataframe = merge_informative_pair(dataframe, inf_tf, self.timeframe, self.informative_timeframe, ffill=True)

        # --- Bollinger Bands (20, 2) ---
        bbands = pta.bbands(dataframe["close"], length=20, std=2.0)
        dataframe["bb_lowerband"] = bbands["BBL_20_2.0"]
        dataframe["bb_middleband"] = bbands["BBM_20_2.0"]
        dataframe["bb_upperband"] = bbands["BBU_20_2.0"]

        # --- Heikin-Ashi ---
        dataframe = self._compute_heikin_ashi(dataframe)

        # --- Fisher RSI ---
        dataframe = self._compute_fisher_rsi(dataframe)

        # --- RSI ---
        dataframe["rsi"] = pta.rsi(dataframe["close"], length=14)

        # --- Entry helper columns ---
        dataframe["bbdelta"] = (dataframe["bb_middleband"] - dataframe["bb_lowerband"]).abs()
        dataframe["closedelta"] = (dataframe["close"] - dataframe["close"].shift(1)).abs()
        dataframe["tail"] = (dataframe["low"] - dataframe["close"]).abs()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["bbdelta"] > dataframe["close"] * self.buy_bbdelta_close.value)
                & (dataframe["closedelta"] > dataframe["bbdelta"] * self.buy_closedelta_factor.value)
                & (dataframe["tail"] < dataframe["bbdelta"] * self.buy_tail_factor.value)
                & (dataframe["ha_close"] < dataframe["bb_lowerband"])
                & (dataframe[f"rocr_{self.informative_timeframe}"] > self.buy_rocr_1h.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["fisher_rsi"] > self.sell_fisher_rsi.value)
                | (dataframe["rsi"] > self.sell_rsi.value)
            )
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Exchange stoploss safety net at -0.15. Base stoploss at -0.99
        allows signal-driven exits to dominate.
        """
        return -0.15
