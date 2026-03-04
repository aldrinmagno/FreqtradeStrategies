"""
ClucHAnix — Heikin-Ashi Noise-Filtered Bollinger Band Scalper

Source: https://github.com/freqtrade/freqtrade-strategies (community)
Timeframe: 5m (with 1h informative)
Description: Uses Heikin-Ashi candle smoothing over Bollinger Bands with a 1h ROCR filter
             and Fisher RSI exit signal. Custom progressive trailing stoploss.
"""

from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import numpy as np
import pandas_ta as pta


class ClucHAnix(IStrategy):

    INTERFACE_VERSION: int = 3

    timeframe: str = "5m"
    informative_timeframe: str = "1h"

    # ROI disabled — custom exits dominate
    minimal_roi: dict = {"0": 10}

    # Base stoploss disabled — custom stoploss handles exits
    stoploss: float = -0.99

    use_exit_signal: bool = True
    exit_profit_only: bool = False

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
        # Normalize RSI to -1..1 range
        rsi_norm = 0.1 * (rsi - 50)
        # Clamp to avoid infinity in log
        rsi_norm = rsi_norm.clip(-0.999, 0.999)
        # Fisher transform
        fisher = (np.log((1 + rsi_norm) / (1 - rsi_norm))).rolling(window=smoothing).mean()
        dataframe["fisher_rsi"] = fisher
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- 1h informative ---
        inf_tf = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
        inf_tf["rocr"] = pta.roc(inf_tf["close"], length=1) / 100 + 1  # ROCR as ratio
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

        # --- Entry helper columns ---
        dataframe["bbdelta"] = (dataframe["bb_middleband"] - dataframe["bb_lowerband"]).abs()
        dataframe["closedelta"] = (dataframe["close"] - dataframe["close"].shift(1)).abs()
        dataframe["tail"] = (dataframe["low"] - dataframe["close"]).abs()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["bbdelta"] > 0)
                & (dataframe["closedelta"] / (0.5 * dataframe["bbdelta"]) > 1.0)
                & (dataframe["tail"] / (0.5 * dataframe["bbdelta"]) < 1.0)
                & (dataframe["ha_close"] < dataframe["ha_open"])
                & (dataframe["ha_close"] < dataframe["ha_close"].shift(1))
                & (dataframe[f"rocr_{self.informative_timeframe}"] > 1.0)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["fisher_rsi"] > 0.3)
                & (dataframe["volume"] > 0)
            ),
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
        Progressive trailing stoploss: rises from -0.05 to -0.01 as profit
        increases from 0% to 2%.
        """
        if current_profit < 0.0:
            return -0.05

        if current_profit >= 0.02:
            return -0.01

        # Linear interpolation between -0.05 and -0.01
        return -0.05 + (current_profit / 0.02) * 0.04
