"""
NostalgiaForInfinityX6 — Institutional-Grade Multi-Signal System (Simplified)

Source: https://github.com/iterativv/NostalgiaForInfinity (simplified adaptation)
Timeframe: 5m (informative: 15m, 1h, 1d)
Description: Faithful simplified version of the NostalgiaForInfinityX architecture.
             Features 8 distinct buy conditions, 5 sell conditions, derisking system,
             and multi-timeframe indicator analysis.

Blacklisted leveraged tokens:
    *UP/USDT, *DOWN/USDT, *BULL/USDT, *BEAR/USDT, *2L/USDT, *2S/USDT,
    *3L/USDT, *3S/USDT, *4L/USDT, *4S/USDT, *5L/USDT, *5S/USDT
    These should be excluded via pair_blacklist in config.
"""

from functools import reduce
from freqtrade.strategy import (
    IStrategy,
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    merge_informative_pair,
)
from pandas import DataFrame
import numpy as np
import pandas_ta as pta


class NostalgiaForInfinityX6(IStrategy):

    INTERFACE_VERSION: int = 3

    timeframe: str = "5m"

    # Fully signal-driven
    minimal_roi: dict = {"0": 100}

    stoploss: float = -0.99

    use_exit_signal: bool = True
    exit_profit_only: bool = False
    ignore_roi_if_entry_signal: bool = True

    # Informative timeframes
    inf_timeframes: list = ["15m", "1h", "1d"]

    # -----------------------------------------------------------------------
    # Buy condition toggles
    # -----------------------------------------------------------------------
    buy_cond_1_enabled = BooleanParameter(default=True, space="buy")
    buy_cond_2_enabled = BooleanParameter(default=True, space="buy")
    buy_cond_3_enabled = BooleanParameter(default=True, space="buy")
    buy_cond_4_enabled = BooleanParameter(default=True, space="buy")
    buy_cond_5_enabled = BooleanParameter(default=True, space="buy")
    buy_cond_6_enabled = BooleanParameter(default=True, space="buy")
    buy_cond_7_enabled = BooleanParameter(default=True, space="buy")
    buy_cond_8_enabled = BooleanParameter(default=True, space="buy")

    # -----------------------------------------------------------------------
    # Buy hyperopt parameters — per condition
    # -----------------------------------------------------------------------
    # Condition 1 — EMA crossover + RSI
    buy_1_rsi = IntParameter(25, 50, default=40, space="buy")

    # Condition 2 — BB lower band bounce + volume
    buy_2_rsi = IntParameter(20, 40, default=30, space="buy")
    buy_2_mfi = IntParameter(15, 40, default=30, space="buy")

    # Condition 3 — Williams %R oversold
    buy_3_willr = IntParameter(-99, -70, default=-85, space="buy")
    buy_3_rsi = IntParameter(20, 45, default=35, space="buy")

    # Condition 4 — VWAP dip
    buy_4_vwap_factor = DecimalParameter(0.96, 1.0, default=0.99, space="buy")
    buy_4_rsi = IntParameter(20, 45, default=35, space="buy")
    buy_4_rsi_1h_min = IntParameter(30, 55, default=40, space="buy")

    # Condition 5 — 1h StochRSI oversold
    buy_5_stochrsi_k = IntParameter(10, 35, default=20, space="buy")
    buy_5_stochrsi_d = IntParameter(10, 35, default=20, space="buy")
    buy_5_rsi = IntParameter(20, 45, default=35, space="buy")

    # Condition 6 — EMA 50/200 golden zone
    buy_6_rsi = IntParameter(15, 40, default=30, space="buy")
    buy_6_willr_1h = IntParameter(-99, -60, default=-75, space="buy")

    # Condition 7 — BB squeeze
    buy_7_bb_width = DecimalParameter(0.02, 0.08, default=0.04, space="buy")
    buy_7_rsi = IntParameter(20, 42, default=32, space="buy")
    buy_7_rsi_1h_min = IntParameter(35, 60, default=45, space="buy")

    # Condition 8 — Multi-TF RSI alignment
    buy_8_rsi = IntParameter(15, 38, default=28, space="buy")
    buy_8_rsi_15m = IntParameter(20, 45, default=35, space="buy")
    buy_8_rsi_1h = IntParameter(30, 55, default=45, space="buy")

    # -----------------------------------------------------------------------
    # Sell hyperopt parameters
    # -----------------------------------------------------------------------
    # Condition 1 — RSI overbought
    sell_1_rsi = IntParameter(60, 85, default=70, space="sell")

    # Condition 2 — Williams %R overbought
    sell_2_willr = IntParameter(-20, -3, default=-10, space="sell")
    sell_2_rsi = IntParameter(55, 80, default=65, space="sell")

    # Condition 3 — EMA bearish crossover
    sell_3_rsi = IntParameter(50, 75, default=60, space="sell")

    # Condition 4 — MFI overbought
    sell_4_mfi = IntParameter(70, 95, default=80, space="sell")
    sell_4_bb_factor = DecimalParameter(0.97, 1.01, default=0.99, space="sell")

    # Condition 5 — 1h StochRSI overbought
    sell_5_stochrsi = IntParameter(65, 95, default=80, space="sell")
    sell_5_rsi = IntParameter(55, 80, default=65, space="sell")

    # Derisking threshold
    derisk_profit_threshold = DecimalParameter(-0.08, -0.01, default=-0.03, space="sell")

    def informative_pairs(self) -> list:
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for tf in self.inf_timeframes:
            informative_pairs.extend([(pair, tf) for pair in pairs])
        return informative_pairs

    def _stochastic_rsi(
        self,
        dataframe: DataFrame,
        rsi_length: int = 14,
        stoch_length: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> tuple:
        """Stochastic RSI indicator."""
        rsi = pta.rsi(dataframe["close"], length=rsi_length)
        stoch_rsi_k = (
            (rsi - rsi.rolling(window=stoch_length).min())
            / (rsi.rolling(window=stoch_length).max() - rsi.rolling(window=stoch_length).min())
        ) * 100
        stoch_rsi_k = stoch_rsi_k.rolling(window=smooth_k).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d).mean()
        return stoch_rsi_k, stoch_rsi_d

    def _williams_r(self, dataframe: DataFrame, length: int = 14) -> "Series":
        """Williams %R indicator."""
        highest_high = dataframe["high"].rolling(window=length).max()
        lowest_low = dataframe["low"].rolling(window=length).min()
        wr = ((highest_high - dataframe["close"]) / (highest_high - lowest_low)) * -100
        return wr

    def _compute_vwap(self, dataframe: DataFrame) -> "Series":
        """Compute VWAP (Volume Weighted Average Price)."""
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        cumulative_tp_vol = (typical_price * dataframe["volume"]).cumsum()
        cumulative_vol = dataframe["volume"].cumsum()
        return cumulative_tp_vol / cumulative_vol

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # =====================================================================
        # 5m indicators
        # =====================================================================
        # EMAs
        for length in [5, 8, 13, 21, 50, 200]:
            dataframe[f"ema_{length}"] = pta.ema(dataframe["close"], length=length)

        # RSI
        dataframe["rsi_14"] = pta.rsi(dataframe["close"], length=14)

        # Bollinger Bands
        bb = pta.bbands(dataframe["close"], length=20, std=2.0)
        dataframe["bb_lowerband"] = bb["BBL_20_2.0"]
        dataframe["bb_middleband"] = bb["BBM_20_2.0"]
        dataframe["bb_upperband"] = bb["BBU_20_2.0"]
        dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]

        # Williams %R
        dataframe["willr_14"] = self._williams_r(dataframe, length=14)

        # VWAP
        dataframe["vwap"] = self._compute_vwap(dataframe)

        # MFI
        dataframe["mfi_14"] = pta.mfi(
            dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"], length=14
        )

        # SMA 200 for trend
        dataframe["sma_200"] = pta.sma(dataframe["close"], length=200)

        # =====================================================================
        # 15m informative
        # =====================================================================
        inf_15m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="15m")
        inf_15m["rsi_14_15m"] = pta.rsi(inf_15m["close"], length=14)
        dataframe = merge_informative_pair(dataframe, inf_15m, self.timeframe, "15m", ffill=True)

        # =====================================================================
        # 1h informative
        # =====================================================================
        inf_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1h")
        inf_1h["rsi_14_1h"] = pta.rsi(inf_1h["close"], length=14)
        for length in [5, 8, 13, 21, 50, 200]:
            inf_1h[f"ema_{length}_1h"] = pta.ema(inf_1h["close"], length=length)
        inf_1h["willr_14_1h"] = self._williams_r(inf_1h, length=14)
        stoch_k, stoch_d = self._stochastic_rsi(inf_1h)
        inf_1h["stochrsi_k_1h"] = stoch_k
        inf_1h["stochrsi_d_1h"] = stoch_d
        dataframe = merge_informative_pair(dataframe, inf_1h, self.timeframe, "1h", ffill=True)

        # =====================================================================
        # 1d informative
        # =====================================================================
        inf_1d = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1d")
        inf_1d["ema_200_1d"] = pta.ema(inf_1d["close"], length=200)
        inf_1d["rsi_14_1d"] = pta.rsi(inf_1d["close"], length=14)
        dataframe = merge_informative_pair(dataframe, inf_1d, self.timeframe, "1d", ffill=True)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions: list = []

        # -----------------------------------------------------------------
        # Buy condition 1 — EMA crossover with RSI confirmation
        # -----------------------------------------------------------------
        if self.buy_cond_1_enabled.value:
            buy_cond_1 = (
                (dataframe["ema_5"] > dataframe["ema_13"])
                & (dataframe["ema_13"] > dataframe["ema_21"])
                & (dataframe["rsi_14"] < self.buy_1_rsi.value)
                & (dataframe["close"] < dataframe["bb_lowerband"])
                & (dataframe["volume"] > 0)
            )
            conditions.append(buy_cond_1)

        # -----------------------------------------------------------------
        # Buy condition 2 — BB lower band bounce with volume
        # -----------------------------------------------------------------
        if self.buy_cond_2_enabled.value:
            buy_cond_2 = (
                (dataframe["close"] < dataframe["bb_lowerband"])
                & (dataframe["rsi_14"] < self.buy_2_rsi.value)
                & (dataframe["mfi_14"] < self.buy_2_mfi.value)
                & (dataframe["volume"] > dataframe["volume"].shift(1))
            )
            conditions.append(buy_cond_2)

        # -----------------------------------------------------------------
        # Buy condition 3 — Williams %R oversold with EMA support
        # -----------------------------------------------------------------
        if self.buy_cond_3_enabled.value:
            buy_cond_3 = (
                (dataframe["willr_14"] < self.buy_3_willr.value)
                & (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["rsi_14"] < self.buy_3_rsi.value)
                & (dataframe["volume"] > 0)
            )
            conditions.append(buy_cond_3)

        # -----------------------------------------------------------------
        # Buy condition 4 — VWAP dip buy
        # -----------------------------------------------------------------
        if self.buy_cond_4_enabled.value:
            buy_cond_4 = (
                (dataframe["close"] < dataframe["vwap"] * self.buy_4_vwap_factor.value)
                & (dataframe["rsi_14"] < self.buy_4_rsi.value)
                & (dataframe[f"rsi_14_1h_1h"] > self.buy_4_rsi_1h_min.value)
                & (dataframe["volume"] > 0)
            )
            conditions.append(buy_cond_4)

        # -----------------------------------------------------------------
        # Buy condition 5 — 1h Stochastic RSI oversold
        # -----------------------------------------------------------------
        if self.buy_cond_5_enabled.value:
            buy_cond_5 = (
                (dataframe[f"stochrsi_k_1h_1h"] < self.buy_5_stochrsi_k.value)
                & (dataframe[f"stochrsi_d_1h_1h"] < self.buy_5_stochrsi_d.value)
                & (dataframe["rsi_14"] < self.buy_5_rsi.value)
                & (dataframe["close"] < dataframe["ema_50"])
                & (dataframe["volume"] > 0)
            )
            conditions.append(buy_cond_5)

        # -----------------------------------------------------------------
        # Buy condition 6 — EMA 50/200 golden zone with RSI
        # -----------------------------------------------------------------
        if self.buy_cond_6_enabled.value:
            buy_cond_6 = (
                (dataframe["close"] < dataframe["ema_50"])
                & (dataframe["ema_50"] > dataframe["ema_200"])
                & (dataframe["rsi_14"] < self.buy_6_rsi.value)
                & (dataframe[f"willr_14_1h_1h"] < self.buy_6_willr_1h.value)
                & (dataframe["volume"] > 0)
            )
            conditions.append(buy_cond_6)

        # -----------------------------------------------------------------
        # Buy condition 7 — BB squeeze with 1h RSI divergence
        # -----------------------------------------------------------------
        if self.buy_cond_7_enabled.value:
            buy_cond_7 = (
                (dataframe["bb_width"] < self.buy_7_bb_width.value)
                & (dataframe["close"] < dataframe["bb_lowerband"])
                & (dataframe[f"rsi_14_1h_1h"] > self.buy_7_rsi_1h_min.value)
                & (dataframe["rsi_14"] < self.buy_7_rsi.value)
                & (dataframe["volume"] > 0)
            )
            conditions.append(buy_cond_7)

        # -----------------------------------------------------------------
        # Buy condition 8 — Multi-timeframe RSI alignment
        # -----------------------------------------------------------------
        if self.buy_cond_8_enabled.value:
            buy_cond_8 = (
                (dataframe["rsi_14"] < self.buy_8_rsi.value)
                & (dataframe[f"rsi_14_15m_15m"] < self.buy_8_rsi_15m.value)
                & (dataframe[f"rsi_14_1h_1h"] < self.buy_8_rsi_1h.value)
                & (dataframe["close"] < dataframe["sma_200"])
                & (dataframe["close"] < dataframe["bb_middleband"])
                & (dataframe["volume"] > 0)
            )
            conditions.append(buy_cond_8)

        # OR all conditions together
        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions: list = []

        # -----------------------------------------------------------------
        # Sell condition 1 — RSI overbought
        # -----------------------------------------------------------------
        sell_cond_1 = (
            (dataframe["rsi_14"] > self.sell_1_rsi.value)
            & (dataframe["close"] > dataframe["bb_upperband"])
            & (dataframe["volume"] > 0)
        )
        conditions.append(sell_cond_1)

        # -----------------------------------------------------------------
        # Sell condition 2 — Williams %R overbought
        # -----------------------------------------------------------------
        sell_cond_2 = (
            (dataframe["willr_14"] > self.sell_2_willr.value)
            & (dataframe["rsi_14"] > self.sell_2_rsi.value)
            & (dataframe["volume"] > 0)
        )
        conditions.append(sell_cond_2)

        # -----------------------------------------------------------------
        # Sell condition 3 — EMA bearish crossover
        # -----------------------------------------------------------------
        sell_cond_3 = (
            (dataframe["ema_5"] < dataframe["ema_13"])
            & (dataframe["ema_13"] < dataframe["ema_21"])
            & (dataframe["rsi_14"] > self.sell_3_rsi.value)
            & (dataframe["volume"] > 0)
        )
        conditions.append(sell_cond_3)

        # -----------------------------------------------------------------
        # Sell condition 4 — MFI overbought with BB upper band
        # -----------------------------------------------------------------
        sell_cond_4 = (
            (dataframe["mfi_14"] > self.sell_4_mfi.value)
            & (dataframe["close"] > dataframe["bb_upperband"] * self.sell_4_bb_factor.value)
            & (dataframe["volume"] > 0)
        )
        conditions.append(sell_cond_4)

        # -----------------------------------------------------------------
        # Sell condition 5 — 1h Stochastic RSI overbought
        # -----------------------------------------------------------------
        sell_cond_5 = (
            (dataframe[f"stochrsi_k_1h_1h"] > self.sell_5_stochrsi.value)
            & (dataframe[f"stochrsi_d_1h_1h"] > self.sell_5_stochrsi.value)
            & (dataframe["rsi_14"] > self.sell_5_rsi.value)
            & (dataframe["volume"] > 0)
        )
        conditions.append(sell_cond_5)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "exit_long"] = 1

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool:
        """
        Derisking system: if open profit drops below -3% AND the 1h EMA-21 < EMA-50
        (downtrend), trigger a sell.
        """
        if current_profit > self.derisk_profit_threshold.value:
            return False

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return False

        last_candle = dataframe.iloc[-1]

        ema_21_1h = last_candle.get(f"ema_21_1h_1h", 0)
        ema_50_1h = last_candle.get(f"ema_50_1h_1h", 0)

        if ema_21_1h > 0 and ema_50_1h > 0 and ema_21_1h < ema_50_1h:
            return "derisk_1h_downtrend"

        return False

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """Exchange stoploss safety net at -0.12."""
        return -0.12
