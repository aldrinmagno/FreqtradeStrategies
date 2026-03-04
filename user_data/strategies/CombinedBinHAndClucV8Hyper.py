"""
CombinedBinHAndClucV8Hyper — Hyperopt-Ready Combined Dip-Buy Scalper

Source: https://github.com/freqtrade/freqtrade-strategies (BinHV45 + ClucMay72018)
Timeframe: 5m (with 1h informative)
Description: Combines BinHV45 and ClucMay72018 dip-buy signals with full hyperopt
             parameter support. Four independently togglable entry conditions with
             SSL Channel and MFI indicators.
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


class CombinedBinHAndClucV8Hyper(IStrategy):

    INTERFACE_VERSION: int = 3

    timeframe: str = "5m"
    informative_timeframe: str = "1h"

    minimal_roi: dict = {
        "0": 0.15,
        "30": 0.05,
        "60": 0.02,
        "120": 0.01,
    }

    stoploss: float = -0.274

    use_exit_signal: bool = True
    exit_profit_only: bool = False

    # -----------------------------------------------------------------------
    # Buy hyperopt parameters — Condition 1 (BB40 squeeze)
    # -----------------------------------------------------------------------
    buy_cond1_enabled = BooleanParameter(default=True, space="buy")
    buy_cond1_bbdelta = DecimalParameter(0.01, 0.05, default=0.025, space="buy")
    buy_cond1_closedelta_factor = DecimalParameter(0.5, 1.5, default=1.0, space="buy")
    buy_cond1_tail_factor = DecimalParameter(0.1, 1.0, default=0.5, space="buy")

    # -----------------------------------------------------------------------
    # Buy hyperopt parameters — Condition 2 (BB20 lower band dip)
    # -----------------------------------------------------------------------
    buy_cond2_enabled = BooleanParameter(default=True, space="buy")
    buy_cond2_close_factor = DecimalParameter(0.97, 1.0, default=0.99, space="buy")
    buy_cond2_volume_factor = DecimalParameter(0.5, 2.0, default=1.0, space="buy")

    # -----------------------------------------------------------------------
    # Buy hyperopt parameters — Condition 3 (RSI 1h divergence)
    # -----------------------------------------------------------------------
    buy_cond3_enabled = BooleanParameter(default=True, space="buy")
    buy_cond3_rsi_1h = IntParameter(30, 50, default=40, space="buy")
    buy_cond3_rsi_diff = DecimalParameter(0.5, 3.0, default=1.5, space="buy")

    # -----------------------------------------------------------------------
    # Buy hyperopt parameters — Condition 4 (EMA momentum)
    # -----------------------------------------------------------------------
    buy_cond4_enabled = BooleanParameter(default=True, space="buy")
    buy_cond4_ema_ratio = DecimalParameter(0.99, 1.01, default=1.0, space="buy")

    # -----------------------------------------------------------------------
    # Sell hyperopt parameters
    # -----------------------------------------------------------------------
    sell_rsi_threshold = IntParameter(60, 80, default=70, space="sell")

    def informative_pairs(self) -> list:
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_timeframe) for pair in pairs]

    def _ssl_channel(self, dataframe: DataFrame, length: int = 10) -> DataFrame:
        """SSL Channel indicator — ATR + SMA hybrid."""
        atr = pta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=length)
        sma_high = pta.sma(dataframe["high"], length=length)
        sma_low = pta.sma(dataframe["low"], length=length)

        hlv = np.where(
            dataframe["close"] > sma_high,
            1,
            np.where(dataframe["close"] < sma_low, -1, np.nan),
        )
        hlv = DataFrame({"hlv": hlv}).ffill().values.flatten()

        dataframe["ssl_down"] = np.where(hlv < 0, sma_high, sma_low)
        dataframe["ssl_up"] = np.where(hlv < 0, sma_low, sma_high)
        dataframe["ssl_atr"] = atr
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- 1h informative ---
        inf_tf = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe)
        inf_tf["rsi_1h"] = pta.rsi(inf_tf["close"], length=14)
        inf_tf["sma200_1h"] = pta.sma(inf_tf["close"], length=200)
        dataframe = merge_informative_pair(dataframe, inf_tf, self.timeframe, self.informative_timeframe, ffill=True)

        # --- Bollinger Bands (20, 2) ---
        bb20 = pta.bbands(dataframe["close"], length=20, std=2.0)
        dataframe["bb20_low"] = bb20["BBL_20_2.0"]
        dataframe["bb20_mid"] = bb20["BBM_20_2.0"]
        dataframe["bb20_upp"] = bb20["BBU_20_2.0"]

        # --- Bollinger Bands (40, 2) for condition 1 ---
        bb40 = pta.bbands(dataframe["close"], length=40, std=2.0)
        dataframe["bb40_low"] = bb40["BBL_40_2.0"]
        dataframe["bb40_mid"] = bb40["BBM_40_2.0"]

        # --- BB40 deltas for condition 1 ---
        dataframe["bb40_delta"] = (dataframe["bb40_mid"] - dataframe["bb40_low"]).abs()
        dataframe["closedelta"] = (dataframe["close"] - dataframe["close"].shift(1)).abs()
        dataframe["tail"] = (dataframe["low"] - dataframe["close"]).abs()

        # --- RSI ---
        dataframe["rsi"] = pta.rsi(dataframe["close"], length=14)

        # --- EMAs ---
        dataframe["ema50"] = pta.ema(dataframe["close"], length=50)
        dataframe["ema200"] = pta.ema(dataframe["close"], length=200)
        dataframe["sma200"] = pta.sma(dataframe["close"], length=200)

        # --- MFI (Money Flow Index, 14) ---
        dataframe["mfi"] = pta.mfi(
            dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"], length=14
        )

        # --- SSL Channel ---
        dataframe = self._ssl_channel(dataframe, length=10)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions: list = []

        # Condition 1 — BB40 squeeze
        if self.buy_cond1_enabled.value:
            cond1 = (
                (dataframe["bb40_delta"] > self.buy_cond1_bbdelta.value)
                & (dataframe["closedelta"] > dataframe["bb40_delta"] * self.buy_cond1_closedelta_factor.value)
                & (dataframe["tail"] < dataframe["bb40_delta"] * self.buy_cond1_tail_factor.value)
                & (dataframe["close"] < dataframe["bb40_low"])
                & (dataframe["volume"] > 0)
            )
            conditions.append(cond1)

        # Condition 2 — BB20 lower band dip
        if self.buy_cond2_enabled.value:
            cond2 = (
                (dataframe["close"] < dataframe["bb20_low"])
                & (dataframe["close"] < dataframe["close"].shift(1) * self.buy_cond2_close_factor.value)
                & (dataframe["volume"] > dataframe["volume"].shift(1) * self.buy_cond2_volume_factor.value)
            )
            conditions.append(cond2)

        # Condition 3 — RSI 1h divergence
        if self.buy_cond3_enabled.value:
            cond3 = (
                (dataframe[f"rsi_1h_{self.informative_timeframe}"] < self.buy_cond3_rsi_1h.value)
                & (dataframe["rsi"] < dataframe["rsi"].shift(1))
                & (dataframe["rsi"] > dataframe["rsi"].shift(1) + self.buy_cond3_rsi_diff.value)
                & (dataframe["volume"] > 0)
            )
            conditions.append(cond3)

        # Condition 4 — EMA momentum
        if self.buy_cond4_enabled.value:
            cond4 = (
                (dataframe["close"] < dataframe["ema50"])
                & (dataframe["ema50"] < dataframe["ema200"] * self.buy_cond4_ema_ratio.value)
                & (dataframe["volume"] > 0)
            )
            conditions.append(cond4)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["rsi"] > self.sell_rsi_threshold.value)
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
        """Check SMA200 direction — tighten stop if 1h trend is bearish."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return self.stoploss

        last_candle = dataframe.iloc[-1]

        # If price is above SMA200 on 1h, use wider stop
        if last_candle.get(f"sma200_1h_{self.informative_timeframe}", 0) > 0:
            if last_candle["close"] > last_candle[f"sma200_1h_{self.informative_timeframe}"]:
                return -0.274
            else:
                return -0.15

        return self.stoploss
