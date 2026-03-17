"""
MeanReversionScalper — Bollinger Band + RSI + Volume Spike Mean-Reversion Scalper

Timeframe: 5m (with 1h informative for trend filter)
Description: Enters when price touches the lower Bollinger Band with oversold RSI
             and a volume spike, targeting a 0.3-0.6% snap-back toward the midline.

Entry conditions:
    - Close <= BB lower band
    - RSI < 30 (momentum exhaustion)
    - Volume > 1.5x 20-period average (capitulation spike)
    - BB width > 0.5% (enough room for profitable reversion)
    - 1h EMA50 slope > -0.3% (avoids buying into waterfall sell-offs)

Risk management:
    - Tiered ROI: 0.6% immediately down to 0.1% at 40 minutes
    - Trailing stop activating at +0.3%
    - Time-based tightening custom stoploss (-0.8% -> -0.3%)
    - Signal exit when price hits BB midline
    - StoplossGuard after 3 consecutive losses + per-pair cooldown

WARNING: This strategy has NOT been backtested. Run through at least 6 months of
data on 8+ liquid pairs, verify win rate > ~55% with profit factor > 1.3, and
paper trade 2-4 weeks before using real capital. Requires fees <= 0.075% per side.
"""

from datetime import datetime, timedelta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import pandas_ta as pta


class MeanReversionScalper(IStrategy):

    INTERFACE_VERSION: int = 3

    timeframe: str = "5m"

    # Tiered ROI — 0.6% at open, tapering to 0.1% at 40 min
    minimal_roi: dict = {
        "0": 0.006,
        "10": 0.005,
        "20": 0.003,
        "30": 0.002,
        "40": 0.001,
    }

    stoploss: float = -0.008  # -0.8% hard stoploss

    # Trailing stop — activates at +0.3%
    trailing_stop: bool = True
    trailing_stop_positive: float = 0.002
    trailing_stop_positive_offset: float = 0.003
    trailing_only_offset_is_reached: bool = True

    # Trade management
    max_open_trades: int = 6
    use_exit_signal: bool = True
    exit_profit_only: bool = False

    # Informative timeframe for trend filter
    informative_timeframe: str = "1h"

    # Process only new candles for performance
    process_only_new_candles: bool = True

    # -----------------------------------------------------------------------
    # Buy hyperopt parameters
    # -----------------------------------------------------------------------
    buy_rsi_threshold = IntParameter(20, 40, default=30, space="buy",
                                     help="RSI must be below this to enter")
    buy_volume_factor = DecimalParameter(1.2, 2.5, default=1.5, space="buy",
                                         help="Volume must exceed SMA(20) * this factor")
    buy_bb_width_min = DecimalParameter(0.003, 0.01, default=0.005, space="buy",
                                         help="Minimum BB width (fraction) for entry")
    buy_ema_slope_min = DecimalParameter(-0.005, 0.0, default=-0.003, space="buy",
                                          help="Min 1h EMA50 slope (fraction) — trend filter")

    # -----------------------------------------------------------------------
    # Sell hyperopt parameters
    # -----------------------------------------------------------------------
    sell_bb_mid_offset = DecimalParameter(0.995, 1.005, default=1.0, space="sell",
                                           help="Exit at BB mid * this offset")

    # -----------------------------------------------------------------------
    # Protection — StoplossGuard + CooldownPeriod
    # -----------------------------------------------------------------------
    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 12,  # 1 hour of 5m candles
                "trade_limit": 3,
                "stop_duration_candles": 12,    # pause for 1 hour
                "only_per_pair": False,
            },
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2,     # 10 min cooldown per pair
                "only_per_pair": True,
            },
        ]

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_timeframe) for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # -- 5m indicators --

        # RSI (14)
        dataframe["rsi"] = pta.rsi(dataframe["close"], length=14)

        # Bollinger Bands (20, 2 std)
        bbands = pta.bbands(dataframe["close"], length=20, std=2.0)
        dataframe["bb_lower"] = bbands["BBL_20_2.0"]
        dataframe["bb_mid"] = bbands["BBM_20_2.0"]
        dataframe["bb_upper"] = bbands["BBU_20_2.0"]

        # BB width as fraction of midline
        dataframe["bb_width"] = (
            (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]
        )

        # Volume SMA (20)
        dataframe["volume_sma_20"] = pta.sma(dataframe["volume"], length=20)

        # -- 1h informative: EMA50 slope as trend filter --
        if self.dp:
            inf_tf = self.dp.get_pair_dataframe(
                pair=metadata["pair"], timeframe=self.informative_timeframe
            )
            if not inf_tf.empty:
                inf_tf["ema50"] = pta.ema(inf_tf["close"], length=50)
                inf_tf["ema50_slope"] = inf_tf["ema50"].pct_change(periods=1)
                # Carry forward the latest 1h EMA slope into every 5m row
                dataframe["ema50_1h_slope"] = inf_tf["ema50_slope"].iloc[-1]
            else:
                dataframe["ema50_1h_slope"] = 0.0
        else:
            dataframe["ema50_1h_slope"] = 0.0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Price at or below lower Bollinger Band
                (dataframe["close"] <= dataframe["bb_lower"])
                # RSI confirms oversold exhaustion
                & (dataframe["rsi"] < self.buy_rsi_threshold.value)
                # Volume spike — capitulation, not low-conviction slide
                & (dataframe["volume"] > dataframe["volume_sma_20"] * self.buy_volume_factor.value)
                # Enough BB width for profitable reversion
                & (dataframe["bb_width"] > self.buy_bb_width_min.value)
                # 1h trend filter — not a waterfall sell-off
                & (dataframe["ema50_1h_slope"] > self.buy_ema_slope_min.value)
                # Sanity
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Price reverted to BB midline
                (dataframe["close"] >= dataframe["bb_mid"] * self.sell_bb_mid_offset.value)
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Time-based tightening stoploss:
          0-10 min : -0.8% (use the static stoploss)
          10-20 min: -0.5%
          20-30 min: -0.4%
          30+ min  : -0.3%
        """
        elapsed = (current_time - trade.open_date_utc).total_seconds() / 60.0

        if elapsed > 30:
            return -0.003
        elif elapsed > 20:
            return -0.004
        elif elapsed > 10:
            return -0.005

        # Return 1 to keep the default stoploss (-0.8%)
        return 1
