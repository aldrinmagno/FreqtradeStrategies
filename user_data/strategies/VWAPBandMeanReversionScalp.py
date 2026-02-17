# --- Do not remove these libs ---
from functools import reduce
from freqtrade.strategy import IStrategy
from freqtrade.strategy import BooleanParameter, IntParameter, DecimalParameter
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime
import numpy as np
# --------------------------------
import talib.abstract as ta
from technical import qtpylib


class VWAPBandMeanReversionScalp(IStrategy):
    """
    VWAP Band Mean Reversion Scalp

    Computes rolling VWAP with standard-deviation bands on the 5m timeframe.
    Enters long when price stretches into the outer lower band and shows
    stabilization (re-entry into inner band, RSI bounce, or bullish candle
    pattern). Exits when price reverts toward VWAP or reaches upper band.

    Designed for high-frequency spot scalping with tight risk management.
    Recommended: 30+ parallel trades to cover unavoidable losses.
    """

    INTERFACE_VERSION: int = 3

    timeframe = '5m'

    # --- Minimal ROI ---
    # Scalping: small targets, time-decaying
    minimal_roi = {
        "0": 0.01,
        "10": 0.007,
        "20": 0.005,
        "40": 0.003,
        "60": 0.0,
    }

    # --- Stoploss ---
    stoploss = -0.02

    # --- Trailing Stop ---
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # --- Order management ---
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC',
    }

    # --- Startup candle count ---
    # Need enough candles for the longest rolling window (VWAP lookback)
    startup_candle_count: int = 250

    # --- Protections ---
    protections = [
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 48,
            "trade_limit": 20,
            "stop_duration_candles": 12,
            "max_allowed_drawdown": 0.15,
        },
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 24,
            "trade_limit": 4,
            "stop_duration_candles": 6,
            "only_per_pair": False,
        },
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 2,
        },
    ]

    # =============================================
    # VWAP Configuration Parameters
    # =============================================
    vwap_lookback = IntParameter(100, 300, default=200, space='buy', optimize=True)
    band_outer_mult = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space='buy', optimize=True)
    band_inner_mult = DecimalParameter(0.5, 1.5, default=1.0, decimals=1, space='buy', optimize=True)

    # =============================================
    # Entry Confirmation Parameters
    # =============================================
    buy_rsi = IntParameter(10, 40, default=30, space='buy', optimize=True)
    buy_rsi_enabled = BooleanParameter(default=True, space='buy')
    buy_mfi = IntParameter(10, 35, default=20, space='buy', optimize=True)
    buy_mfi_enabled = BooleanParameter(default=True, space='buy')
    buy_adx = IntParameter(15, 40, default=25, space='buy', optimize=True)
    buy_adx_enabled = BooleanParameter(default=True, space='buy')
    buy_candle_reentry_enabled = BooleanParameter(default=True, space='buy')
    buy_bullish_candle_enabled = BooleanParameter(default=False, space='buy')

    # =============================================
    # Exit Parameters
    # =============================================
    sell_rsi = IntParameter(60, 85, default=70, space='sell', optimize=True)
    sell_rsi_enabled = BooleanParameter(default=True, space='sell')
    sell_mfi = IntParameter(70, 95, default=80, space='sell', optimize=True)
    sell_mfi_enabled = BooleanParameter(default=False, space='sell')
    sell_vwap_cross_enabled = BooleanParameter(default=True, space='sell')
    sell_upper_band_enabled = BooleanParameter(default=True, space='sell')

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # =============================================
        # 1. Rolling VWAP Computation
        # =============================================
        # Typical price = (high + low + close) / 3
        dataframe['typical_price'] = (
            dataframe['high'] + dataframe['low'] + dataframe['close']
        ) / 3.0

        # Guard against zero volume (exchange data gaps)
        safe_volume = dataframe['volume'].replace(0, np.nan)

        # Volume-weighted typical price
        dataframe['vp'] = dataframe['typical_price'] * safe_volume

        # Rolling sums for VWAP
        lookback = self.vwap_lookback.value
        min_periods = max(1, lookback // 2)

        dataframe['rolling_vp_sum'] = dataframe['vp'].rolling(
            window=lookback, min_periods=min_periods
        ).sum()
        dataframe['rolling_vol_sum'] = safe_volume.rolling(
            window=lookback, min_periods=min_periods
        ).sum()

        # VWAP = cumulative(volume * typical_price) / cumulative(volume)
        dataframe['vwap'] = (
            dataframe['rolling_vp_sum'] / dataframe['rolling_vol_sum']
        )

        # Forward-fill any NaN values from zero-volume candles
        dataframe['vwap'] = dataframe['vwap'].ffill()

        # =============================================
        # 2. Standard Deviation Bands around VWAP
        # =============================================
        dataframe['vwap_deviation'] = dataframe['typical_price'] - dataframe['vwap']

        dataframe['vwap_std'] = dataframe['vwap_deviation'].rolling(
            window=lookback, min_periods=min_periods
        ).std()

        # Outer bands (entry trigger zones)
        dataframe['vwap_upper_outer'] = (
            dataframe['vwap'] + dataframe['vwap_std'] * self.band_outer_mult.value
        )
        dataframe['vwap_lower_outer'] = (
            dataframe['vwap'] - dataframe['vwap_std'] * self.band_outer_mult.value
        )

        # Inner bands (confirmation / re-entry zones)
        dataframe['vwap_upper_inner'] = (
            dataframe['vwap'] + dataframe['vwap_std'] * self.band_inner_mult.value
        )
        dataframe['vwap_lower_inner'] = (
            dataframe['vwap'] - dataframe['vwap_std'] * self.band_inner_mult.value
        )

        # =============================================
        # 3. Supporting Indicators
        # =============================================
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)

        # EMA for trend context
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # Candle pattern detection: stabilization/reversal confirmation
        dataframe['cdl_hammer'] = ta.CDLHAMMER(dataframe)
        dataframe['cdl_invhammer'] = ta.CDLINVERTEDHAMMER(dataframe)
        dataframe['cdl_engulfing'] = ta.CDLENGULFING(dataframe)
        dataframe['cdl_dragonfly'] = ta.CDLDRAGONFLYDOJI(dataframe)

        dataframe['bullish_candle'] = (
            (dataframe['cdl_hammer'] > 0) |
            (dataframe['cdl_invhammer'] > 0) |
            (dataframe['cdl_engulfing'] > 0) |
            (dataframe['cdl_dragonfly'] > 0)
        ).astype(int)

        # =============================================
        # 4. Band Position Signals (pre-computed)
        # =============================================
        # Previous candle was below lower outer band
        dataframe['prev_below_outer'] = (
            dataframe['close'].shift(1) < dataframe['vwap_lower_outer'].shift(1)
        ).astype(int)

        # Current candle re-entered inner band from below
        dataframe['reentry_inner'] = (
            (dataframe['prev_below_outer'] == 1) &
            (dataframe['close'] > dataframe['vwap_lower_inner'])
        ).astype(int)

        # Close above upper inner band (exit zone)
        dataframe['above_upper_inner'] = (
            dataframe['close'] > dataframe['vwap_upper_inner']
        ).astype(int)

        # Close crossed above VWAP (exit trigger)
        dataframe['cross_above_vwap'] = (
            qtpylib.crossed_above(dataframe['close'], dataframe['vwap'])
        ).astype(int)

        # =============================================
        # 5. Volatility / Spread Filter
        # =============================================
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        # =============================================
        # Core VWAP Mean Reversion Signal (always active)
        # Previous candle stretched below the outer lower band
        # =============================================
        conditions.append(dataframe['prev_below_outer'] == 1)

        # =============================================
        # Confirmation Signals (hyperopt-toggleable)
        # =============================================

        # Candle re-entry into inner band (stabilization)
        if self.buy_candle_reentry_enabled.value:
            conditions.append(dataframe['reentry_inner'] == 1)

        # RSI oversold bounce
        if self.buy_rsi_enabled.value:
            conditions.append(dataframe['rsi'] < self.buy_rsi.value)

        # MFI oversold (volume confirms selling exhaustion)
        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] < self.buy_mfi.value)

        # ADX below threshold (ranging market = good for mean reversion)
        if self.buy_adx_enabled.value:
            conditions.append(dataframe['adx'] < self.buy_adx.value)

        # Bullish candlestick pattern
        if self.buy_bullish_candle_enabled.value:
            conditions.append(dataframe['bullish_candle'] == 1)

        # =============================================
        # Static Filters (always active)
        # =============================================

        # Trend guard: do not buy in sustained downtrends
        conditions.append(dataframe['close'] > dataframe['ema_200'])

        # Volatility filter: avoid dead markets and flash crashes
        conditions.append(dataframe['atr_pct'] > 0.05)
        conditions.append(dataframe['atr_pct'] < 3.0)

        # VWAP must be valid
        conditions.append(dataframe['vwap'].notna())

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        # =============================================
        # Core VWAP Mean Reversion Exit
        # Exit triggers use OR logic (any target hit)
        # =============================================
        exit_signals = []

        # Price crossed above VWAP (mean reversion target reached)
        if self.sell_vwap_cross_enabled.value:
            exit_signals.append(dataframe['cross_above_vwap'] == 1)

        # Price reached upper inner band
        if self.sell_upper_band_enabled.value:
            exit_signals.append(dataframe['above_upper_inner'] == 1)

        # Combine exit signals with OR
        if exit_signals:
            combined_exit = reduce(lambda x, y: x | y, exit_signals)
            conditions.append(combined_exit)
        else:
            # Fallback: use VWAP cross if no exit signal enabled
            conditions.append(dataframe['cross_above_vwap'] == 1)

        # =============================================
        # Additional Exit Confirmations (hyperopt-toggleable)
        # =============================================

        # RSI overbought confirmation
        if self.sell_rsi_enabled.value:
            conditions.append(dataframe['rsi'] > self.sell_rsi.value)

        # MFI overbought confirmation
        if self.sell_mfi_enabled.value:
            conditions.append(dataframe['mfi'] > self.sell_mfi.value)

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade,
                        current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool,
                        **kwargs) -> float:
        """
        Dynamic stoploss that tightens over time.

        - First 15 minutes: use the configured stoploss (-2%)
        - After 15 minutes: tighten to -1.5%
        - After 30 minutes: tighten to -1.0%
        - After 60 minutes: tighten to -0.5% (force exit near break-even)

        Scalps that haven't moved in our favor within an hour
        are likely failed setups.
        """
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60.0

        if trade_duration < 15:
            return -0.02
        elif trade_duration < 30:
            return -0.015
        elif trade_duration < 60:
            return -0.01
        else:
            return -0.005

    plot_config = {
        "main_plot": {
            "vwap": {"color": "orange"},
            "vwap_upper_outer": {"color": "red"},
            "vwap_lower_outer": {"color": "red"},
            "vwap_upper_inner": {"color": "rgba(255,0,0,0.3)"},
            "vwap_lower_inner": {"color": "rgba(255,0,0,0.3)"},
            "ema_50": {"color": "blue"},
            "ema_200": {"color": "purple"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "blue"},
            },
            "MFI": {
                "mfi": {"color": "green"},
            },
            "ADX": {
                "adx": {"color": "red"},
            },
            "ATR%": {
                "atr_pct": {"color": "gray"},
            },
        },
    }
