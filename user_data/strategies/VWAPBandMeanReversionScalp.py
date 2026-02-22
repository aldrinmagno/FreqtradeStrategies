# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from freqtrade.strategy import BooleanParameter, DecimalParameter, IntParameter
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

    Computes rolling VWAP with volume-weighted standard-deviation bands on
    the 5m timeframe. Enters long when the previous candle stretched below
    the outer lower band and at least one confirmation signal fires
    (re-entry into inner band, RSI oversold, MFI oversold, low ADX, or
    bullish candle pattern — combined with OR logic). Exits when any exit
    target is reached (VWAP cross, upper band, RSI overbought, or MFI
    overbought — also OR logic).

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
        "60": 0.001,
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
    VWAP_LOOKBACK = 200  # Fixed; not hyperopt-optimizable (recomputing per epoch is too expensive)
    band_outer_mult = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space='buy', optimize=True)
    band_inner_mult = DecimalParameter(0.5, 1.4, default=1.0, decimals=1, space='buy', optimize=True)

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
    buy_ema_guard_enabled = BooleanParameter(default=True, space='buy')

    # =============================================
    # Exit Parameters
    # =============================================
    sell_rsi = IntParameter(60, 85, default=70, space='sell', optimize=True)
    sell_rsi_enabled = BooleanParameter(default=True, space='sell')
    sell_mfi = IntParameter(70, 95, default=80, space='sell', optimize=True)
    sell_mfi_enabled = BooleanParameter(default=False, space='sell')
    sell_vwap_cross_enabled = BooleanParameter(default=True, space='sell')
    sell_upper_band_enabled = BooleanParameter(default=True, space='sell')

    def _get_band_multipliers(self):
        """Return (outer_mult, clamped_inner_mult) for band computation."""
        outer = self.band_outer_mult.value
        inner = min(self.band_inner_mult.value, outer - 0.1)
        return outer, inner

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
        lookback = self.VWAP_LOOKBACK
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
        # 2. Volume-Weighted Standard Deviation Bands around VWAP
        # =============================================
        # Volume-weighted variance: sum(vol * (tp - vwap)^2) / sum(vol)
        vw_sq_deviation = safe_volume * (
            dataframe['typical_price'] - dataframe['vwap']
        ) ** 2

        rolling_vw_sq_dev = vw_sq_deviation.rolling(
            window=lookback, min_periods=min_periods
        ).sum()
        rolling_vol_for_std = safe_volume.rolling(
            window=lookback, min_periods=min_periods
        ).sum()

        dataframe['vwap_std'] = np.sqrt(rolling_vw_sq_dev / rolling_vol_for_std)

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
        # 4. VWAP Cross Signal
        # =============================================
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

        # Compute bands on-the-fly so hyperopt can vary the multipliers
        outer_mult, inner_mult = self._get_band_multipliers()
        lower_outer = dataframe['vwap'] - dataframe['vwap_std'] * outer_mult
        lower_inner = dataframe['vwap'] - dataframe['vwap_std'] * inner_mult

        # Derived signals
        prev_below_outer = dataframe['close'].shift(1) < lower_outer.shift(1)
        reentry_inner = prev_below_outer & (dataframe['close'] > lower_inner)

        # =============================================
        # Confirmation Signals (OR logic, vectorized for hyperopt)
        #
        # Each signal is ANDed with its boolean toggle so that:
        #   enabled=True  → the signal is active
        #   enabled=False → evaluates to all-False (does not contribute)
        #
        # At least one enabled confirmation must fire for entry.
        # If ALL toggles are off, confirmation is all-False → no entries.
        # =============================================
        confirmation = (
            (reentry_inner & self.buy_candle_reentry_enabled.value)
            | ((dataframe['rsi'] < self.buy_rsi.value) & self.buy_rsi_enabled.value)
            | ((dataframe['mfi'] < self.buy_mfi.value) & self.buy_mfi_enabled.value)
            | ((dataframe['adx'] < self.buy_adx.value) & self.buy_adx_enabled.value)
            | ((dataframe['bullish_candle'] == 1) & self.buy_bullish_candle_enabled.value)
        )

        # =============================================
        # AND-logic guard (vectorized for hyperopt)
        #
        # Uses (signal | (not enabled)) so that:
        #   enabled=True  → signal must be True
        #   enabled=False → always True (transparent / no-op)
        # =============================================
        ema_guard = (
            (dataframe['close'] > dataframe['ema_200']) | (not self.buy_ema_guard_enabled.value)
        )

        # =============================================
        # Combine all conditions with AND
        # =============================================
        dataframe.loc[
            prev_below_outer                      # Core: previous candle below outer band
            & confirmation                        # At least one confirmation must fire
            & (dataframe['close'] > lower_outer)  # Bounced back above outer band
            & ema_guard                           # Trend guard (toggleable)
            & (dataframe['atr_pct'] > 0.05)      # Minimum volatility
            & (dataframe['atr_pct'] < 3.0)       # Maximum volatility
            & dataframe['vwap'].notna()           # VWAP must be valid
            & (dataframe['volume'] > 0),          # Non-zero volume
            'enter_long'] = 1

        # Persist bands in dataframe for plotting
        dataframe['vwap_lower_outer'] = lower_outer
        dataframe['vwap_upper_outer'] = dataframe['vwap'] + dataframe['vwap_std'] * outer_mult
        dataframe['vwap_lower_inner'] = lower_inner
        dataframe['vwap_upper_inner'] = dataframe['vwap'] + dataframe['vwap_std'] * inner_mult

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Compute upper inner band on-the-fly so hyperopt can vary the multiplier
        _, inner_mult = self._get_band_multipliers()
        upper_inner = dataframe['vwap'] + dataframe['vwap_std'] * inner_mult

        # =============================================
        # Exit signals (OR logic, vectorized for hyperopt)
        #
        # Each signal is ANDed with its boolean toggle so that:
        #   enabled=True  → the signal is active
        #   enabled=False → evaluates to all-False (does not contribute)
        #
        # When all toggles are off, no signal-based exit fires and the
        # trade exits only via ROI / stoploss / trailing stop.
        # =============================================
        combined_exit = (
            ((dataframe['cross_above_vwap'] == 1) & self.sell_vwap_cross_enabled.value)
            | ((dataframe['close'] > upper_inner) & self.sell_upper_band_enabled.value)
            | ((dataframe['rsi'] > self.sell_rsi.value) & self.sell_rsi_enabled.value)
            | ((dataframe['mfi'] > self.sell_mfi.value) & self.sell_mfi_enabled.value)
        )

        # Apply exit where any signal fires and volume is present
        dataframe.loc[
            combined_exit & (dataframe['volume'] > 0),
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
