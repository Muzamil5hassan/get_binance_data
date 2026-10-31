import requests
import pandas as pd
import numpy as np
import os
from scipy.stats import linregress

# ============================================================
# SETTINGS
# ============================================================

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"

# SYMBOL = "BTCUSDT"
SYMBOL = "XAUUSDT"
INTERVAL = "5m"

LIMIT = 999


# ============================================================
# BINANCE DATA
# ============================================================


def get_binance_data(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT):

    params = {"symbol": symbol, "interval": interval, "limit": limit}

    response = requests.get(BINANCE_URL, params=params, timeout=15)

    response.raise_for_status()

    data = response.json()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(data, columns=columns)

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")

    return df


# ============================================================
# OHLC FEATURES
# ============================================================


def basic_features_OHLC(df):

    df["direction"] = (df["close"] > df["open"]).astype(int)

    df["candle_range"] = df["high"] - df["low"]

    df["body_change"] = df["close"] - df["open"]

    df["body_abs"] = df["body_change"].abs()

    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)

    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / (avg_loss + 1e-9)

    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --------------------------------------------------------
    # Wick
    # --------------------------------------------------------

    df["wick_imbalance"] = (df["upper_wick"] - df["lower_wick"]) / (
        df["upper_wick"] + df["lower_wick"] + 1e-9
    )

    df["wick_size_pct"] = (df["upper_wick"] + df["lower_wick"]) / (
        df["high"] - df["low"] + 1e-9
    )

    # --------------------------------------------------------
    # Normalized
    # --------------------------------------------------------

    df["body_pct"] = df["body_change"] / (df["open"] + 1e-9)

    df["range_pct"] = df["candle_range"] / (df["open"] + 1e-9)

    # --------------------------------------------------------
    # Support / Resistance
    # Previous candles only
    # --------------------------------------------------------

    previous_high = df["high"].shift(1)

    previous_low = df["low"].shift(1)

    resistance_24 = previous_high.rolling(24).max()

    support_24 = previous_low.rolling(24).min()

    df["distance_from_support"] = (df["close"] - support_24) / (df["close"] + 1e-9)

    df["distance_from_resistance"] = (resistance_24 - df["close"]) / (
        df["close"] + 1e-9
    )

    # --------------------------------------------------------
    # 14 candle trend
    # --------------------------------------------------------

    previous_direction = df["direction"].shift(1)

    df["bullish_ratio_14"] = previous_direction.rolling(14).mean()

    df["avg_body_14"] = df["body_change"].shift(1).rolling(14).mean()

    # --------------------------------------------------------
    # Directional pressure
    # --------------------------------------------------------

    body_sum_14 = df["body_change"].shift(1).rolling(14).sum()

    body_abs_sum_14 = df["body_abs"].shift(1).rolling(14).sum()

    df["directional_pressure_14"] = body_sum_14 / (body_abs_sum_14 + 1e-9)

    return df


# ============================================================
# TRADING FEATURES
# ============================================================


def basic_features_trading(df):

    df["avg_trade_size_quote"] = df["quote_volume"] / (df["trades"] + 1e-9)

    df["avg_trade_size_base"] = df["volume"] / (df["trades"] + 1e-9)

    df["trades_ratio_5"] = df["trades"] / (
        df["trades"].shift(1).rolling(5).mean() + 1e-9
    )

    df["taker_buy_ratio"] = df["taker_buy_base"] / (df["volume"] + 1e-9)

    df["taker_buy_quote_ratio"] = df["taker_buy_quote"] / (df["quote_volume"] + 1e-9)

    return df


# ============================================================
# DIFFERENT TIMEFRAME FEATURES
# ============================================================


def create_in_differ_frames(df):

    previous_direction = df["direction"].shift(1)

    # --------------------------------------------------------
    # 7 candle
    # --------------------------------------------------------

    df["bullish_ratio_7"] = previous_direction.rolling(7).mean()

    df["avg_body_7"] = df["body_change"].shift(1).rolling(7).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    # RSI 7

    avg_gain_7 = gain.rolling(7).mean()

    avg_loss_7 = loss.rolling(7).mean()

    rs_7 = avg_gain_7 / (avg_loss_7 + 1e-9)

    df["rsi_7"] = 100 - (100 / (1 + rs_7))

    # RSI 3

    avg_gain_3 = gain.rolling(3).mean()

    avg_loss_3 = loss.rolling(3).mean()

    rs_3 = avg_gain_3 / (avg_loss_3 + 1e-9)

    df["rsi_3"] = 100 - (100 / (1 + rs_3))

    # --------------------------------------------------------
    # Directional pressure 7
    # --------------------------------------------------------

    body_sum_7 = df["body_change"].shift(1).rolling(7).sum()

    body_abs_sum_7 = df["body_abs"].shift(1).rolling(7).sum()

    df["directional_pressure_7"] = body_sum_7 / (body_abs_sum_7 + 1e-9)

    # --------------------------------------------------------
    # Buy pressure
    # --------------------------------------------------------

    df["taker_buy_ratio_change"] = df["taker_buy_ratio"].diff()

    df["taker_buy_quote_ratio_change"] = df["taker_buy_quote_ratio"].diff()

    df["current_buy_pressure"] = 2 * df["taker_buy_ratio"] - 1

    df["change_in_buy_pressure"] = df["current_buy_pressure"].diff()

    return df


# ============================================================
# VOLUME FEATURES
# ============================================================


def make_volume_useable(df):

    df["trades_change"] = df["trades"].diff()

    df["trades_pct_change"] = df["trades"].pct_change()

    df["buy_pressure_std_10"] = df["current_buy_pressure"].rolling(10).std()

    return df


# ============================================================
# ATR
# ============================================================


def add_atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (df["high"] - previous_close).abs()

    tr3 = (df["low"] - previous_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr"] = true_range.rolling(period).mean()

    df["atr_pct"] = df["atr"] / (df["close"] + 1e-9)

    return df


# ============================================================
# EMA
# ============================================================


def add_ema_features(df):

    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()

    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    df["ema_9_distance"] = (df["close"] - df["ema_9"]) / (df["close"] + 1e-9)

    df["ema_21_distance"] = (df["close"] - df["ema_21"]) / (df["close"] + 1e-9)

    df["ema_50_distance"] = (df["close"] - df["ema_50"]) / (df["close"] + 1e-9)

    return df


# ============================================================
# ROLLING SLOPE + R2
# ============================================================


def rolling_slope_r2(series, window=5):

    slopes = np.full(len(series), np.nan)

    r2s = np.full(len(series), np.nan)

    x = np.arange(window)

    for i in range(window - 1, len(series)):

        y = series.iloc[i - window + 1 : i + 1].values

        try:

            result = linregress(x, y)

            slopes[i] = result.slope

            r2s[i] = result.rvalue**2

        except Exception:

            pass

    return slopes, r2s


# ============================================================
# CLEAN DATA
# ============================================================


def drop_useless_col_rows(df):

    df = df.drop(columns=["ignore", "close_time"], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    return df


# ============================================================
# CREATE ALL FEATURES
# ============================================================


def get_all_features_ready(df):

    df = df.copy()

    df = basic_features_OHLC(df)

    df = basic_features_trading(df)

    df = create_in_differ_frames(df)

    df["close_slope_3"], df["close_r2_3"] = rolling_slope_r2(df["close"], 3)

    df["volume_slope_3"], df["volume_r2_3"] = rolling_slope_r2(df["volume"], 3)

    df["rsi3_slope_3"], df["rsi3_r2_3"] = rolling_slope_r2(df["rsi_3"], 3)

    df["close_slope_5"], df["close_r2_5"] = rolling_slope_r2(df["close"], 5)

    df = make_volume_useable(df)

    df = add_atr(df)

    df = add_ema_features(df)

    df = drop_useless_col_rows(df)

    return df


# ============================================================
# PUBLIC FUNCTION
# ============================================================


def get_market_data(SYMBOL, INTERVAL):

    try:

        df = get_binance_data(symbol=SYMBOL, interval=INTERVAL)

        # Last Binance candle can still be forming.
        # Remove it so prediction uses completed candles only.
        if len(df) > 2:

            df = df.iloc[:-1].copy()

        df = get_all_features_ready(df)

        if len(df) < 100:

            print("[DATA] Not enough candles.")

            return None

        return df

    except Exception as e:

        print("[DATA ERROR]", type(e).__name__, e)

        return None


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    df = get_market_data(
        SYMBOL="BTCUSDT",
        INTERVAL="5m"
    )

    if df is not None:

        os.makedirs("data", exist_ok=True)

        output_file = "data/BTCUSDT_5m.csv"

        df.to_csv(output_file, index=False)

        print("BTC DATA SAVED SUCCESSFULLY")
        print("Rows:", len(df))
        print("Latest:", df["timestamp"].iloc[-1])
        print("BTC:", df["close"].iloc[-1])
        print("ATR:", df["atr"].iloc[-1])
        print("Columns:", len(df.columns))

    else:

        print("No data generated.")
        raise SystemExit(1)
