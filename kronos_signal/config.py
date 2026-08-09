"""Daily BTC Kronos → long/hold/short defaults."""

# Market data: Binance public klines (NOT Yahoo).
# Primary: https://data-api.binance.vision/api/v3/klines
# Fallback: api.binance.us, api.binance.com
# Interval is true daily UTC candles (open time 00:00 UTC).
DATA_PROVIDER = "binance"
SYMBOL = "BTCUSDT"
INTERVAL = "1d"

# Kronos-base max context is 512; stay under that.
LOOKBACK = 400
PRED_LEN = 5  # signal horizon in daily bars

MODEL_ID = "NeoQuasar/Kronos-base"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
MAX_CONTEXT = 512

# Monte Carlo
N_PATHS = 30
TEMPERATURE = 1.0
TOP_P = 0.9

# Decision thresholds on expected return over PRED_LEN days
P_UP_LONG = 0.60
P_UP_SHORT = 0.40
TAU = 0.005  # 0.5% mean move required to leave HOLD
