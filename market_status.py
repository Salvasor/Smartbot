# -*- coding: utf-8 -*-
"""
Módulo 2 — Lectura del Mundo (mercado) · Nivel 1 → 6 · v2
- world_summary por símbolo (listo para M3)
- frescura/TTL por capa, telemetría y errores no bloqueantes
- toggles por bucket y orderbook
- paredes cercanas (wall_near_flag)
- helpers rápidos para M3 (last_price, atr_pct_latest, imb_zone1)
"""

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — Imports, configuración y utilidades
# ──────────────────────────────────────────────────────────────────────────────

import os
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime, timezone

# Config por entorno
# Nota: este módulo es solo lectura de datos públicos (tickers, OHLCV, orderbook).
# Las credenciales de API se usan exclusivamente en el módulo de ejecución de órdenes.
EXCHANGE_ID: str = os.getenv("EXCHANGE_ID", "binance")
SYMBOLS: List[str] = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT").split(",") if s.strip()]
TIMEFRAMES: List[str] = [t.strip() for t in os.getenv("TIMEFRAMES", "1m,5m,15m,1h,4h,1d").split(",") if t.strip()]
ORDERBOOK_LEVELS: int = int(os.getenv("ORDERBOOK_LEVELS", "20"))
ORDERBOOK_SAMPLE_EVERY: int = int(os.getenv("ORDERBOOK_SAMPLE_EVERY", "3"))
FEATURES_ORDERBOOK_ENABLED: bool = os.getenv("FEATURES_ORDERBOOK_ENABLED", "1") == "1"
FEATURES_LEVEL6_ENABLED: bool = os.getenv("FEATURES_LEVEL6_ENABLED", "1") == "1"

IND_BASE_TF: str = os.getenv("IND_BASE_TF", "5m")
VOL_ANOM_WINDOW: int = int(os.getenv("VOL_ANOM_WINDOW", "50"))
CORR_WINDOW: int = int(os.getenv("CORR_WINDOW", "24"))

TF_LIMITS: Dict[str, int] = {"1m":500, "5m":500, "15m":500, "1h":500, "4h":500, "1d":500}

def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _tf_to_ms(tf: str) -> int:
    table = {"1m":60000, "3m":180000, "5m":300000, "15m":900000,
             "30m":1800000, "1h":3600000, "2h":7200000, "4h":14400000, "1d":86400000}
    if tf not in table:
        raise ValueError(f"Timeframe no soportado: {tf}")
    return table[tf]

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 2 — Cliente ccxt y validación de símbolos
# ──────────────────────────────────────────────────────────────────────────────

def build_exchange() -> ccxt.Exchange:
    """Conexión pública (sin credenciales). Este módulo solo lee datos de mercado."""
    ex_class = getattr(ccxt, EXCHANGE_ID)
    ex: ccxt.Exchange = ex_class({"enableRateLimit": True})

    merged_options = dict(getattr(ex, "options", {}))
    merged_options.setdefault("adjustForTimeDifference", True)
    merged_options["defaultType"] = "spot"
    ex.options = merged_options

    ex.load_markets()
    missing = [s for s in SYMBOLS if s not in ex.markets]
    if missing:
        raise ValueError(f"Símbolos no disponibles en {EXCHANGE_ID}: {missing}")
    return ex

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 3 — Nivel 1: Tickers
# ──────────────────────────────────────────────────────────────────────────────

def read_market_ticks(ex: ccxt.Exchange, symbols: List[str]) -> Tuple[pd.DataFrame, int]:
    t0 = ex.milliseconds()
    ticks = ex.fetch_tickers(symbols)
    t1 = ex.milliseconds()
    rows = []
    for sym in symbols:
        t = ticks.get(sym, {})
        last = t.get("last"); bid = t.get("bid"); ask = t.get("ask")
        spr_abs = spr_pct = None
        if bid is not None and ask is not None and bid > 0:
            spr_abs = float(ask) - float(bid)
            spr_pct = (spr_abs / float(bid)) * 100.0
        rows.append({
            "symbol": sym, "ts": _utc_iso(),
            "last": last, "bid": bid, "ask": ask,
            "spread_abs": spr_abs, "spread_pct": spr_pct,
            "ex_latency_ms": int(t1 - t0),
        })
    return pd.DataFrame(rows), int(t1 - t0)

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — Nivel 2: OHLCV incremental y cache
# ──────────────────────────────────────────────────────────────────────────────

def _merge_ohlcv(df_old: Optional[pd.DataFrame], df_new: pd.DataFrame) -> pd.DataFrame:
    if df_old is None or df_old.empty:
        return df_new
    merged = pd.concat([df_old, df_new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return merged

def fetch_ohlcv_incremental(ex: ccxt.Exchange, symbol: str, timeframe: str,
                            buffer: Optional[pd.DataFrame], limit: int) -> Tuple[pd.DataFrame, int]:
    t0 = ex.milliseconds()
    tf_ms = _tf_to_ms(timeframe)
    since = None
    if buffer is not None and not buffer.empty:
        last_ts = int(buffer["timestamp"].iloc[-1])
        since = last_ts - tf_ms
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    t1 = ex.milliseconds()
    if not raw:
        if buffer is None:
            cols = ["timestamp","ts","open","high","low","close","volume"]
            return pd.DataFrame(columns=cols), int(t1 - t0)
        return buffer, int(t1 - t0)
    df_new = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df_new["timestamp"] = df_new["timestamp"].astype("int64")
    df_new["ts"] = pd.to_datetime(df_new["timestamp"], unit="ms", utc=True)
    return _merge_ohlcv(buffer, df_new), int(t1 - t0)

def read_all_ohlcv(ex: ccxt.Exchange, symbols: List[str], timeframes: List[str],
                   ohlcv_cache: Optional[Dict[Tuple[str,str], pd.DataFrame]] = None,
                   limits: Dict[str,int] = TF_LIMITS) -> Tuple[Dict[Tuple[str,str], pd.DataFrame], int]:
    if ohlcv_cache is None:
        ohlcv_cache = {}
    latencies = []
    for sym in symbols:
        for tf in timeframes:
            key = (sym, tf)
            old = ohlcv_cache.get(key)
            updated, l_ms = fetch_ohlcv_incremental(ex, sym, tf, old, limits.get(tf, 500))
            latencies.append(l_ms)
            if not updated.empty:
                max_keep = limits.get(tf, 500)
                updated = updated.iloc[-max_keep:].reset_index(drop=True)
            ohlcv_cache[key] = updated
    avg_lat = int(np.mean(latencies)) if latencies else 0
    return ohlcv_cache, avg_lat

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 5 — Nivel 3: Indicadores y regímenes
# ──────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0); down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def bollinger(series: pd.Series, n: int = 20, k: float = 2.0):
    ma = series.rolling(n, min_periods=n).mean()
    sd = series.rolling(n, min_periods=n).std(ddof=0)
    upper = ma + k * sd; lower = ma - k * sd
    return lower, ma, upper

def macd(series: pd.Series, fast: int=12, slow: int=26, signal: int=9):
    ema_fast = ema(series, fast); ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def stoch(df: pd.DataFrame, n: int=14, d: int=3):
    low_min = df["low"].rolling(n, min_periods=n).min()
    high_max = df["high"].rolling(n, min_periods=n).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    dline = k.rolling(d, min_periods=d).mean()
    return k, dline

def indicators_from_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    if df.empty: return df
    df["range"] = df["high"] - df["low"]
    df["body"]  = (df["close"] - df["open"]).abs()
    df["upper_wick"] = df["high"] - df[["close","open"]].max(axis=1)
    df["lower_wick"] = df[["close","open"]].min(axis=1) - df["low"]
    df["upper_wick_pct"] = (df["upper_wick"] / df["range"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    df["lower_wick_pct"] = (df["lower_wick"] / df["range"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    for n in (5,12,20,50,200):
        df[f"ema_{n}"] = ema(df["close"], n)
    df["rsi_14"] = rsi(df["close"], 14)
    df["atr"] = atr(df, 14)
    df["atr_pct"] = (df["atr"] / df["close"]) * 100.0
    bb_low, bb_mid, bb_high = bollinger(df["close"], 20, 2.0)
    df["bb_low"], df["bb_mid"], df["bb_high"] = bb_low, bb_mid, bb_high
    df["bb_width"] = ((bb_high - bb_low) / bb_mid) * 100.0
    macd_line, signal_line, hist = macd(df["close"], 12, 26, 9)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd_line, signal_line, hist
    k, dline = stoch(df, 14, 3)
    df["stoch_k"], df["stoch_d"] = k, dline
    def _trend_state(row):
        up = (row["ema_5"] > row["ema_12"] > row["ema_20"])
        down = (row["ema_5"] < row["ema_12"] < row["ema_20"])
        if up: return "up"
        if down: return "down"
        return "flat-weak"
    df["trend_state"] = df.apply(_trend_state, axis=1)
    q1, q2, q3 = df["atr_pct"].quantile([0.25, 0.50, 0.75])
    def _vol_bucket(v):
        if pd.isna(v): return None
        if v <= q1: return "calm"
        if v <= q2: return "normal"
        if v <= q3: return "high"
        return "chaos"
    df["vol_state"] = df["atr_pct"].apply(_vol_bucket)

    # --- velocidad (retorno porcentual por vela) ---
    df["ret_pct_1"] = df["close"].pct_change() * 100.0
    # umbral de "rápido" en función del histórico reciente (percentil 70)
    thr_speed = df["ret_pct_1"].abs().quantile(0.70)
    def _speed_state(v):
        if pd.isna(v):
            return None
        return "high" if abs(v) >= thr_speed else "low"
    df["speed_state"] = df["ret_pct_1"].apply(_speed_state)

    # --- volumen relativo (low / normal / high) ---
    qv1, qv2 = df["volume"].quantile([0.33, 0.66])
    def _volume_state(v):
        if pd.isna(v):
            return None
        if v <= qv1:
            return "low"
        if v <= qv2:
            return "normal"
        return "high"
    df["volume_state"] = df["volume"].apply(_volume_state)

    # --- posición del precio en su rango reciente (0=piso, 1=techo) ---
    # Responde: ¿estoy comprando caro o barato dentro del contexto reciente?
    high_20 = df["high"].rolling(20, min_periods=1).max()
    low_20  = df["low"].rolling(20, min_periods=1).min()
    rng_20  = (high_20 - low_20).replace(0, np.nan)
    df["price_position"] = ((df["close"] - low_20) / rng_20).clip(0.0, 1.0)

    # --- volumen confirma tendencia ---
    # up + vol high/normal = convicción real · up + vol low = posible trampa
    def _vol_confirms(row):
        ts   = row["trend_state"]
        vs   = row["volume_state"]
        if ts == "flat-weak" or vs is None: return False
        if ts == "up"   and vs in ("normal", "high"): return True
        if ts == "down" and vs in ("normal", "high"): return True
        return False
    df["vol_confirms_trend"] = df.apply(_vol_confirms, axis=1)

    # --- niveles de soporte y resistencia recientes (últimas 20 velas) ---
    # Precio cerca de un máximo/mínimo donde el mercado ya reaccionó antes
    recent_high = df["high"].rolling(20, min_periods=1).max().shift(1)
    recent_low  = df["low"].rolling(20, min_periods=1).min().shift(1)
    df["near_resistance"] = (recent_high - df["close"]).abs() <= df["atr"]
    df["near_support"]    = (df["close"] - recent_low).abs()  <= df["atr"]

    # --- indicadores semanales (últimas 5 velas) ---
    # Útiles sobre timeframe 1d — dan contexto de la semana completa
    # price_position_1w: dónde está el precio en el rango semanal (0=fondo, 1=techo)
    high_5 = df["high"].rolling(5, min_periods=1).max()
    low_5  = df["low"].rolling(5, min_periods=1).min()
    rng_5  = (high_5 - low_5).replace(0, np.nan)
    df["price_position_1w"] = ((df["close"] - low_5) / rng_5).clip(0.0, 1.0)

    # weekly_range_pct: cuánto % se movió esta semana (< 3% muerta, > 8% consumida)
    df["weekly_range_pct"] = ((high_5 - low_5) / low_5 * 100).fillna(0)

    # week_trend_pct: retorno desde hace 4 velas hasta ahora (positivo=alcista)
    open_4 = df["open"].shift(4)
    df["week_trend_pct"] = ((df["close"] - open_4) / open_4 * 100).fillna(0)

    keep = ["timestamp","ts","open","high","low","close","volume",
        "ema_5","ema_12","ema_20","ema_50","ema_200",
        "rsi_14","atr_pct","bb_low","bb_mid","bb_high","bb_width",
        "macd","macd_signal","macd_hist","stoch_k","stoch_d",
        "range","body","upper_wick_pct","lower_wick_pct",
        "trend_state","vol_state",
        "ret_pct_1","speed_state","volume_state",
        "price_position","vol_confirms_trend","near_support","near_resistance",
        "price_position_1w","weekly_range_pct","week_trend_pct"]

    for c in keep:
        if c not in df.columns: df[c] = pd.NA
    return df[keep]

def build_indicators_for_all(ohlcv_cache: Dict[Tuple[str,str], pd.DataFrame]) -> Dict[Tuple[str,str], pd.DataFrame]:
    out: Dict[Tuple[str,str], pd.DataFrame] = {}
    for key, df in ohlcv_cache.items():
        out[key] = indicators_from_ohlcv(df) if df is not None else pd.DataFrame()
    return out

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 6 — Nivel 4: Orderbook y derivados
# ──────────────────────────────────────────────────────────────────────────────

def fetch_orderbook_safe(ex: ccxt.Exchange, symbol: str, levels: int) -> tuple[pd.DataFrame, int]:
    """
    Llama fetch_order_book con una lista de límites válidos si el 'levels' pedido falla.
    Retorna (df_levels, latency_ms). No levanta excepción: devuelve df vacío si no hay datos.
    """
    t0 = ex.milliseconds()
    # orden de prueba: el pedido, luego válidos comunes de Binance/ccxt
    candidates = []
    if levels not in (5, 10, 20, 50, 100, 500):
        candidates = [levels, 20, 10, 5, 50, 100]
    else:
        candidates = [levels, 20, 10, 5, 50, 100]

    last_err = None
    for lim in candidates:
        try:
            ob = ex.fetch_order_book(symbol, limit=int(lim))
            t1 = ex.milliseconds()
            ts = _utc_iso()
            bids = (ob.get("bids") or [])[:int(lim)]
            asks = (ob.get("asks") or [])[:int(lim)]
            rows = []
            for i, pair in enumerate(bids, 1):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2: continue
                px, qty = pair[0], pair[1]
                rows.append({"symbol": symbol, "ts": ts, "side": "bid", "level": i, "px": float(px), "qty": float(qty)})
            for i, pair in enumerate(asks, 1):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2: continue
                px, qty = pair[0], pair[1]
                rows.append({"symbol": symbol, "ts": ts, "side": "ask", "level": i, "px": float(px), "qty": float(qty)})
            return pd.DataFrame(rows), int(t1 - t0)
        except Exception as e:
            last_err = e
            continue
    # si todos fallan, devolvemos df vacío con la latencia total acumulada
    t1 = ex.milliseconds()
    return pd.DataFrame(columns=["symbol","ts","side","level","px","qty"]), int(t1 - t0)


def read_orderbook_levels(ex: ccxt.Exchange, symbol: str, levels: int) -> Tuple[pd.DataFrame, int]:
    t0 = ex.milliseconds()
    ob = ex.fetch_order_book(symbol, limit=levels)
    t1 = ex.milliseconds()
    ts = _utc_iso()

    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    # algunos exchanges sólo aceptan ciertos 'limit'; si viene más largo, truncá igual:
    bids = bids[:levels] if isinstance(bids, list) else []
    asks = asks[:levels] if isinstance(asks, list) else []

    rows = []
    for i, pair in enumerate(bids, 1):
        if not isinstance(pair, (list, tuple)) or len(pair) < 2: continue
        px, qty = pair[0], pair[1]
        rows.append({"symbol": symbol, "ts": ts, "side": "bid", "level": i, "px": float(px), "qty": float(qty)})
    for i, pair in enumerate(asks, 1):
        if not isinstance(pair, (list, tuple)) or len(pair) < 2: continue
        px, qty = pair[0], pair[1]
        rows.append({"symbol": symbol, "ts": ts, "side": "ask", "level": i, "px": float(px), "qty": float(qty)})

    return pd.DataFrame(rows), int(t1 - t0)


def build_depth_snapshot(levels_df: pd.DataFrame, last_price: Optional[float], atr_pct_latest: Optional[float]) -> pd.DataFrame:
    if levels_df is None or levels_df.empty:
        cols = ["symbol","ts","bid_qty_topN","ask_qty_topN","bid_notional_topN","ask_notional_topN",
                "bid_qty_zone1","ask_qty_zone1","bid_qty_zone2","ask_qty_zone2","bid_qty_zone3","ask_qty_zone3"]
        return pd.DataFrame(columns=cols)
    df = levels_df.copy()
    df["notional"] = df["px"] * df["qty"]
    def _zones_mask(px_row, last, atr_pct):
        if last is None or atr_pct is None or atr_pct <= 0: return [False, False, False]
        dist_pct = abs(px_row - last) / last * 100.0
        return [dist_pct <= (0.3*atr_pct), dist_pct <= (0.7*atr_pct), dist_pct <= (1.5*atr_pct)]
    ts = df["ts"].iloc[0]; sym = df["symbol"].iloc[0]
    top_bid = df[df["side"]=="bid"]; top_ask = df[df["side"]=="ask"]
    bid_qty_topN = float(top_bid["qty"].sum()); ask_qty_topN = float(top_ask["qty"].sum())
    bid_notional_topN = float(top_bid["notional"].sum()); ask_notional_topN = float(top_ask["notional"].sum())
    b1=b2=b3=a1=a2=a3=0.0
    if last_price is not None and atr_pct_latest is not None and atr_pct_latest>0:
        for _, r in top_bid.iterrows():
            z1,z2,z3 = _zones_mask(r["px"], last_price, atr_pct_latest)
            if z1: b1 += r["qty"]
            if z2: b2 += r["qty"]
            if z3: b3 += r["qty"]
        for _, r in top_ask.iterrows():
            z1,z2,z3 = _zones_mask(r["px"], last_price, atr_pct_latest)
            if z1: a1 += r["qty"]
            if z2: a2 += r["qty"]
            if z3: a3 += r["qty"]
    return pd.DataFrame([{
        "symbol": sym, "ts": ts,
        "bid_qty_topN": bid_qty_topN, "ask_qty_topN": ask_qty_topN,
        "bid_notional_topN": bid_notional_topN, "ask_notional_topN": ask_notional_topN,
        "bid_qty_zone1": b1, "ask_qty_zone1": a1,
        "bid_qty_zone2": b2, "ask_qty_zone2": a2,
        "bid_qty_zone3": b3, "ask_qty_zone3": a3,
    }])

def build_imbalance(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df is None or snapshot_df.empty:
        return pd.DataFrame(columns=["symbol","ts","imb_topN","imb_zone1","imb_zone2","imb_zone3"])
    s = snapshot_df.iloc[0]
    def _imb(b, a):
        tot = (b + a);  return None if (tot is None or tot == 0) else float((b - a)/tot)
    return pd.DataFrame([{
        "symbol": s["symbol"], "ts": s["ts"],
        "imb_topN": _imb(s["bid_qty_topN"], s["ask_qty_topN"]),
        "imb_zone1": _imb(s["bid_qty_zone1"], s["ask_qty_zone1"]),
        "imb_zone2": _imb(s["bid_qty_zone2"], s["ask_qty_zone2"]),
        "imb_zone3": _imb(s["bid_qty_zone3"], s["ask_qty_zone3"]),
    }])

class WallTracker:
    """Rastrea persistencia de “paredes” y calcula si están cerca del precio (zone1)."""
    def __init__(self):
        self._state: Dict[Tuple[str,str,float], Dict[str, float]] = {}
    def update_and_extract(self, levels_df: pd.DataFrame,
                           last_price: Optional[float] = None,
                           atr_pct_latest: Optional[float] = None,
                           z_min: float = 2.5) -> pd.DataFrame:
        if levels_df is None or levels_df.empty:
            return pd.DataFrame(columns=["symbol","ts","side","px","qty","zscore_qty",
                                         "persistence_s","wall_flag","near_flag"])
        ts = levels_df["ts"].iloc[0]
        #epoch_now = datetime.fromisoformat(ts.replace("Z","")).replace(tzinfo=None).timestamp()
        try:
            epoch_now = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            epoch_now = datetime.now(timezone.utc).timestamp()
        out_rows = []
        for (sym, side), side_df in levels_df.groupby(["symbol","side"]):
            qtys = side_df["qty"].values.astype(float)
            med = np.median(qtys) if qtys.size else 0.0
            mad = np.median(np.abs(qtys - med)) if qtys.size else 0.0
            denom = (1.4826*mad) if mad>0 else (np.std(qtys) if qtys.size>1 else 1.0)
            for _, r in side_df.iterrows():
                key = (r["symbol"], r["side"], float(r["px"]))
                prev = self._state.get(key)
                if prev is None:
                    self._state[key] = {"first_ts": epoch_now, "last_ts": epoch_now, "qty": float(r["qty"])}
                    persistence_s = 0.0
                else:
                    prev["last_ts"] = epoch_now; prev["qty"] = float(r["qty"])
                    persistence_s = float(prev["last_ts"] - prev["first_ts"])
                z = (float(r["qty"]) - med) / denom if denom>0 else 0.0
                wall_flag = bool(z >= z_min and persistence_s >= 3.0)
                near_flag = False
                if last_price is not None and atr_pct_latest is not None and atr_pct_latest>0:
                    dist_pct = abs(float(r["px"]) - float(last_price)) / float(last_price) * 100.0
                    near_flag = dist_pct <= (0.3 * float(atr_pct_latest))
                out_rows.append({
                    "symbol": r["symbol"], "ts": ts, "side": r["side"], "px": float(r["px"]), "qty": float(r["qty"]),
                    "zscore_qty": float(z), "persistence_s": persistence_s, "wall_flag": wall_flag, "near_flag": near_flag
                })
        return pd.DataFrame(out_rows)

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 7 — Nivel 5: Watchlist (indicadores clave)
# ──────────────────────────────────────────────────────────────────────────────

def build_indicators_watchlist(indicators_cache, market_ticks, watchlist_symbols: Set[str], timeframes_focus: List[str]) -> pd.DataFrame:
    rows = []
    last_map = {r["symbol"]: r["last"] for _, r in market_ticks.iterrows()} if market_ticks is not None else {}
    for sym in watchlist_symbols:
        for tf in timeframes_focus:
            key = (sym, tf); df = indicators_cache.get(key)
            if df is None or df.empty: continue
            last_row = df.iloc[-1]
            rows.append({
                "symbol": sym, "timeframe": tf, "ts": last_row["ts"],
                "last": last_map.get(sym),
                "rsi_14": last_row["rsi_14"],
                "ema_5": last_row["ema_5"], "ema_12": last_row["ema_12"], "ema_20": last_row["ema_20"],
                "ema_50": last_row["ema_50"], "ema_200": last_row["ema_200"],
                "macd": last_row["macd"], "macd_signal": last_row["macd_signal"], "macd_hist": last_row["macd_hist"],
                "atr_pct": last_row["atr_pct"], "bb_width": last_row["bb_width"],
                "trend_state": last_row["trend_state"], "vol_state": last_row["vol_state"]
            })
    cols = ["symbol","timeframe","ts","last","rsi_14","ema_5","ema_12","ema_20","ema_50","ema_200",
            "macd","macd_signal","macd_hist","atr_pct","bb_width","trend_state","vol_state"]
    return pd.DataFrame(rows, columns=cols)

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 8 — Nivel 6: Correlaciones, volumen anómalo y régimen global
# ──────────────────────────────────────────────────────────────────────────────

def build_correlation_matrix(ohlcv_cache, symbols: List[str], base_tf: str, window: int) -> Tuple[pd.DataFrame, int]:
    series_map = {}; idx_union = None
    for sym in symbols:
        df = ohlcv_cache.get((sym, base_tf))
        if df is None or df.empty: continue
        sr = df.set_index("timestamp")["close"].astype(float)
        series_map[sym] = sr
        idx_union = sr.index if idx_union is None else idx_union.union(sr.index)
    if not series_map or idx_union is None:
        return pd.DataFrame(columns=["ts","window","method","symbol_i","symbol_j","rho"]), 0
    prices = pd.DataFrame({k: v.reindex(idx_union) for k, v in series_map.items()}).ffill()
    rets = prices.pct_change().dropna()
    eff = len(rets)
    if eff < max(window, 2):
        return pd.DataFrame(columns=["ts","window","method","symbol_i","symbol_j","rho"]), eff
    rets_win = rets.iloc[-window:]
    corr = rets_win.corr(method="pearson")
    ts = pd.to_datetime(idx_union.max(), unit="ms", utc=True)
    rows = []
    syms = corr.columns.tolist()
    for i in range(len(syms)):
        for j in range(i, len(syms)):
            rows.append({"ts": ts, "window": window, "method": "pearson",
                         "symbol_i": syms[i], "symbol_j": syms[j], "rho": corr.iloc[i, j]})
    return pd.DataFrame(rows), eff

def build_volume_anomaly(ohlcv_cache, symbols: List[str], timeframes: List[str], window: int) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        for tf in timeframes:
            df = ohlcv_cache.get((sym, tf))
            if df is None or df.empty or len(df) < window: continue
            vol = df["volume"].astype(float)
            mu = vol.rolling(window, min_periods=window).mean()
            sd = vol.rolling(window, min_periods=window).std(ddof=0)
            z = (vol - mu) / sd.replace(0, np.nan)
            last = df.iloc[-1]
            last_z = z.iloc[-1]
            rows.append({
                "symbol": sym, "timeframe": tf, "ts": last["ts"],
                "vol_zscore": last_z, "vol_spike_flag": bool(last_z is not None and last_z >= 3.0),
                "rolling_mean": mu.iloc[-1], "rolling_std": sd.iloc[-1]
            })
    cols = ["symbol","timeframe","ts","vol_zscore","vol_spike_flag","rolling_mean","rolling_std"]
    return pd.DataFrame(rows, columns=cols)

def build_vol_regime_global(indicators_cache, symbols: List[str], ref_symbol: str, tf_for_breadth: str) -> pd.DataFrame:
    ref = indicators_cache.get((ref_symbol, tf_for_breadth))
    vol_state_global = atr_ref_pct = bb_ref_width = None
    if ref is not None and not ref.empty:
        last = ref.iloc[-1]
        vol_state_global = last.get("vol_state")
        atr_ref_pct = last.get("atr_pct")
        bb_ref_width = last.get("bb_width")
    up_count = 0; total = 0
    for sym in symbols:
        df = indicators_cache.get((sym, tf_for_breadth))
        if df is None or df.empty: continue
        st = df.iloc[-1].get("trend_state")
        if st is None: continue
        total += 1;  up_count += (st == "up")
    breadth = (up_count / total * 100.0) if total > 0 else None
    return pd.DataFrame([{
        "ts": _utc_iso(), "ref_symbol": ref_symbol,
        "vol_state_global": vol_state_global, "atr_ref_pct": atr_ref_pct,
        "bb_ref_width": bb_ref_width, "breadth_trend_up_pct": breadth
    }])

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 9 — Orquestador con toggles, TTLs y world_summary
# ──────────────────────────────────────────────────────────────────────────────

BUCKET_TF_FOCUS = {
    1: ["1m","5m"],          # day/surf
    2: ["5m","15m","1h"],    # oportunista
    3: ["1h","4h"],          # holder
}

class WorldReader:
    def __init__(self, symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None):
        self.symbols = symbols or SYMBOLS
        self.timeframes = timeframes or TIMEFRAMES
        self.exchange = build_exchange()
        # caches
        self.df_market_ticks: Optional[pd.DataFrame] = None
        self.ohlcv_cache: Dict[Tuple[str,str], pd.DataFrame] = {}
        self.indicators_cache: Dict[Tuple[str,str], pd.DataFrame] = {}
        # orderbook
        self.wall_tracker = WallTracker()
        # nivel 6 fallback
        self.df_corr_last: Optional[pd.DataFrame] = None
        self.df_vol_anom_last: Optional[pd.DataFrame] = None
        self.df_vol_regime_last: Optional[pd.DataFrame] = None
        # timestamps para TTL
        self._ts_ticks_iso: Optional[str] = None
        self._ts_orderbook_iso: Optional[str] = None
        # telemetría
        self._lat_ms_ticks = 0
        self._lat_ms_ohlcv_avg = 0
        self._lat_ms_orderbook_avg = 0

    # Helpers para M3
    def last_price(self, symbol: str) -> Optional[float]:
        if self.df_market_ticks is None or self.df_market_ticks.empty: return None
        m = self.df_market_ticks
        row = m[m["symbol"] == symbol]
        return float(row["last"].iloc[0]) if not row.empty and pd.notna(row["last"].iloc[0]) else None

    def atr_pct_latest(self, symbol: str, tf: str = "1m") -> Optional[float]:
        df = self.indicators_cache.get((symbol, tf))
        if df is None or df.empty: return None
        v = df.iloc[-1]["atr_pct"]
        return float(v) if pd.notna(v) else None

    def imb_zone1(self, df_imbalance: pd.DataFrame, symbol: str) -> Optional[float]:
        if df_imbalance is None or df_imbalance.empty: return None
        r = df_imbalance[df_imbalance["symbol"] == symbol]
        if r.empty: return None
        v = r.iloc[-1]["imb_zone1"]
        return float(v) if pd.notna(v) else None

    def read_once(self,
                  snapshot_id: str,
                  loop_idx: int,
                  watchlist_symbols: Optional[Set[str]] = None,
                  force_flags: Optional[Dict[str, bool]] = None,
                  *,
                  bucket_focus: Optional[int] = None,
                  use_orderbook: Optional[bool] = None) -> Dict[str, object]:

        force_flags = force_flags or {}
        errors: List[str] = []
        ts_cycle = _utc_iso()

        # Timeframes de foco para panel watchlist (no afecta construcción de indicadores base)
        tf_focus = BUCKET_TF_FOCUS.get(bucket_focus, ["1m","5m","15m"])

        # N1 — Tickers
        try:
            self.df_market_ticks, self._lat_ms_ticks = read_market_ticks(self.exchange, self.symbols)
            self._ts_ticks_iso = _utc_iso()
        except Exception as e:
            errors.append(f"ticks:{type(e).__name__}")
            self.df_market_ticks = pd.DataFrame()

        # N2/N3 — OHLCV + Indicadores
        try:
            self.ohlcv_cache, self._lat_ms_ohlcv_avg = read_all_ohlcv(self.exchange, self.symbols, self.timeframes, self.ohlcv_cache, TF_LIMITS)
            self.indicators_cache = build_indicators_for_all(self.ohlcv_cache)
        except Exception as e:
            errors.append(f"ohlcv/ind:{type(e).__name__}")

        # N4 — Orderbook (muestreado / forzado / deshabilitado por toggle)
        ob_enabled = (use_orderbook if use_orderbook is not None else FEATURES_ORDERBOOK_ENABLED)
        df_orderbook_levels = pd.DataFrame()
        df_orderbook_depth_snapshot = pd.DataFrame()
        df_orderbook_imbalance = pd.DataFrame()
        df_liquidity_persistence = pd.DataFrame()
        try:
            if ob_enabled and (force_flags.get("orderbook") or (loop_idx % ORDERBOOK_SAMPLE_EVERY == 0)):
                levels_rows = []; depth_rows = []; imb_rows = []; persist_rows = []
                _ticks = self.df_market_ticks if self.df_market_ticks is not None and not self.df_market_ticks.empty else pd.DataFrame()
                last_map = {r["symbol"]: r["last"] for _, r in _ticks.iterrows()}
                atr_map = {sym: self.atr_pct_latest(sym, "1m") for sym in self.symbols}
                
                latencies = []
                for sym in self.symbols:
                    lv, l_ms = fetch_orderbook_safe(self.exchange, sym, ORDERBOOK_LEVELS)
                    latencies.append(l_ms)
                    if lv.empty: 
                        continue
                    levels_rows.append(lv)
                    depth = build_depth_snapshot(lv, last_map.get(sym), atr_map.get(sym))
                    imb   = build_imbalance(depth)
                    persist = self.wall_tracker.update_and_extract(lv, last_map.get(sym), atr_map.get(sym), z_min=2.5)
                    depth_rows.append(depth); imb_rows.append(imb); persist_rows.append(persist)
                self._lat_ms_orderbook_avg = int(np.mean(latencies)) if latencies else 0
                self._ts_orderbook_iso = _utc_iso()
                if levels_rows: df_orderbook_levels = pd.concat(levels_rows, ignore_index=True)
                if depth_rows: df_orderbook_depth_snapshot = pd.concat(depth_rows, ignore_index=True)
                if imb_rows: df_orderbook_imbalance = pd.concat(imb_rows, ignore_index=True)
                if persist_rows: df_liquidity_persistence = pd.concat(persist_rows, ignore_index=True)
        except Exception as e:
            errors.append(f"orderbook:{type(e).__name__}: {e}")

        # N5 — Watchlist indicadores (enfocado en símbolos con inventario)
        df_indicators_watchlist = pd.DataFrame()
        try:
            if watchlist_symbols:
                df_indicators_watchlist = build_indicators_watchlist(self.indicators_cache, self.df_market_ticks, set(watchlist_symbols), timeframes_focus=tf_focus)
        except Exception as e:
            errors.append(f"watchlist:{type(e).__name__}")

        # N6 — Correlaciones, volumen anómalo y régimen global
        df_correlation_matrix = pd.DataFrame()
        df_volume_anomaly = pd.DataFrame()
        df_vol_regime_global = pd.DataFrame()
        corr_window_effective = 0
        if FEATURES_LEVEL6_ENABLED:
            try:
                df_correlation_matrix, corr_window_effective = build_correlation_matrix(self.ohlcv_cache, self.symbols, IND_BASE_TF, CORR_WINDOW)
                self.df_corr_last = df_correlation_matrix if not df_correlation_matrix.empty else self.df_corr_last
            except Exception as e:
                errors.append(f"corr:{type(e).__name__}")
                df_correlation_matrix = self.df_corr_last if self.df_corr_last is not None else pd.DataFrame()
            try:
                df_volume_anomaly = build_volume_anomaly(self.ohlcv_cache, self.symbols, self.timeframes, VOL_ANOM_WINDOW)
                self.df_vol_anom_last = df_volume_anomaly if not df_volume_anomaly.empty else self.df_vol_anom_last
            except Exception as e:
                errors.append(f"volanom:{type(e).__name__}")
                df_volume_anomaly = self.df_vol_anom_last if self.df_vol_anom_last is not None else pd.DataFrame()
            try:
                ref_symbol = self.symbols[0]
                df_vol_regime_global = build_vol_regime_global(self.indicators_cache, self.symbols, ref_symbol, "1h")
                self.df_vol_regime_last = df_vol_regime_global if not df_vol_regime_global.empty else self.df_vol_regime_last
            except Exception as e:
                errors.append(f"volregime:{type(e).__name__}")
                df_vol_regime_global = self.df_vol_regime_last if self.df_vol_regime_last is not None else pd.DataFrame()

        # ── Frescura / TTLs
        def _age_s_from_iso(ts_iso: Optional[str]) -> Optional[float]:
            if not ts_iso: return None
            dt = datetime.fromisoformat(ts_iso.replace("Z","")).replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())

        ttl = {
            "ttl_ticks_s": _age_s_from_iso(self._ts_ticks_iso),
            "ttl_orderbook_s": _age_s_from_iso(self._ts_orderbook_iso),
            "ttl_ohlcv_s": {}
        }
        for tf in self.timeframes:
            ages = []
            for sym in self.symbols:
                df = self.ohlcv_cache.get((sym, tf))
                if df is None or df.empty: continue
                last_ts = int(df["timestamp"].iloc[-1])
                dt = datetime.fromtimestamp(last_ts/1000.0, tz=timezone.utc)
                ages.append(max(0.0, (datetime.now(timezone.utc) - dt).total_seconds()))
            ttl["ttl_ohlcv_s"][tf] = float(np.mean(ages)) if ages else None

        # ── world_summary por símbolo
        def _last_indicator(sym: str, tf: str, col: str):
            df = self.indicators_cache.get((sym, tf))
            if df is None or df.empty: return None
            v = df.iloc[-1].get(col);  return (float(v) if isinstance(v, (int,float,np.floating)) else (None if pd.isna(v) else v))
        def _vol_spike_1m(sym: str):
            df = df_volume_anomaly
            if df is None or df.empty: return None
            r = df[(df["symbol"]==sym) & (df["timeframe"]=="1m")]
            if r.empty: return None
            v = r.iloc[-1]["vol_spike_flag"]
            return bool(v) if pd.notna(v) else None
        def _imb_zone1_sym(sym: str):
            if df_orderbook_imbalance is None or df_orderbook_imbalance.empty: return None
            r = df_orderbook_imbalance[df_orderbook_imbalance["symbol"]==sym]
            if r.empty: return None
            v = r.iloc[-1]["imb_zone1"]
            return float(v) if pd.notna(v) else None
        def _wall_near_flag_sym(sym: str):
            if df_liquidity_persistence is None or df_liquidity_persistence.empty: return None
            r = df_liquidity_persistence[(df_liquidity_persistence["symbol"]==sym)]
            if r.empty: return None
            rr = r.iloc[-50:]  # ventana corta
            any_near = bool(((rr["wall_flag"]==True) & (rr["near_flag"]==True)).any())
            return any_near

        def _wall_persistence_sym(sym: str) -> float:
            """
            Retorna la persistencia máxima en segundos del muro activo más reciente.
            0.0 si no hay wall activo. Permite distinguir fake whale (fugaz) de whale real (>=60s).
            """
            if df_liquidity_persistence is None or df_liquidity_persistence.empty:
                return 0.0
            r = df_liquidity_persistence[df_liquidity_persistence["symbol"] == sym]
            if r.empty:
                return 0.0
            active = r[(r["wall_flag"] == True) & (r["near_flag"] == True)]
            if active.empty:
                return 0.0
            if "persistence_s" in active.columns:
                return float(active["persistence_s"].max())
            return 0.0

        def _weekly_indicator(sym: str, field: str):
            """Lee un indicador semanal calculado sobre velas 1d."""
            key = (sym, "1d")
            if key not in self.indicators_cache:
                return None
            df_ind = self.indicators_cache[key]
            if df_ind is None or df_ind.empty or field not in df_ind.columns:
                return None
            val = df_ind.iloc[-1][field]
            return None if pd.isna(val) else float(val)

        world_rows = []
        for sym in self.symbols:
            last = self.last_price(sym)
            spread_pct = None
            if self.df_market_ticks is not None and not self.df_market_ticks.empty:
                tm = self.df_market_ticks
                rr = tm[tm["symbol"]==sym]
                spread_pct = float(rr.iloc[-1]["spread_pct"]) if not rr.empty and pd.notna(rr.iloc[-1]["spread_pct"]) else None
            # ── Helpers para rangos y espacios disponibles ──────────────
            def _range_pct(sym: str, tf: str):
                """% de rango (high-low) / precio en el timeframe dado."""
                key = (sym, tf)
                if key not in self.indicators_cache:
                    return None
                df = self.indicators_cache[key]
                if df.empty:
                    return None
                row = df.iloc[-1]
                h = float(row["high"]) if "high" in df.columns and pd.notna(row.get("high")) else None
                l = float(row["low"])  if "low"  in df.columns and pd.notna(row.get("low"))  else None
                c = float(row["close"]) if "close" in df.columns and pd.notna(row.get("close")) else None
                if h is None or l is None or c is None or c == 0:
                    return None
                try:
                    return round((h - l) / c * 100, 4)
                except:
                    return None

            def _upside_pct(sym: str, tf: str):
                """% disponible hacia arriba: (1 - pos) × range_pct."""
                pos = _last_indicator(sym, tf, "price_position")
                rng = _range_pct(sym, tf)
                if pos is None or rng is None:
                    return None
                return round((1.0 - pos) * rng, 4)

            def _downside_pct(sym: str, tf: str):
                """% disponible hacia abajo: pos × range_pct."""
                pos = _last_indicator(sym, tf, "price_position")
                rng = _range_pct(sym, tf)
                if pos is None or rng is None:
                    return None
                return round(pos * rng, 4)

            summary = {
                "symbol": sym,
                "ts": ts_cycle,
                "last": last,
                "spread_pct": spread_pct,
                # ── Tendencia multi-timeframe (campo de visión completo)
                "trend_1m":  _last_indicator(sym, "1m",  "trend_state"),
                "trend_5m":  _last_indicator(sym, "5m",  "trend_state"),
                "trend_15m": _last_indicator(sym, "15m", "trend_state"),
                "trend_1h":  _last_indicator(sym, "1h",  "trend_state"),  # contexto del día
                "trend_4h":  _last_indicator(sym, "4h",  "trend_state"),  # contexto de la semana
                "trend_1d":  _last_indicator(sym, "1d",  "trend_state"),  # contexto del mes
                # ── Volatilidad y momentum (1m — decisión inmediata)
                "vol_state_1m":    _last_indicator(sym, "1m", "vol_state"),
                "atr_pct_1m":      _last_indicator(sym, "1m", "atr_pct"),
                "bb_width_1m":     _last_indicator(sym, "1m", "bb_width"),
                "speed_state_1m":  _last_indicator(sym, "1m", "speed_state"),
                "volume_state_1m": _last_indicator(sym, "1m", "volume_state"),
                # ── Contexto de precio (¿caro o barato en su rango reciente?)
                "price_position_1m": _last_indicator(sym, "1m", "price_position"),
                "price_position_1h": _last_indicator(sym, "1h", "price_position"),
                # ── Confirmación y niveles clave
                "vol_confirms_trend": _last_indicator(sym, "1m", "vol_confirms_trend"),
                "near_support":       _last_indicator(sym, "1m", "near_support"),
                "near_resistance":    _last_indicator(sym, "1m", "near_resistance"),
                # ── Señales de monstruos / orderbook
                "imb_zone1":          _imb_zone1_sym(sym),
                "wall_near_flag":     _wall_near_flag_sym(sym),
                "wall_persistence_s": _wall_persistence_sym(sym),
                "vol_spike_1m":       _vol_spike_1m(sym),
                "ohlcv_age_1m_s":     ttl["ttl_ohlcv_s"].get("1m"),
                # ── Indicadores semanales (calculados sobre velas 1d)
                "price_position_1w":  _weekly_indicator(sym, "price_position_1w"),
                "weekly_range_pct":   _weekly_indicator(sym, "weekly_range_pct"),
                "week_trend_pct":     _weekly_indicator(sym, "week_trend_pct"),
                # ── Posición en rango diario
                "price_position_1d":  _last_indicator(sym, "1d", "price_position"),
                # ── Espacios disponibles arriba/abajo por timeframe
                "range_1m_pct":     _range_pct(sym, "1m"),
                "range_1h_pct":     _range_pct(sym, "1h"),
                "range_1d_pct":     _range_pct(sym, "1d"),
                "upside_1h_pct":    _upside_pct(sym, "1h"),
                "downside_1h_pct":  _downside_pct(sym, "1h"),
                "upside_1d_pct":    _upside_pct(sym, "1d"),
                "downside_1d_pct":  _downside_pct(sym, "1d"),
                # ── Contexto macro: RSI, MACD, slope EMA (para decisiones de calidad)
                # Responden "¿cuánto ya cayó el mercado?" antes de comprar en caída
                "rsi_14_5m":        _last_indicator(sym, "5m",  "rsi_14"),
                "rsi_14_15m":       _last_indicator(sym, "15m", "rsi_14"),
                "rsi_14_1h":        _last_indicator(sym, "1h",  "rsi_14"),
                "macd_hist_1m":     _last_indicator(sym, "1m",  "macd_hist"),
                "macd_hist_5m":     _last_indicator(sym, "5m",  "macd_hist"),
                "ret_pct_1m":       _last_indicator(sym, "1m",  "ret_pct_1"),
            }
            world_rows.append(summary)
        df_world_summary = pd.DataFrame(world_rows)

        # Telemetría y conteos
        calls_this_cycle = {
            "ticks": 1 if self._lat_ms_ticks else 0,
            "ohlcv": len(self.symbols)*len(self.timeframes),
            "orderbook": (len(self.symbols) if not df_orderbook_levels.empty else 0)
        }
        telemetry = {
            "ex_latency_ms_ticks": self._lat_ms_ticks,
            "ex_latency_ms_ohlcv_avg": self._lat_ms_ohlcv_avg,
            "ex_latency_ms_orderbook_avg": self._lat_ms_orderbook_avg,
            "calls_this_cycle": calls_this_cycle,
            "corr_window_effective": corr_window_effective
        }

        # Salida
        out: Dict[str, object] = {
            "ts": ts_cycle,
            "snapshot_id": snapshot_id,
            # Nivel 1
            "df_market_ticks": self.df_market_ticks.copy(deep=True) if self.df_market_ticks is not None else pd.DataFrame(),
            # Nivel 2
            "df_ohlcv": {k: v.copy(deep=True) for k, v in self.ohlcv_cache.items()},
            # Nivel 3
            "df_indicators": {k: v.copy(deep=True) for k, v in self.indicators_cache.items()},
            # Nivel 4
            "df_orderbook_levels": df_orderbook_levels,
            "df_orderbook_depth_snapshot": df_orderbook_depth_snapshot,
            "df_orderbook_imbalance": df_orderbook_imbalance,
            "df_liquidity_persistence": df_liquidity_persistence,
            # Nivel 5
            "df_indicators_watchlist": df_indicators_watchlist,
            # Nivel 6
            "df_correlation_matrix": df_correlation_matrix,
            "df_volume_anomaly": df_volume_anomaly,
            "df_vol_regime_global": df_vol_regime_global,
            # Adicionales
            "world_summary": df_world_summary,
            "ttl": ttl,
            "telemetry": telemetry,
            "errors": errors,
        }
        return out

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 10 — Resumen visual de mercado en terminal
# ──────────────────────────────────────────────────────────────────────────────

def _trend_icon(state: Optional[str]) -> str:
    if state == "up":        return "📈"
    if state == "down":      return "📉"
    if state == "flat-weak": return "➡️ "
    return "❓"

def _vol_icon(state: Optional[str]) -> str:
    icons = {"calm": "🟢", "normal": "🟡", "high": "🟠", "chaos": "🔴"}
    return icons.get(str(state) if state else "", "⚪")

def _speed_icon(state: Optional[str]) -> str:
    return "⚡" if state == "high" else "🐢" if state == "low" else "❓"

def _imb_bar(val: Optional[float], width: int = 10) -> str:
    """Barra visual de imbalance bid/ask (−1 a +1)."""
    if val is None:
        return "─" * width
    filled = round((val + 1) / 2 * width)
    filled = max(0, min(width, filled))
    bar = "█" * filled + "░" * (width - filled)
    label = f"{val:+.2f}"
    return f"[{bar}] {label}"

def print_market_summary(out: dict) -> None:
    """
    Imprime en terminal un resumen legible del snapshot de mercado.
    Secciones: precio/spread, indicadores por timeframe, orderbook y errores.
    """
    SEP  = "─" * 80
    SEP2 = "═" * 80

    ts       = out.get("ts", "?")
    snap_id  = out.get("snapshot_id", "?")
    errors   = out.get("errors", [])
    tel      = out.get("telemetry", {})
    ttl      = out.get("ttl", {})
    ws       = out.get("world_summary")
    df_ticks = out.get("df_market_ticks")
    df_wb    = out.get("df_indicators_watchlist")
    df_ob    = out.get("df_orderbook_imbalance")
    df_regime = out.get("df_vol_regime_global")

    print(f"\n{SEP2}")
    print(f"  📡  MARKET STATUS  ·  {ts}  ·  {snap_id}")
    print(SEP2)

    # ── SECCIÓN 1: Precio y spread ────────────────────────────────────────────
    print(f"\n  {'SÍMBOLO':<12} {'ÚLTIMO PRECIO':>16} {'BID':>14} {'ASK':>14} {'SPREAD %':>10}  LATENCIA")
    print(f"  {SEP}")
    if df_ticks is not None and not df_ticks.empty:
        for _, r in df_ticks.iterrows():
            last   = r.get("last");     bid = r.get("bid");   ask = r.get("ask")
            spr    = r.get("spread_pct")
            lat_ms = r.get("ex_latency_ms")
            fmt_last = f"{float(last):>16,.2f}" if pd.notna(last) else f"{'N/A':>16}"
            fmt_bid  = f"{float(bid):>14,.2f}"  if pd.notna(bid)  else f"{'N/A':>14}"
            fmt_ask  = f"{float(ask):>14,.2f}"  if pd.notna(ask)  else f"{'N/A':>14}"
            fmt_spr  = f"{float(spr):>9,.4f}%" if pd.notna(spr)  else f"{'N/A':>9} "
            fmt_lat  = f"{int(lat_ms)} ms"      if pd.notna(lat_ms) else "N/A"
            print(f"  {str(r['symbol']):<12} {fmt_last} {fmt_bid} {fmt_ask} {fmt_spr}  {fmt_lat}")
    else:
        print("  ⚠️  Sin datos de ticks.")

    # ── SECCIÓN 2: World summary (señales por símbolo) ────────────────────────
    print(f"\n  {SEP}")
    print(f"  📊  SEÑALES DE MERCADO")
    print(f"  {SEP}")
    if ws is not None and not ws.empty:
        # Tendencia multi-timeframe
        print(f"\n  {'SÍMBOLO':<12} {'1m':<12} {'5m':<12} {'15m':<12} {'1h':<12} {'4h':<12} {'1d':<12}")
        print(f"  {SEP}")
        for _, r in ws.iterrows():
            sym = str(r.get("symbol", ""))
            def _fmt_trend(t):
                return f"{_trend_icon(t)} {str(t or '?'):<8}"
            print(f"  {sym:<12} "
                  f"{_fmt_trend(r.get('trend_1m')):<14} "
                  f"{_fmt_trend(r.get('trend_5m')):<14} "
                  f"{_fmt_trend(r.get('trend_15m')):<14} "
                  f"{_fmt_trend(r.get('trend_1h')):<14} "
                  f"{_fmt_trend(r.get('trend_4h')):<14} "
                  f"{_fmt_trend(r.get('trend_1d')):<14}")

        # Volatilidad, momentum y contexto de precio
        print(f"\n  {'SÍMBOLO':<12} {'VOL STATE':<12} {'SPEED':<10} {'ATR%':>7} {'BB W%':>7} "
              f"{'PX POS 1m':>10} {'PX POS 1h':>10} {'VOL OK':>7} {'SUPP':>6} {'RES':>6}")
        print(f"  {SEP}")
        for _, r in ws.iterrows():
            sym  = str(r.get("symbol", ""))
            vs   = r.get("vol_state_1m")
            spd  = r.get("speed_state_1m")
            atr  = r.get("atr_pct_1m")
            bbw  = r.get("bb_width_1m")
            pp1m = r.get("price_position_1m")
            pp1h = r.get("price_position_1h")
            vct  = r.get("vol_confirms_trend")
            ns   = r.get("near_support")
            nr   = r.get("near_resistance")
            fmt_vs   = f"{_vol_icon(vs)} {str(vs or '?'):<8}"
            fmt_spd  = f"{_speed_icon(spd)} {str(spd or '?'):<5}"
            fmt_atr  = f"{float(atr):>7.3f}"  if pd.notna(atr)  else f"{'N/A':>7}"
            fmt_bbw  = f"{float(bbw):>7.3f}"  if pd.notna(bbw)  else f"{'N/A':>7}"
            fmt_pp1m = f"{float(pp1m):>10.2f}" if pd.notna(pp1m) else f"{'N/A':>10}"
            fmt_pp1h = f"{float(pp1h):>10.2f}" if pd.notna(pp1h) else f"{'N/A':>10}"
            fmt_vct  = "  ✅" if vct else "  ❌"
            fmt_ns   = "  🟢" if ns  else "  ─ "
            fmt_nr   = "  🔴" if nr  else "  ─ "
            print(f"  {sym:<12} {fmt_vs:<14} {fmt_spd:<12} {fmt_atr} {fmt_bbw} "
                  f"{fmt_pp1m} {fmt_pp1h} {fmt_vct} {fmt_ns} {fmt_nr}")

        # Alertas de monstruos
        spike_syms = [str(r["symbol"]) for _, r in ws.iterrows() if r.get("vol_spike_1m") is True]
        wall_syms  = [str(r["symbol"]) for _, r in ws.iterrows() if r.get("wall_near_flag") is True]
        supp_syms  = [str(r["symbol"]) for _, r in ws.iterrows() if r.get("near_support") is True]
        res_syms   = [str(r["symbol"]) for _, r in ws.iterrows() if r.get("near_resistance") is True]
        if spike_syms: print(f"\n  ⚡ Vol spike 1m  : {', '.join(spike_syms)}")
        if wall_syms:  print(f"  🧱 Pared cercana : {', '.join(wall_syms)}")
        if supp_syms:  print(f"  🟢 Cerca soporte : {', '.join(supp_syms)}")
        if res_syms:   print(f"  🔴 Cerca resisten: {', '.join(res_syms)}")
    else:
        print("  ⚠️  Sin world_summary disponible.")

    # ── SECCIÓN 3: Indicadores watchlist ─────────────────────────────────────
    if df_wb is not None and not df_wb.empty:
        print(f"\n  {SEP}")
        print(f"  🔬  INDICADORES WATCHLIST")
        print(f"  {SEP}")
        print(f"  {'SÍMBOLO':<12} {'TF':<6} {'RSI':>7} {'EMA5':>10} {'EMA12':>10} "
              f"{'EMA20':>10} {'MACD_H':>9} {'ATR%':>7} {'TREND':<12} {'VOL'}")
        print(f"  {SEP}")
        for _, r in df_wb.iterrows():
            rsi_v   = r.get("rsi_14");    ema5  = r.get("ema_5");   ema12 = r.get("ema_12")
            ema20   = r.get("ema_20");    mh    = r.get("macd_hist"); atr_v = r.get("atr_pct")
            trend_v = r.get("trend_state"); vol_v = r.get("vol_state")
            print(f"  {str(r['symbol']):<12} {str(r['timeframe']):<6} "
                  f"{float(rsi_v):>7.1f}" if pd.notna(rsi_v) else f"  {'?':>7}", end="")
            print(f" {float(ema5):>10,.2f}"  if pd.notna(ema5)  else f" {'N/A':>10}", end="")
            print(f" {float(ema12):>10,.2f}" if pd.notna(ema12) else f" {'N/A':>10}", end="")
            print(f" {float(ema20):>10,.2f}" if pd.notna(ema20) else f" {'N/A':>10}", end="")
            print(f" {float(mh):>+9.2f}"    if pd.notna(mh)    else f" {'N/A':>9}", end="")
            print(f" {float(atr_v):>7.3f}"  if pd.notna(atr_v) else f" {'N/A':>7}", end="")
            print(f"  {_trend_icon(trend_v)} {str(trend_v or '?'):<9} {_vol_icon(vol_v)} {str(vol_v or '?')}")

    # ── SECCIÓN 4: Orderbook imbalance ────────────────────────────────────────
    if FEATURES_ORDERBOOK_ENABLED and df_ob is not None and not df_ob.empty:
        print(f"\n  {SEP}")
        print(f"  📖  ORDERBOOK — IMBALANCE  (−1=asks dominan · +1=bids dominan)")
        print(f"  {SEP}")
        print(f"  {'SÍMBOLO':<12} {'IMB TOTAL':<25} {'ZONA 1':<25} {'ZONA 2':<25}")
        print(f"  {SEP}")
        for _, r in df_ob.iterrows():
            print(f"  {str(r['symbol']):<12} "
                  f"{_imb_bar(r.get('imb_topN')):<25} "
                  f"{_imb_bar(r.get('imb_zone1')):<25} "
                  f"{_imb_bar(r.get('imb_zone2')):<25}")

    # ── SECCIÓN 5: Régimen global de volatilidad ──────────────────────────────
    if FEATURES_LEVEL6_ENABLED and df_regime is not None and not df_regime.empty:
        rg = df_regime.iloc[-1]
        print(f"\n  {SEP}")
        print(f"  🌐  RÉGIMEN GLOBAL  ·  ref: {rg.get('ref_symbol', '?')}")
        print(f"  {SEP}")
        vsg   = rg.get("vol_state_global"); atr_r = rg.get("atr_ref_pct")
        bbw_r = rg.get("bb_ref_width");    br    = rg.get("breadth_trend_up_pct")
        print(f"  Vol state global : {_vol_icon(vsg)} {str(vsg or 'N/A')}")
        print(f"  ATR ref (%)      : {float(atr_r):.3f}" if pd.notna(atr_r) else "  ATR ref (%)      : N/A")
        print(f"  BB Width ref     : {float(bbw_r):.3f}" if pd.notna(bbw_r) else "  BB Width ref     : N/A")
        print(f"  Breadth UP (%)   : {float(br):.1f}%"   if pd.notna(br)    else "  Breadth UP (%)   : N/A")

    # ── SECCIÓN 6: Telemetría y TTL ───────────────────────────────────────────
    print(f"\n  {SEP}")
    print(f"  ⏱️   TELEMETRÍA")
    print(f"  {SEP}")
    lat_ticks  = tel.get("ex_latency_ms_ticks", 0)
    lat_ohlcv  = tel.get("ex_latency_ms_ohlcv_avg", 0)
    lat_ob     = tel.get("ex_latency_ms_orderbook_avg", 0)
    calls      = tel.get("calls_this_cycle", {})
    ttl_ticks  = ttl.get("ttl_ticks_s")
    ttl_ob     = ttl.get("ttl_orderbook_s")
    ttl_ohlcv  = ttl.get("ttl_ohlcv_s", {})

    print(f"  Latencia ticks   : {lat_ticks} ms")
    print(f"  Latencia OHLCV   : {lat_ohlcv} ms (avg)")
    print(f"  Latencia OB      : {lat_ob} ms (avg)")
    print(f"  Calls este ciclo : ticks={calls.get('ticks',0)}  ohlcv={calls.get('ohlcv',0)}  orderbook={calls.get('orderbook',0)}")
    if ttl_ticks is not None:
        print(f"  Edad ticks       : {ttl_ticks:.1f}s")
    if ttl_ob is not None:
        print(f"  Edad orderbook   : {ttl_ob:.1f}s")
    ohlcv_ages = {k: f"{v:.0f}s" for k, v in ttl_ohlcv.items() if v is not None}
    if ohlcv_ages:
        ages_str = "  ".join(f"{tf}={age}" for tf, age in ohlcv_ages.items())
        print(f"  Edad OHLCV       : {ages_str}")

    # ── SECCIÓN 7: Errores ────────────────────────────────────────────────────
    if errors:
        print(f"\n  {SEP}")
        print(f"  ⚠️   ERRORES EN ESTE CICLO: {len(errors)}")
        for e in errors:
            print(f"    • {e}")

    print(f"\n{SEP2}\n")


# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 11 — Ejecución directa (prueba)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    reader  = WorldReader()
    snap_id = "SNP-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = reader.read_once(
        snapshot_id=snap_id,
        loop_idx=0,
        watchlist_symbols=set(SYMBOLS),
        force_flags={"orderbook": True},
        bucket_focus=1,
        use_orderbook=None,
    )
    print_market_summary(out)
