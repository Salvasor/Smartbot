
import pandas as pd

# =========================
# 1. Cargar datos de BTC
# =========================
# Asumimos un CSV con al menos: timestamp, open, high, low, close, volume
# Ajustá el nombre del archivo y las columnas según tu realidad.

ruta_csv = "btc_ohlcv.csv"  # <-- cambia esto

df = pd.read_csv(ruta_csv)

# --- ajustar timestamp ---
# Si viene en milisegundos tipo ccxt:
# df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

# Si YA viene como fecha tipo "2025-01-01 00:00:00", usá:
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# ordenar y poner como índice
df = df.sort_values("timestamp")
df = df.set_index("timestamp")

# =========================
# 2. Función para resumir por periodo
# =========================

def resumir_por_periodo(df, rule):
    """
    df: dataframe con índice datetime y columnas: open, high, low, close, volume
    rule: '1D' para días, '1W' para semanas, etc.
    """
    # OHLC por periodo
    ohlc = df["close"].resample(rule).ohlc()
    # open/high/low/close correctos usando todas las velas
    ohlc["open"] = df["open"].resample(rule).first()
    ohlc["high"] = df["high"].resample(rule).max()
    ohlc["low"] = df["low"].resample(rule).min()
    ohlc["close"] = df["close"].resample(rule).last()

    # volumen total del periodo
    ohlc["volume"] = df["volume"].resample(rule).sum()

    # =========================
    #  CÁLCULOS EN PORCENTAJE
    # =========================

    # 1) % cambio open -> close dentro del mismo periodo
    #    (te dice cuánto se movió en ese día/semana)
    ohlc["ret_pct_open_close"] = (ohlc["close"] / ohlc["open"] - 1.0) * 100.0

    # 2) % cambio vs cierre anterior del mismo timeframe
    #    (ej: cierre de esta semana vs cierre de la semana pasada)
    ohlc["ret_pct_vs_prev_close"] = ohlc["close"].pct_change() * 100.0

    # 3) % rango interno (high vs low) del periodo
    #    (qué tan “amplio” fue el movimiento interno de ese día/semana)
    ohlc["range_pct_high_low"] = (ohlc["high"] / ohlc["low"] - 1.0) * 100.0

    # limpiar periodos vacíos
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"])

    return ohlc

# =========================
# 3. Calcular diario y semanal
# =========================

df_daily = resumir_por_periodo(df, "1D")
df_weekly = resumir_por_periodo(df, "1W")

# =========================
# 4. Guardar a Excel/CSV
# =========================

# CSV (fácil de abrir en Excel)
df_daily.to_csv("btc_daily_stats.csv", index=True)
df_weekly.to_csv("btc_weekly_stats.csv", index=True)

print("Listo maje:")
print(" - btc_daily_stats.csv  → stats diarios")
print(" - btc_weekly_stats.csv → stats semanales")
