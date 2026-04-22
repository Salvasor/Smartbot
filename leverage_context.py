# -*- coding: utf-8 -*-
"""
Módulo 2.5 — Contexto de Apalancamiento (Futuros) · leverage_context.py

Lee datos públicos de Binance Futures SIN credenciales:
  - Funding rate actual
  - Open Interest en USD
  - Long/Short ratio global de cuentas

Calcula:
  - leverage_risk: low / medium / high  → usado por M3 y policies.json
  - Señales individuales: funding_signal, oi_price_signal, ls_signal

Filosofía:
  - No bloqueante: si falla, retorna leverage_risk="low" con flag de error
  - Caché con TTL propio (futuros cambian más lento que spot)
  - Solo lectura pública, nunca escribe ni ejecuta órdenes

Requisitos:
    pip install requests pandas
"""

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — Configuración
# ──────────────────────────────────────────────────────────────────────────────

# Símbolo en formato Futures (sin slash)
FUTURES_SYMBOL: str = "BTCUSDT"

# TTL por dato (segundos)
TTL_FUNDING_S:   int = 300   # funding cambia cada 8h pero chequeamos cada 5 min
TTL_OI_S:        int = 60    # OI cambia cada minuto
TTL_LS_RATIO_S:  int = 300   # long/short ratio cada 5 min

# Umbrales funding rate
FUNDING_LONGS_ELEVATED: float =  0.0005   # +0.05%
FUNDING_LONGS_EXTREME:  float =  0.0010   # +0.10%
FUNDING_SHORTS_ELEVATED: float = -0.0005  # -0.05%
FUNDING_SHORTS_EXTREME:  float = -0.0010  # -0.10%

# Umbrales long/short ratio
LS_LONGS_EXTREME:  float = 1.8   # demasiados longs
LS_SHORTS_EXTREME: float = 0.6   # demasiados shorts

# Umbral cambio de OI para considerar movimiento significativo (%)
OI_CHANGE_SIGNIFICANT: float = 2.0

BASE_URL_FUTURES = "https://fapi.binance.com"
BASE_URL_DATA    = "https://www.binance.com"

TIMEOUT_S: int = 5

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 2 — Lecturas individuales (endpoints públicos)
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_funding_rate(symbol: str) -> Optional[Dict]:
    """
    Funding rate actual y próximo funding time.
    Endpoint: GET /fapi/v1/premiumIndex
    Público, sin credenciales.
    """
    try:
        r = requests.get(
            f"{BASE_URL_FUTURES}/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            timeout=TIMEOUT_S
        )
        r.raise_for_status()
        d = r.json()
        return {
            "funding_rate":      float(d.get("lastFundingRate", 0)),
            "mark_price":        float(d.get("markPrice", 0)),
            "index_price":       float(d.get("indexPrice", 0)),
            "next_funding_time": int(d.get("nextFundingTime", 0)),
        }
    except Exception as e:
        return {"error": str(e)}

def _fetch_open_interest(symbol: str) -> Optional[Dict]:
    """
    Open Interest actual en contratos y en USD (usando mark price).
    Endpoint: GET /fapi/v1/openInterest
    Público, sin credenciales.
    """
    try:
        r = requests.get(
            f"{BASE_URL_FUTURES}/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=TIMEOUT_S
        )
        r.raise_for_status()
        d = r.json()
        oi_contracts = float(d.get("openInterest", 0))
        return {
            "oi_contracts": oi_contracts,
            "ts":           d.get("time", 0),
        }
    except Exception as e:
        return {"error": str(e)}

def _fetch_long_short_ratio(symbol: str, period: str = "5m") -> Optional[Dict]:
    """
    Ratio global de cuentas long vs short.
    Endpoint: GET /futures/data/globalLongShortAccountRatio
    Público, sin credenciales.
    period: "5m", "15m", "1h", "4h", "1d"
    """
    try:
        r = requests.get(
            f"{BASE_URL_FUTURES}/futures/data/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": period, "limit": 2},
            timeout=TIMEOUT_S
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return {"error": "empty response"}
        latest = data[-1]
        return {
            "long_short_ratio":  float(latest.get("longShortRatio", 1.0)),
            "long_account_pct":  float(latest.get("longAccount",    0.5)) * 100,
            "short_account_pct": float(latest.get("shortAccount",   0.5)) * 100,
        }
    except Exception as e:
        return {"error": str(e)}

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 3 — Señales individuales
# ──────────────────────────────────────────────────────────────────────────────

def _funding_signal(rate: float) -> str:
    """
    Interpreta el funding rate como señal de riesgo.
    Positivo = longs pagan a shorts (mercado alcista apalancado)
    Negativo = shorts pagan a longs (mercado bajista apalancado)
    """
    if rate >= FUNDING_LONGS_EXTREME:   return "longs_extreme"    # riesgo dump cascade
    if rate >= FUNDING_LONGS_ELEVATED:  return "longs_elevated"   # precaución
    if rate <= FUNDING_SHORTS_EXTREME:  return "shorts_extreme"   # riesgo squeeze
    if rate <= FUNDING_SHORTS_ELEVATED: return "shorts_elevated"  # precaución
    return "neutral"

def _oi_price_signal(oi_now: float, oi_prev: float, price_now: float, price_prev: float) -> str:
    """
    Combina cambio de OI con cambio de precio para inferir qué está pasando.
    OI sube + precio sube  → longs_building   (tendencia con convicción)
    OI sube + precio baja  → shorts_building  (shorts acumulándose, posible squeeze)
    OI baja + precio baja  → liq_cascade      (liquidaciones activas — peligro)
    OI baja + precio sube  → shorts_covering  (shorts cerrando, rally puede agotarse)
    """
    if oi_prev <= 0 or price_prev <= 0:
        return "unknown"
    oi_chg    = (oi_now - oi_prev) / oi_prev * 100
    price_chg = (price_now - price_prev) / price_prev * 100

    if abs(oi_chg) < OI_CHANGE_SIGNIFICANT:
        return "stable"

    if oi_chg > 0 and price_chg > 0:  return "longs_building"
    if oi_chg > 0 and price_chg < 0:  return "shorts_building"
    if oi_chg < 0 and price_chg < 0:  return "liq_cascade"
    if oi_chg < 0 and price_chg > 0:  return "shorts_covering"
    return "stable"

def _ls_signal(ratio: float) -> str:
    if ratio >= LS_LONGS_EXTREME:  return "longs_extreme"   # trampa alcista posible
    if ratio <= LS_SHORTS_EXTREME: return "shorts_extreme"  # trampa bajista posible
    return "neutral"

def _compute_leverage_risk(
    funding_sig: str,
    oi_sig:      str,
    ls_sig:      str,
) -> str:
    """
    Combina las tres señales en un semáforo de riesgo de apalancamiento.

    HIGH   → condición extrema que puede generar cascade inminente
    MEDIUM → señal elevada en al menos una dimensión, operar con cautela
    LOW    → todo en zona neutral
    """
    # Condiciones HIGH (cualquiera de estas sola ya es high)
    if funding_sig == "longs_extreme" and ls_sig == "longs_extreme":
        return "high"   # longs apalancados al máximo — cascade muy probable
    if oi_sig == "liq_cascade":
        return "high"   # liquidaciones activas detectadas
    if funding_sig == "shorts_extreme" and ls_sig == "shorts_extreme":
        return "high"   # short squeeze inminente

    # Condiciones MEDIUM
    if funding_sig in ("longs_extreme", "longs_elevated", "shorts_extreme", "shorts_elevated"):
        return "medium"
    if oi_sig in ("shorts_building", "longs_building"):
        return "medium"
    if ls_sig in ("longs_extreme", "shorts_extreme"):
        return "medium"

    return "low"

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — Caché con TTL propio
# ──────────────────────────────────────────────────────────────────────────────

class LeverageCache:
    """
    Caché liviano para los tres endpoints de futuros.
    Cada dato tiene su propio TTL porque cambian a distinta velocidad.
    """
    def __init__(self):
        self._funding:  Optional[Dict] = None
        self._oi:       Optional[Dict] = None
        self._oi_prev:  Optional[Dict] = None   # para calcular cambio
        self._ls:       Optional[Dict] = None
        self._price_prev: Optional[float] = None

        self._ts_funding: float = 0.0
        self._ts_oi:      float = 0.0
        self._ts_ls:      float = 0.0

    def _stale(self, ts: float, ttl: int) -> bool:
        return (time.time() - ts) >= ttl

    def refresh(self, symbol: str, current_price: Optional[float] = None) -> Dict:
        """
        Refresca solo los datos cuyo TTL venció.
        Retorna el contexto completo calculado.
        No lanza excepciones — errores quedan en el campo 'errors'.
        """
        errors = []

        # ── Funding rate
        if self._stale(self._ts_funding, TTL_FUNDING_S):
            result = _fetch_funding_rate(symbol)
            if "error" not in result:
                self._funding = result
                self._ts_funding = time.time()
            else:
                errors.append(f"funding:{result['error']}")

        # ── Open Interest
        if self._stale(self._ts_oi, TTL_OI_S):
            result = _fetch_open_interest(symbol)
            if "error" not in result:
                self._oi_prev = self._oi   # guardamos el anterior para calcular cambio
                self._oi = result
                self._ts_oi = time.time()
            else:
                errors.append(f"oi:{result['error']}")

        # ── Long/Short ratio
        if self._stale(self._ts_ls, TTL_LS_RATIO_S):
            result = _fetch_long_short_ratio(symbol)
            if "error" not in result:
                self._ls = result
                self._ts_ls = time.time()
            else:
                errors.append(f"ls_ratio:{result['error']}")

        return self._build_context(current_price, errors)

    def _build_context(self, current_price: Optional[float], errors: list) -> Dict:
        """Arma el diccionario de contexto completo con señales y leverage_risk."""

        # ── Valores con fallback seguro
        funding_rate = self._funding.get("funding_rate", 0.0) if self._funding else 0.0
        mark_price   = self._funding.get("mark_price", current_price or 0.0) if self._funding else (current_price or 0.0)
        next_ft      = self._funding.get("next_funding_time", 0) if self._funding else 0

        oi_contracts  = self._oi.get("oi_contracts", 0.0) if self._oi else 0.0
        oi_usd        = oi_contracts * mark_price if mark_price > 0 else 0.0

        oi_prev_contracts = self._oi_prev.get("oi_contracts", oi_contracts) if self._oi_prev else oi_contracts
        oi_prev_usd       = oi_prev_contracts * mark_price if mark_price > 0 else oi_usd
        oi_change_pct     = ((oi_usd - oi_prev_usd) / oi_prev_usd * 100) if oi_prev_usd > 0 else 0.0

        ls_ratio         = self._ls.get("long_short_ratio", 1.0) if self._ls else 1.0
        long_account_pct = self._ls.get("long_account_pct", 50.0) if self._ls else 50.0
        short_account_pct= self._ls.get("short_account_pct", 50.0) if self._ls else 50.0

        price_prev = self._price_prev or mark_price
        if current_price:
            self._price_prev = current_price

        # ── Señales
        f_sig  = _funding_signal(funding_rate)
        oi_sig = _oi_price_signal(oi_usd, oi_prev_usd, mark_price, price_prev)
        ls_sig = _ls_signal(ls_ratio)

        # ── Riesgo consolidado
        lev_risk = _compute_leverage_risk(f_sig, oi_sig, ls_sig)

        # ── Funding rate anualizado (para contexto humano)
        # Funding se paga 3 veces al día → × 3 × 365
        funding_annual_pct = funding_rate * 3 * 365 * 100

        # ── Próximo funding en minutos
        next_funding_min = None
        if next_ft > 0:
            next_funding_min = max(0, int((next_ft / 1000 - time.time()) / 60))

        return {
            "ts":                  _utc_now_str(),
            "symbol":              FUTURES_SYMBOL,
            # Funding
            "funding_rate":        round(funding_rate, 6),
            "funding_rate_pct":    round(funding_rate * 100, 4),
            "funding_annual_pct":  round(funding_annual_pct, 2),
            "next_funding_min":    next_funding_min,
            "funding_signal":      f_sig,
            # Open Interest
            "oi_usd":              round(oi_usd, 0),
            "oi_change_pct":       round(oi_change_pct, 3),
            "oi_price_signal":     oi_sig,
            # Long/Short
            "long_short_ratio":    round(ls_ratio, 3),
            "long_account_pct":    round(long_account_pct, 2),
            "short_account_pct":   round(short_account_pct, 2),
            "ls_signal":           ls_sig,
            # Semáforo consolidado → lo que usa M3 y policies.json
            "leverage_risk":       lev_risk,
            # Meta
            "errors":              errors,
            "data_available":      len(errors) < 3,   # al menos un dato válido
        }

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 5 — API pública del módulo
# ──────────────────────────────────────────────────────────────────────────────

# Instancia singleton del caché (M3 la importa y reutiliza entre loops)
_cache = LeverageCache()

def read_leverage_context(current_price: Optional[float] = None) -> Dict:
    """
    Punto de entrada principal para M3.
    Llama en cada loop — el caché interno maneja los TTLs.
    Nunca lanza excepciones.

    Retorna dict con al menos:
        leverage_risk: "low" | "medium" | "high"
        funding_signal, oi_price_signal, ls_signal
        errors: lista de errores (vacía si todo OK)
    """
    return _cache.refresh(FUTURES_SYMBOL, current_price)

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 6 — Display en terminal
# ──────────────────────────────────────────────────────────────────────────────

def _risk_icon(risk: str) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")

def _sig_label(sig: str) -> str:
    labels = {
        "neutral":        "neutral",
        "longs_elevated": "longs elevados ⚠️",
        "longs_extreme":  "longs extremos 🔴",
        "shorts_elevated":"shorts elevados ⚠️",
        "shorts_extreme": "shorts extremos 🔴",
        "longs_building": "acumulando longs 📈",
        "shorts_building":"acumulando shorts 📉",
        "liq_cascade":    "liquidaciones activas 💥",
        "shorts_covering":"shorts cerrando 🔄",
        "stable":         "estable",
        "unknown":        "sin datos",
    }
    return labels.get(sig, sig)

def print_leverage_summary(ctx: Dict) -> None:
    SEP  = "─" * 70
    SEP2 = "═" * 70
    risk = ctx.get("leverage_risk", "low")

    print(f"\n{SEP2}")
    print(f"  {_risk_icon(risk)}  CONTEXTO DE APALANCAMIENTO  ·  {ctx.get('ts','?')}  ·  riesgo: {risk.upper()}")
    print(SEP2)

    print(f"\n  {'FUNDING RATE':<30}")
    print(f"  {'Tasa actual':<28} : {ctx.get('funding_rate_pct', 0):>+8.4f}%")
    print(f"  {'Tasa anualizada':<28} : {ctx.get('funding_annual_pct', 0):>+8.2f}%")
    nfm = ctx.get("next_funding_min")
    print(f"  {'Próximo funding':<28} : {'en ' + str(nfm) + ' min' if nfm is not None else 'N/A'}")
    print(f"  {'Señal':<28} : {_sig_label(ctx.get('funding_signal','?'))}")

    print(f"\n  {'OPEN INTEREST':<30}")
    oi = ctx.get("oi_usd", 0)
    print(f"  {'OI total (USD)':<28} : ${oi:>15,.0f}")
    print(f"  {'Cambio vs anterior':<28} : {ctx.get('oi_change_pct', 0):>+8.3f}%")
    print(f"  {'Señal':<28} : {_sig_label(ctx.get('oi_price_signal','?'))}")

    print(f"\n  {'LONG / SHORT RATIO':<30}")
    print(f"  {'Ratio L/S':<28} : {ctx.get('long_short_ratio', 0):>8.3f}")
    print(f"  {'Longs':<28} : {ctx.get('long_account_pct', 0):>8.2f}%")
    print(f"  {'Shorts':<28} : {ctx.get('short_account_pct', 0):>8.2f}%")
    print(f"  {'Señal':<28} : {_sig_label(ctx.get('ls_signal','?'))}")

    errors = ctx.get("errors", [])
    if errors:
        print(f"\n  {SEP}")
        print(f"  ⚠️  Errores: {len(errors)}")
        for e in errors:
            print(f"    • {e}")

    print(f"\n{SEP2}\n")

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 7 — Ejecución directa
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Leyendo contexto de apalancamiento desde Binance Futures...")
    ctx = read_leverage_context()
    print_leverage_summary(ctx)
