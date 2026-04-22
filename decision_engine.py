"""
decision_engine.py — Bot BTC V2
Sistema híbrido C+D: Clasificador de Régimen + Score de Calidad + Auditor

Arquitectura:
  CAPA 0 — Guardián:    4 reglas duras irrompibles
  CAPA 1 — Régimen:     Dirección + Convicción → régimen del mercado
  CAPA 2 — Score:       Calidad de la entrada/salida
  CAPA 3 — Decisión:    Acción + sizing
  CAPA 4 — Auditor:     Validación final + hook BiLSTM

Autor: Salvador + Claude
"""

from __future__ import annotations
import json
import logging
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Deque, Optional, Tuple

log = logging.getLogger("M3")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
_POLICY_PATH = Path(__file__).parent / "policies.json"
_LOG_DIR     = Path(__file__).parent / "logs"
DRY_RUN      = False  # se sobreescribe desde loop_principal

def _load_policies() -> Dict:
    with open(_POLICY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

POLICIES: Dict = _load_policies()
log.info(f"policies.json cargado · versión {POLICIES['meta']['version']}")


def reload_policies():
    global POLICIES
    POLICIES = _load_policies()
    log.info(f"Policies recargadas · versión {POLICIES['meta']['version']}")


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _new_decision_id() -> str:
    return f"DEC-{uuid.uuid4().hex[:10].upper()}"


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS DE CONTEXTO
# ──────────────────────────────────────────────────────────────────────────────

def _get(ctx: Dict, key: str, default=None):
    """Obtiene variable del contexto con fallback seguro."""
    v = ctx.get(key)
    return default if v is None else v


def _calc_lotes_en_perdida(df_inventory) -> int:
    """Cuántos lotes OPEN tienen PnL negativo."""
    if df_inventory is None or df_inventory.empty:
        return 0
    try:
        open_lots = df_inventory[df_inventory["status"].astype(str) == "OPEN"]
        count = 0
        for _, lot in open_lots.iterrows():
            pnl = lot.get("pnl_pct_actual", None)
            if pnl is not None:
                try:
                    if float(pnl) < 0:
                        count += 1
                except:
                    pass
        return count
    except:
        return 0


def _calc_pnl_dia(df_ventas_hoy) -> float:
    """PnL acumulado del día en % promedio."""
    if df_ventas_hoy is None or df_ventas_hoy.empty:
        return 0.0
    try:
        pnls = []
        for _, row in df_ventas_hoy.iterrows():
            p = row.get("pnl_pct", 0)
            try:
                pnls.append(float(p))
            except:
                pass
        return round(sum(pnls), 4) if pnls else 0.0
    except:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# CAPA 0 — GUARDIÁN
# ──────────────────────────────────────────────────────────────────────────────

def _guardian(ctx: Dict) -> Tuple[bool, str, str]:
    """
    Evalúa las 4 reglas duras irrompibles.
    Retorna: (pasa, efecto, razón)
    """
    g = POLICIES.get("guardian", {})

    usdt_pct    = _get(ctx, "usdt_reserve_pct", 100.0)
    btc_disp    = _get(ctx, "btc_disponible", 1.0)
    lev_risk    = _get(ctx, "leverage_risk", "low")
    pnl_dia     = _get(ctx, "pnl_dia_pct", 0.0)

    usdt_min    = g.get("usdt_min_pct", 15.0)
    pnl_min     = g.get("pnl_dia_min_pct", -5.0)
    lev_block   = g.get("leverage_block", "high")

    # G1 — Capital mínimo
    if usdt_pct < usdt_min:
        return False, "BLOCK_BUY", f"G1: USDT={usdt_pct:.1f}% < mínimo={usdt_min}%"

    # G2 — BTC disponible (solo afecta ventas, se maneja en executor)
    # G3 — Riesgo sistémico
    if lev_risk == lev_block:
        return False, "BLOCK_BUY", f"G3: leverage_risk={lev_risk} → riesgo sistémico"

    # G4 — Stop del día
    if pnl_dia <= pnl_min:
        return False, "BLOCK_ALL", f"G4: pnl_dia={pnl_dia:.2f}% <= stop={pnl_min}%"

    return True, "PASS", ""


# ──────────────────────────────────────────────────────────────────────────────
# CAPA 1 — CLASIFICADOR DE RÉGIMEN
# ──────────────────────────────────────────────────────────────────────────────

def _calc_direccion(ctx: Dict) -> float:
    """Calcula la dirección del mercado: -100 a +100."""
    w   = POLICIES["regimen"]["direccion_weights"]
    dir = 0.0

    # Tendencias
    for tf_key, tf_name in [("trend_1d", "trend_1d"), ("trend_4h", "trend_4h"), ("trend_1h", "trend_1h")]:
        val = _get(ctx, tf_key, "flat-weak")
        dir += w.get(tf_name, {}).get(val, 0)

    # Week trend
    wt  = _get(ctx, "week_trend_pct", 0.0)
    ww  = w.get("week_trend_pct", {})
    th  = ww.get("thresholds", {"strong": 2.0, "mild": 0.0})
    if wt >= th["strong"]:    dir += ww.get("strong_up", 7)
    elif wt >= th["mild"]:    dir += ww.get("mild_up", 3)
    elif wt >= -th["strong"]: dir += ww.get("mild_down", -3)
    else:                     dir += ww.get("strong_down", -7)

    # Funding
    fr  = _get(ctx, "funding_rate", 0.0)
    fw  = w.get("funding_rate", {})
    fth = fw.get("thresholds", {"strong_bull": 0.005, "mild_bull": 0.0, "mild_bear": -0.003})
    if fr >= fth["strong_bull"]:   dir += fw.get("strong_bull", 10)
    elif fr >= fth["mild_bull"]:   dir += fw.get("mild_bull", 4)
    elif fr >= fth["mild_bear"]:   dir += fw.get("mild_bear", -4)
    else:                          dir += fw.get("strong_bear", -10)

    # Imbalance
    imb = _get(ctx, "imb_zone1", 0.0)
    iw  = w.get("imb_zone1", {})
    ith = iw.get("thresholds", {"strong_buy": 0.5, "mild_buy": 0.2, "mild_sell": -0.2, "strong_sell": -0.5})
    if imb >= ith["strong_buy"]:   dir += iw.get("strong_buy", 10)
    elif imb >= ith["mild_buy"]:   dir += iw.get("mild_buy", 5)
    elif imb >= ith["mild_sell"]:  dir += iw.get("neutral", 0)
    elif imb >= ith["strong_sell"]:dir += iw.get("mild_sell", -5)
    else:                          dir += iw.get("strong_sell", -10)

    # Leverage risk
    lev = _get(ctx, "leverage_risk", "low")
    dir += w.get("leverage_risk", {}).get(lev, 0)

    # Posición semanal
    pw  = _get(ctx, "price_position_1w", 0.5)
    pww = w.get("pos_1w", {})
    pth = pww.get("thresholds", {"fondo": 0.25, "bajo": 0.50, "alto": 0.75})
    if pw <= pth["fondo"]:  dir += pww.get("fondo", 8)
    elif pw <= pth["bajo"]: dir += pww.get("bajo", 3)
    elif pw <= pth["alto"]: dir += pww.get("alto", -3)
    else:                   dir += pww.get("techo", -8)

    # Downside semanal
    pos_1w       = _get(ctx, "price_position_1w", 0.5)
    weekly_range = _get(ctx, "weekly_range_pct", 0.0)
    downside_1w  = pos_1w * weekly_range
    dw           = w.get("downside_1w_pct", {})
    dth          = dw.get("thresholds", {"poco": 2.0, "medio": 4.0})
    if downside_1w <= dth["poco"]:  dir += dw.get("poco", 5)
    elif downside_1w <= dth["medio"]:dir += dw.get("medio", -3)
    else:                            dir += dw.get("mucho", -7)

    # Soporte/resistencia
    # near_support solo es válido en zona baja (pos_1h<0.50 Y pos_1w<0.70).
    # En rangos altos "near_support" solo significa "mínimo de 1 vela" — no soporte real.
    _ns_pos1h = float(_get(ctx, "price_position_1h") or 0.5)
    _ns_pos1w = float(_get(ctx, "price_position_1w") or 0.5)
    _ns_valid  = _ns_pos1h < 0.50 and _ns_pos1w < 0.70
    if _get(ctx, "near_support", False) and _ns_valid:
        dir += w.get("near_support", {}).get("True", 5)
    if _get(ctx, "near_resistance", False): dir += w.get("near_resistance", {}).get("True", -5)

    return round(max(-100.0, min(100.0, dir)), 1)


def _calc_conviccion(ctx: Dict) -> float:
    """Calcula la convicción del mercado: 0 a 100."""
    w   = POLICIES["regimen"]["conviccion_weights"]
    con = 0.0

    # Alineación de timeframes
    trends = [_get(ctx, f"trend_{tf}", "flat-weak") for tf in ["1d", "4h", "1h"]]
    n_up   = trends.count("up")
    n_down = trends.count("down")
    aw     = w.get("alineacion_tf", {})
    if n_up == 3 or n_down == 3:       con += aw.get("3_alineados", 15)
    elif n_up == 2 or n_down == 2:     con += aw.get("2_alineados", 8)
    elif n_up == 1 or n_down == 1:     con += aw.get("1_alineado", 0)
    else:                              con += aw.get("contradiccion", -8)

    # Volatilidad
    vol = _get(ctx, "vol_state", "normal")
    con += w.get("vol_state", {}).get(vol, 0)

    # Velocidad
    spd = _get(ctx, "speed_state", "low")
    con += w.get("speed_state", {}).get(spd, 0)

    # Volumen confirma
    vc  = _get(ctx, "vol_confirms_trend", False)
    con += w.get("vol_confirms_trend", {}).get(str(vc), 0)

    # Funding absoluto
    fr     = abs(_get(ctx, "funding_rate", 0.0))
    fw     = w.get("funding_abs", {})
    fth    = fw.get("thresholds", {"fuerte": 0.005, "medio": 0.002})
    if fr >= fth["fuerte"]:  con += fw.get("fuerte", 10)
    elif fr >= fth["medio"]: con += fw.get("medio", 5)
    else:                    con += fw.get("debil", 0)

    # Imbalance absoluto
    imb    = abs(_get(ctx, "imb_zone1", 0.0))
    iw     = w.get("imb_abs", {})
    ith    = iw.get("thresholds", {"fuerte": 0.5, "medio": 0.2})
    if imb >= ith["fuerte"]:  con += iw.get("fuerte", 12)
    elif imb >= ith["medio"]: con += iw.get("medio", 6)
    else:                     con += iw.get("debil", 0)

    # Leverage
    lev = _get(ctx, "leverage_risk", "low")
    con += w.get("leverage_risk", {}).get(lev, 0)

    # Near support o resistance
    # near_support solo suma convicción si estamos en zona baja (pos_1h<0.50 Y pos_1w<0.70).
    # near_resistance siempre aplica (precio en techo = señal válida en cualquier posición).
    _nsr_pos1h = float(_get(ctx, "price_position_1h") or 0.5)
    _nsr_pos1w = float(_get(ctx, "price_position_1w") or 0.5)
    _nsr_valid  = _nsr_pos1h < 0.50 and _nsr_pos1w < 0.70
    if (_get(ctx, "near_support", False) and _nsr_valid) or _get(ctx, "near_resistance", False):
        con += w.get("near_support_or_resistance", {}).get("True", 6)

    return round(max(0.0, min(100.0, con)), 1)


def _clasificar_regimen(direccion: float, conviccion: float) -> str:
    """Clasifica el régimen del mercado según dirección y convicción."""
    d, c = direccion, conviccion

    if d >= 50:
        return "ALCISTA_FUERTE" if c >= 40 else "ALCISTA_DUDOSO"
    elif d >= 20:
        if c >= 40: return "ALCISTA_MODERADO"
        else:       return "CORRECCION"
    elif d >= -20:
        return "LATERAL_DEFINIDO" if c >= 35 else "LATERAL_DEBIL"
    elif d >= -50:
        if c >= 40: return "BAJISTA_MODERADO"
        else:       return "REBOTE_BAJISTA"
    else:
        return "BAJISTA_FUERTE" if c >= 40 else "BAJISTA_DUDOSO"


def _get_regimen_config(regimen: str) -> Dict:
    """Obtiene configuración del régimen."""
    return POLICIES["regimen"]["clasificacion"].get(regimen, {
        "max_lotes": 1, "umbral_compra": 100
    })


# ──────────────────────────────────────────────────────────────────────────────
# CAPA 2 — SCORES DE CALIDAD
# ──────────────────────────────────────────────────────────────────────────────

def _score_compra(ctx: Dict) -> float:
    """Score de calidad de entrada: 0-100."""
    s  = POLICIES["scores"]["compra"]
    sc = 0.0

    # pos_1m
    p1m = _get(ctx, "price_position_1m")
    if p1m is not None:
        sp = s.get("pos_1m", {})
        if p1m <= sp.get("fondo_extremo", {}).get("max", 0.03):   sc += sp["fondo_extremo"]["pts"]
        elif p1m <= sp.get("fondo", {}).get("max", 0.08):         sc += sp["fondo"]["pts"]
        elif p1m <= sp.get("zona_baja", {}).get("max", 0.20):     sc += sp["zona_baja"]["pts"]

    # pos_1h — curva completa: premia fondo, penaliza techo
    p1h = _get(ctx, "price_position_1h")
    if p1h is not None:
        sp = s.get("pos_1h", {})
        if p1h <= sp.get("fondo_extremo", {}).get("max", 0.05):   sc += sp["fondo_extremo"]["pts"]
        elif p1h <= sp.get("fondo", {}).get("max", 0.15):         sc += sp["fondo"]["pts"]
        elif p1h <= sp.get("zona_baja", {}).get("max", 0.30):     sc += sp["zona_baja"]["pts"]
        elif p1h < sp.get("zona_alta", {}).get("min", 0.70):      sc += sp.get("neutral", {}).get("pts", 0)
        elif p1h < sp.get("techo", {}).get("min", 0.85):          sc += sp.get("zona_alta", {}).get("pts", -8)
        else:                                                       sc += sp.get("techo", {}).get("pts", -18)

    # downside_1h_pct — también penaliza cuando hay MUCHO espacio para caer
    d1h = _get(ctx, "downside_1h_pct")
    if d1h is not None:
        sp  = s.get("downside_1h_pct", {})
        if d1h <= sp.get("muy_poco", {}).get("max", 0.3):         sc += sp["muy_poco"]["pts"]
        elif d1h <= sp.get("poco", {}).get("max", 0.6):           sc += sp["poco"]["pts"]
        elif d1h < sp.get("demasiado", {}).get("min", 2.0):       sc += sp.get("mucho", {}).get("pts", 0)
        else:                                                       sc += sp.get("demasiado", {}).get("pts", -12)

    # near_support — solo aplica el bonus si estamos en zona baja real
    # (pos_1h < 0.50 Y pos_1w < 0.70). Evita que un mínimo de 1-vela en rango alto
    # se confunda con soporte estructural y empuje el score a comprar en techo.
    ns       = _get(ctx, "near_support", False)
    _sc_p1h  = float(p1h or 0.5)
    _sc_p1w  = float(_get(ctx, "price_position_1w") or 0.5)
    _sc_ns_valid = _sc_p1h < 0.50 and _sc_p1w < 0.70
    if ns and not _sc_ns_valid:
        ns = False  # ignorar near_support en zona alta
    sc += s.get("near_support", {}).get(str(ns), 0)

    # vol_confirms_trend
    vc    = _get(ctx, "vol_confirms_trend", False)
    trend = _get(ctx, "trend_1m", "flat-weak")
    if vc:
        if trend == "up":   sc += s.get("vol_confirms_trend", {}).get("confirma_subida", 10)
        elif trend == "down": sc += s.get("vol_confirms_trend", {}).get("confirma_bajada", 0)
    else:
        sc += s.get("vol_confirms_trend", {}).get("no_confirma", 5)

    # imb_zone1
    imb = _get(ctx, "imb_zone1", 0.0)
    iw  = s.get("imb_zone1", {})
    ith = iw.get("thresholds", {"compradores": 0.3, "vendedores": -0.3})
    if imb >= ith["compradores"]:     sc += iw.get("compradores", 10)
    elif imb >= ith["vendedores"]:    sc += iw.get("neutro", 5)
    else:                             sc += iw.get("vendedores", 0)

    # funding puntual
    fr  = _get(ctx, "funding_rate", 0.0)
    fw  = s.get("funding_puntual", {})
    thr = fw.get("threshold", 0.0)
    sc += fw.get("positivo", 10) if fr >= thr else fw.get("negativo", 0)

    return round(max(0.0, min(100.0, sc)), 1)


def _score_venta_tp(ctx: Dict, lot: Dict, strategy: str = "S1") -> float:
    """Score de calidad para tomar ganancia: 0-100.

    strategy="S1": salida rápida. Scoring graduado desde 0.20% PnL.
                   El score base es bajo → solo dispara si el mercado lo confirma.
    strategy="S2": salida paciente. Mínimo 1.0% igual que antes.
    """
    s  = POLICIES["scores"]["venta_tp"]
    sc = 0.0

    # PnL del lote
    pnl = float(lot.get("pnl_pct_actual", 0.0) or 0.0)
    sp  = s.get("pnl_lote_pct", {})

    if strategy == "S1":
        # Fuzzy TP: puntuación graduada desde 0.20% (micro-movimiento mínimo)
        # El score base solo llega al umbral si las condiciones de mercado lo confirman.
        if   pnl >= sp.get("grande", {}).get("min", 2.0):    sc += sp["grande"]["pts"]
        elif pnl >= sp.get("bueno", {}).get("min", 1.5):     sc += sp["bueno"]["pts"]
        elif pnl >= sp.get("minimo", {}).get("min", 1.0):    sc += sp["minimo"]["pts"]
        elif pnl >= 0.50:  sc += 28   # zona media: necesita 2+ señales de mercado
        elif pnl >= 0.30:  sc += 18   # zona baja: necesita 3+ señales de mercado
        elif pnl >= 0.20:  sc += 10   # mínimo viable: requiere casi todas las señales
        else: return 0.0   # menos de 0.20% → demasiado ruido
    else:
        # S2 — salida paciente: mínimo 1.0% como antes
        if pnl >= sp.get("grande", {}).get("min", 2.0):    sc += sp["grande"]["pts"]
        elif pnl >= sp.get("bueno", {}).get("min", 1.5):   sc += sp["bueno"]["pts"]
        elif pnl >= sp.get("minimo", {}).get("min", 1.0):  sc += sp["minimo"]["pts"]
        else: return 0.0  # no llegó al mínimo para TP

    # pos_1m en techo
    p1m = _get(ctx, "price_position_1m")
    if p1m is not None:
        pm = s.get("pos_1m", {})
        if p1m >= pm.get("techo", {}).get("min", 0.85):      sc += pm["techo"]["pts"]
        elif p1m >= pm.get("casi_techo", {}).get("min", 0.70): sc += pm["casi_techo"]["pts"]

    # near_resistance
    nr = _get(ctx, "near_resistance", False)
    sc += s.get("near_resistance", {}).get(str(nr), 0)

    # imbalance — vendedores llegando
    imb = _get(ctx, "imb_zone1", 0.0)
    iw  = s.get("imb_zone1", {})
    thr = iw.get("vendedores_llegando", -0.3)
    if imb <= thr:      sc += iw.get("pts_vendedores", 15)
    elif imb >= 0.3:    sc += iw.get("compradores", 0)
    else:               sc += iw.get("neutro", 5)

    # chaos subiendo
    vol   = _get(ctx, "vol_state", "normal")
    trend = _get(ctx, "trend_1m", "flat-weak")
    if vol == "chaos" and trend == "up":
        sc += s.get("vol_chaos_subiendo", {}).get("True", 5)

    return round(max(0.0, min(100.0, sc)), 1)


def _score_venta_sl(ctx: Dict, lot: Dict, regimen: str) -> float:
    """Score para cortar pérdida: 0-100."""
    s  = POLICIES["scores"]["venta_sl"]
    sc = 0.0

    # PnL del lote
    pnl = float(lot.get("pnl_pct_actual", 0.0) or 0.0)
    sp  = s.get("pnl_lote_pct", {})
    if pnl <= sp.get("grave", {}).get("max", -2.0):    sc += sp["grave"]["pts"]
    elif pnl <= sp.get("serio", {}).get("max", -1.5):  sc += sp["serio"]["pts"]
    elif pnl <= sp.get("leve", {}).get("max", -1.0):   sc += sp["leve"]["pts"]
    else: return 0.0  # no hay pérdida suficiente

    # Régimen bajista fuerte
    if regimen in ("BAJISTA_FUERTE", "BAJISTA_MODERADO"):
        sc += s.get("regimen_bajista_fuerte", {}).get("True", 20)

    # Chaos bajando
    vol   = _get(ctx, "vol_state", "normal")
    trend = _get(ctx, "trend_1m", "flat-weak")
    if vol == "chaos" and trend == "down":
        sc += s.get("vol_chaos_bajando", {}).get("True", 15)

    # Funding muy negativo
    fr  = _get(ctx, "funding_rate", 0.0)
    fthr= s.get("funding_muy_negativo", {}).get("threshold", -0.005)
    if fr <= fthr:
        sc += s.get("funding_muy_negativo", {}).get("pts", 10)

    # Leverage high
    lev = _get(ctx, "leverage_risk", "low")
    if lev == "high":
        sc += s.get("leverage_high", {}).get("True", 15)

    return round(max(0.0, min(100.0, sc)), 1)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — TP POR CAMBIO DE RÉGIMEN
# ──────────────────────────────────────────────────────────────────────────────

def _calc_regime_tp_threshold(prev_regimen: str, curr_regimen: str) -> Optional[float]:
    """
    Calcula el umbral de PnL% para ejecutar TP cuando el régimen mejora.
    Retorna None si el régimen no mejoró (TP no aplica).
    El umbral escala con la magnitud del salto: salto grande → más margen para correr.
    """
    rc = POLICIES.get("regime_change_tp", {})
    if not rc.get("enabled", True):
        return None

    hierarchy  = rc.get("hierarchy", {})
    thresholds = rc.get("thresholds_por_salto", {})

    prev_rank = hierarchy.get(prev_regimen)
    curr_rank = hierarchy.get(curr_regimen)

    if prev_rank is None or curr_rank is None:
        return None

    jump = curr_rank - prev_rank
    if jump <= 0:
        return None  # régimen igual o empeoró — no aplica TP

    key = "6_plus" if jump >= 6 else str(jump)
    return float(thresholds.get(key, thresholds.get("1", 0.25)))


# ──────────────────────────────────────────────────────────────────────────────
# CAPA 3 — DECISIÓN + SIZING
# ──────────────────────────────────────────────────────────────────────────────

def _get_sizing_compra(regimen: str, score: float, usdt_disponible: float, min_trade: float) -> float:
    """Calcula el monto en USDT a comprar.

    Versión flat: 40% del capital disponible para cualquier régimen y estrategia.
    Racional: el sizing por régimen era inverso al riesgo real — apostaba MÁS en
    ALCISTA_FUERTE (precio alto) y MENOS en CORRECCION (precio bajo con soporte real).
    Con 40% flat más los multiplicadores del Auditor (lotes en pérdida, pnl_dia bajo),
    el sizing efectivo ya baja automáticamente cuando las condiciones empeoran.
    """
    pct   = 0.40  # flat 40% — ajustable vía policies["sizing"]["flat_pct"] en el futuro
    monto = round(usdt_disponible * pct, 2)
    return monto if monto >= min_trade else 0.0


def _get_sell_pct_tp(regimen: str) -> float:
    """% del lote a vender en TP según régimen."""
    tp_table = POLICIES.get("sizing", {}).get("tp_sizing", {})
    return tp_table.get(regimen, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# CAPA 4 — AUDITOR
# ──────────────────────────────────────────────────────────────────────────────

def _auditor(
    accion: str,
    regimen: str,
    score_compra: float,
    umbral_compra: float,
    sizing_usdt: float,
    ctx: Dict,
    last_buy_ctx: Dict,
    n_lotes: int,
    max_lotes: int,
    lotes_en_perdida: int,
    pnl_dia: float,
) -> Tuple[str, float, str]:
    """
    Valida y calibra la decisión.
    Retorna: (veredicto, sizing_final, razon)
    
    Veredictos: APROBADO | APROBADO_REDUCIDO | VETADO
    
    [Hook BiLSTM]: cuando enabled=True, el modelo ajusta sizing aquí.
    """
    a   = POLICIES.get("auditor", {})
    mul = a.get("multiplicadores", {})

    if accion != "BUY":
        return "APROBADO", sizing_usdt, ""

    # ── VETOS ────────────────────────────────────────────────────────────────
    # 1. Lotes al máximo del régimen
    if n_lotes >= max_lotes:
        return "VETADO", 0.0, f"n_lotes={n_lotes} >= max_régimen={max_lotes}"

    # 2. Stop del día
    if pnl_dia <= POLICIES["guardian"]["pnl_dia_min_pct"]:
        return "VETADO", 0.0, f"pnl_dia={pnl_dia:.2f}% — stop del día"

    # ── CAMBIO MÍNIMO ─────────────────────────────────────────────────────────
    if last_buy_ctx:
        cm        = a.get("cambio_minimo", {})
        cambio_ok = False

        # ¿Cambió el régimen?
        if last_buy_ctx.get("regimen") != regimen:
            cambio_ok = True

        # ¿Cambió pos_1h suficiente?
        delta_pos = abs(
            (_get(ctx, "price_position_1h") or 0.5) -
            (last_buy_ctx.get("pos_1h") or 0.5)
        )
        if delta_pos >= cm.get("pos_1h_delta_min", 0.15):
            cambio_ok = True

        # ¿Cambió el score suficiente?
        delta_score = abs(score_compra - (last_buy_ctx.get("score_compra") or 0))
        if delta_score >= cm.get("score_delta_min", 15):
            cambio_ok = True

        if not cambio_ok:
            return "VETADO", 0.0, "cambio_minimo: mismo contexto desde última compra"

    # ── MULTIPLICADORES ────────────────────────────────────────────────────────
    sizing_final = sizing_usdt
    razones = []

    if lotes_en_perdida >= 2:
        m = mul.get("lotes_en_perdida_2_plus", 0.5)
        sizing_final *= m
        razones.append(f"lotes_en_perdida={lotes_en_perdida} → ×{m}")

    if pnl_dia < -2.0:
        m = mul.get("pnl_dia_lt_menos2", 0.7)
        sizing_final *= m
        razones.append(f"pnl_dia={pnl_dia:.1f}% → ×{m}")

    if score_compra < umbral_compra * 0.9:
        m = mul.get("score_lt_umbral_90pct", 0.6)
        sizing_final *= m
        razones.append(f"score bajo → ×{m}")

    # Sizing mínimo de emergencia si pnl_dia muy bajo
    if pnl_dia < -4.0:
        sizing_final = min(sizing_final, POLICIES["sizing"].get("min_trade_usdt", 10.0) * 0.8)
        razones.append("near_stop_day → sizing mínimo")

    sizing_final = round(sizing_final, 2)
    min_trade    = POLICIES["sizing"].get("min_trade_usdt", 10.0)

    if sizing_final < min_trade:
        return "VETADO", 0.0, f"sizing={sizing_final} < mínimo={min_trade}"

    # ── HOOK BILSTM ────────────────────────────────────────────────────────────
    bilstm = a.get("bilstm_hook", {})
    if bilstm.get("enabled", False):
        # En el futuro: multiplicador del modelo
        # ml_mult = bilstm_model.predict(ctx, regimen, score_compra)
        # sizing_final *= ml_mult
        pass

    veredicto = "APROBADO_REDUCIDO" if razones else "APROBADO"
    return veredicto, sizing_final, " | ".join(razones)


# ──────────────────────────────────────────────────────────────────────────────
# LOGGER DE DECISIONES
# ──────────────────────────────────────────────────────────────────────────────

def _log_decision(decision: Dict) -> None:
    """Escribe la decisión en el .jsonl mensual."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        month    = datetime.now().strftime("%Y%m")
        log_path = _LOG_DIR / f"decisions_{month}.jsonl"
        entry    = {**decision, "event": "DECISION"}
        with open(log_path, "a", encoding="utf-8") as f:
            import json as _json
            f.write(_json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        log.warning(f"No se pudo escribir log de decisión: {e}")


def _make_decision(
    action: str,
    rule_id: str,
    reason: str,
    ctx: Dict,
    snapshot_id: str,
    regimen: str = "",
    direccion: float = 0.0,
    conviccion: float = 0.0,
    score_compra: float = 0.0,
    score_venta: float = 0.0,
    signal_strength: str = "",
    qty_usdt: float = 0.0,
    lot_id: str = "",
    sell_pct: float = 1.0,
    veredicto_auditor: str = "",
    strategy: str = "S1",
) -> Dict:
    """Construye y loguea el dict de decisión."""
    dec_id = _new_decision_id()
    now    = _utc_now_str()

    d = {
        "decision_id":       dec_id,
        "snapshot_id":       snapshot_id,
        "ts":                now,
        "action":            action,
        "rule_id":           rule_id,
        "reason":            reason,
        # Contexto del régimen — para BiLSTM training data
        "regimen":           regimen,
        "direccion":         round(direccion, 1),
        "conviccion":        round(conviccion, 1),
        "score_compra":      round(score_compra, 1),
        "score_venta":       round(score_venta, 1),
        "signal_strength":   signal_strength,
        "veredicto_auditor": veredicto_auditor,
        "strategy":          strategy,
        # Acción específica
        "qty_usdt":          round(qty_usdt, 2),
        "lot_id":            lot_id,
        "sell_pct":          sell_pct,
        # Variables de mercado clave (para análisis posterior)
        "precio":            _get(ctx, "last_price", 0),
        "trend_1m":          _get(ctx, "trend_1m"),
        "trend_1h":          _get(ctx, "trend_1h"),
        "trend_1d":          _get(ctx, "trend_1d"),
        "vol_state":         _get(ctx, "vol_state"),
        "pos_1m":            _get(ctx, "price_position_1m"),
        "pos_1h":            _get(ctx, "price_position_1h"),
        "pos_1w":            _get(ctx, "price_position_1w"),
        "near_support":      _get(ctx, "near_support"),
        "near_resistance":   _get(ctx, "near_resistance"),
        "funding_rate":      _get(ctx, "funding_rate"),
        "funding_signal":    _get(ctx, "funding_signal"),
        "oi_price_signal":   _get(ctx, "oi_price_signal"),
        "oi_change_pct":     _get(ctx, "oi_change_pct"),
        "ls_signal":         _get(ctx, "ls_signal"),
        "long_short_ratio":  _get(ctx, "long_short_ratio"),
        "rsi_14_5m":         _get(ctx, "rsi_14_5m"),
        "rsi_14_15m":        _get(ctx, "rsi_14_15m"),
        "imb_zone1":         _get(ctx, "imb_zone1"),
        "leverage_risk":     _get(ctx, "leverage_risk"),
        "inv_state":         _get(ctx, "inv_state"),
        "usdt_reserve_pct":  _get(ctx, "usdt_reserve_pct"),
        "n_lotes":           _get(ctx, "n_lotes_open"),
        "downside_1h_pct":   _get(ctx, "downside_1h_pct"),
        "downside_1d_pct":   _get(ctx, "downside_1d_pct"),
        "pnl_dia_pct":       _get(ctx, "pnl_dia_pct"),
        "dry_run":           DRY_RUN,
    }

    _log_decision(d)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# PRINT DE DECISIÓN
# ──────────────────────────────────────────────────────────────────────────────

def print_decision(decision: Dict, df_inventory=None) -> None:
    """Imprime la decisión en terminal de forma legible."""
    SEP  = "═" * 72
    act  = decision.get("action", "WAIT")
    rid  = decision.get("rule_id", "?")
    ts   = decision.get("ts", "")
    reg  = decision.get("regimen", "")
    ddir = decision.get("direccion", 0)
    conv = decision.get("conviccion", 0)
    sc   = decision.get("score_compra", 0)
    sv   = decision.get("score_venta", 0)
    ver  = decision.get("veredicto_auditor", "")

    # Umbral compra dinámico según régimen (para mostrarlo en pantalla)
    try:
        reg_cfg    = POLICIES.get("regimen", {}).get("clasificacion", {}).get(reg, {})
        umbral_cmp = int(reg_cfg.get("umbral_compra", "?")) if isinstance(reg_cfg, dict) else "?"
        max_lotes  = reg_cfg.get("max_lotes", "?") if isinstance(reg_cfg, dict) else "?"
    except Exception:
        umbral_cmp = "?"
        max_lotes  = "?"

    # Umbral venta TP dinámico según régimen
    try:
        _tp_umbrales = POLICIES.get("scores", {}).get("venta_tp", {}).get("umbral_por_regimen", {})
        umbral_venta = _tp_umbrales.get(reg, _tp_umbrales.get("_default", 35))
    except Exception:
        umbral_venta = 35

    icons = {
        "BUY":    "🟢 BUY",
        "SELL_V1":"💰 SELL V1",
        "SELL_V2":"🔴 SELL V2",
        "WAIT":   "⏸️  WAIT",
    }
    icon = icons.get(act, f"⚙️  {act}")

    print(f"\n{SEP}")
    print(f"  {icon}  ·  {rid}  ·  {ts}")
    print(f"{SEP}")
    print(f"  Razón         : {decision.get('reason', '')}")
    print(f"  Régimen       : {reg}  Dir={ddir:+.0f}  Conv={conv:.0f}  max_lotes={max_lotes}")
    print(f"  Score         : compra={sc:.0f}/{umbral_cmp}  venta={sv:.0f}/{umbral_venta}  auditor={ver}")

    ctx_keys = [
        ("trend",        f"1m={decision.get('trend_1m','?'):<14} 1h={decision.get('trend_1h','?'):<14} 1d={decision.get('trend_1d','?')}"),
        ("vol/speed",    f"vol={decision.get('vol_state','?')}"),
        ("precio",       f"{decision.get('precio',0):,.2f}  pos_1m={decision.get('pos_1m','?')}  pos_1h={decision.get('pos_1h','?')}"),
        ("semanal",      f"pos_1w={decision.get('pos_1w','?')}"),
        ("espacio",      f"down_1h={decision.get('downside_1h_pct','?')}%  down_1d={decision.get('downside_1d_pct','?')}%"),
        ("soporte/res",  f"near_support={decision.get('near_support','?')}  near_resistance={decision.get('near_resistance','?')}"),
        ("inventario",   f"inv={decision.get('inv_state','?')}  lotes={decision.get('n_lotes','?')}  USDT_res={decision.get('usdt_reserve_pct','?')}%"),
        ("apalancamiento", (
            f"riesgo={decision.get('leverage_risk','?')}  "
            f"funding={decision.get('funding_rate',0):+.5f}  [{decision.get('funding_signal','?')}]"
        )),
        ("futuros/OI",   (
            f"oi_signal={decision.get('oi_price_signal','?')}  "
            f"oi_chg={decision.get('oi_change_pct',0.0):+.3f}%  "
            f"L/S={decision.get('long_short_ratio',0.0):.3f}  [{decision.get('ls_signal','?')}]"
        )),
        ("RSI",          (
            f"5m={decision.get('rsi_14_5m','?')}  "
            f"15m={decision.get('rsi_14_15m','?')}"
        )),
        ("pnl_dia",      f"{decision.get('pnl_dia_pct',0):+.2f}%"),
    ]
    for label, val in ctx_keys:
        print(f"  {label:<16}: {val}")

    # ── Detalle de lotes abiertos ─────────────────────────────────────────────
    # Muestra entry price, PnL actual, y niveles aproximados de TP/SL por lote
    try:
        import pandas as _pd
        precio_now = float(decision.get("precio", 0) or 0)
        if df_inventory is not None and not df_inventory.empty and precio_now > 0:
            open_mask = df_inventory["status"].astype(str) == "OPEN"
            open_lots = df_inventory[open_mask]
            if not open_lots.empty:
                print(f"  {'─'*68}")
                for _, lot in open_lots.iterrows():
                    lid       = str(lot.get("lot_id", "?"))[:14]
                    entry     = float(lot.get("price_usdt", 0) or 0)
                    pnl_now   = float(lot.get("pnl_pct_actual", 0) or 0)
                    qty       = float(lot.get("qty_restante_btc", 0) or 0)
                    # Niveles TP/SL aproximados (por defecto del mercado)
                    tp1_px    = entry * 1.015  # +1.5% (TP mínimo)
                    sl_px     = entry * 0.980  # -2.0% (SL grave)
                    trail_cfg = POLICIES.get("trailing_profit", {})
                    t_min     = float(trail_cfg.get("min_pnl_activacion_pct", 0.5))
                    t_ret     = float(trail_cfg.get("retroceso_pct", 0.35))
                    pnl_sym   = "▲" if pnl_now >= 0 else "▼"
                    pnl_col   = "+" if pnl_now >= 0 else ""
                    print(
                        f"  📦 {lid}  entry=${entry:,.0f}  ahora=${precio_now:,.0f}  "
                        f"pnl={pnl_col}{pnl_now:.2f}% {pnl_sym}  qty={qty:.5f}BTC"
                    )
                    print(
                        f"     TP1≈${tp1_px:,.0f}(+1.5%)  SL≈${sl_px:,.0f}(-2.0%)  "
                        f"Trail: activa si pnl>+{t_min}% y retrocede {t_ret}%"
                    )
    except Exception:
        pass  # display extra es siempre no-crítico

    if act == "BUY":
        print(f"  Monto         : ${decision.get('qty_usdt', 0):.2f} USDT")
    elif act in ("SELL_V1", "SELL_V2"):
        print(f"  Lote          : {decision.get('lot_id', '?')}")
        print(f"  Vender        : {decision.get('sell_pct', 1.0)*100:.0f}% del lote")
    print(f"{SEP}")


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL: decide()
# ──────────────────────────────────────────────────────────────────────────────

def decide(
    ctx: Dict,
    df_inventory=None,
    df_ventas_hoy=None,
    last_buy_ctx: Optional[Dict] = None,
    last_regimen: Optional[str] = None,
    snapshot_id: str = "",
    dir_override: Optional[float] = None,
    conv_override: Optional[float] = None,
    score_compra_ema: Optional[float] = None,
    max_pnl_seen: Optional[Dict[str, float]] = None,
    lot_strategies: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Función principal del motor de decisiones.

    Args:
        ctx:               Snapshot del mercado (de M2)
        df_inventory:      DataFrame de lotes OPEN (de M1)
        df_ventas_hoy:     DataFrame de ventas del día (de M1)
        last_buy_ctx:      Contexto de la última compra (para cambio mínimo)
        snapshot_id:       ID del snapshot actual
        dir_override:      Dirección pre-suavizada (EMA). Si None, se calcula raw.
        conv_override:     Convicción pre-suavizada (EMA). Si None, se calcula raw.
        score_compra_ema:  EMA del score_compra (de DecisionEngine). Para señales sostenidas.
        max_pnl_seen:      Máximo PnL histórico por lote {lot_id: pct}. Para trailing profit.

    Returns:
        Dict con la decisión completa
    """

    # ── Enriquecer ctx con variables calculadas ───────────────────────────────
    lotes_en_perdida = _calc_lotes_en_perdida(df_inventory)
    pnl_dia          = _calc_pnl_dia(df_ventas_hoy)
    ctx["lotes_en_perdida"] = lotes_en_perdida
    ctx["pnl_dia_pct"]      = pnl_dia

    n_lotes    = int(_get(ctx, "n_lotes_open", 0))
    usdt_pct   = float(_get(ctx, "usdt_reserve_pct", 100.0))
    budget     = float(_get(ctx, "budget_usdt", 0.0))
    usdt_disp  = budget * usdt_pct / 100.0

    def _wait(reason: str, rule_id: str = "WAIT-DEFAULT", **kwargs) -> Dict:
        return _make_decision(
            action="WAIT", rule_id=rule_id, reason=reason,
            ctx=ctx, snapshot_id=snapshot_id, **kwargs
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CAPA 0 — GUARDIÁN
    # ══════════════════════════════════════════════════════════════════════════
    guardian_ok, guardian_effect, guardian_reason = _guardian(ctx)

    # ══════════════════════════════════════════════════════════════════════════
    # CAPA 1 — CLASIFICADOR DE RÉGIMEN
    # EMA aplicado externamente (DecisionEngine) para evitar oscilación en
    # la frontera de regímenes. dir/conv raw se calculan siempre para tener
    # el valor real, pero se clasifica con los valores suavizados.
    # ══════════════════════════════════════════════════════════════════════════
    direccion_raw  = _calc_direccion(ctx)
    conviccion_raw = _calc_conviccion(ctx)
    # Usar valor suavizado si viene del motor (EMA), raw si es llamada directa
    direccion  = dir_override  if dir_override  is not None else direccion_raw
    conviccion = conv_override if conv_override is not None else conviccion_raw
    regimen    = _clasificar_regimen(direccion, conviccion)
    reg_cfg    = _get_regimen_config(regimen)
    max_lotes  = reg_cfg.get("max_lotes", 1)
    umbral_cmp = reg_cfg.get("umbral_compra", 100)

    # ══════════════════════════════════════════════════════════════════════════
    # CAPA 2 — SCORES
    # ══════════════════════════════════════════════════════════════════════════
    sc_compra = _score_compra(ctx)

    # ══════════════════════════════════════════════════════════════════════════
    # PRIORIDAD 1 — BLOCK_ALL (Stop del día)
    # ══════════════════════════════════════════════════════════════════════════
    if guardian_effect == "BLOCK_ALL":
        return _wait(
            guardian_reason, "GUARDIAN-BLOCK-ALL",
            regimen=regimen, direccion=direccion, conviccion=conviccion,
            score_compra=sc_compra
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PRIORIDAD 2 — VENTAS (siempre evaluar, independiente del Guardián de compra)
    # ══════════════════════════════════════════════════════════════════════════
    if df_inventory is not None and not df_inventory.empty:
        open_lots = df_inventory[df_inventory["status"].astype(str) == "OPEN"]

        # ── TRAILING PROFIT LOCK ──────────────────────────────────────────────
        # Si el PnL del lote retrocedió desde su máximo más de retroceso_pct → vender.
        # Se evalúa ANTES del Regime-TP porque la ganancia ya está retrocediendo.
        tp_trail_cfg = POLICIES.get("trailing_profit", {})
        if tp_trail_cfg.get("enabled", False) and max_pnl_seen:
            min_activacion = float(tp_trail_cfg.get("min_pnl_activacion_pct", 0.5))
            retroceso      = float(tp_trail_cfg.get("retroceso_pct", 0.35))
            trail_candidates = []
            for _, lot in open_lots.iterrows():
                lot_dict = lot.to_dict()
                lid      = str(lot_dict.get("lot_id", ""))
                pnl_now  = float(lot_dict.get("pnl_pct_actual", 0.0) or 0.0)
                pnl_max  = max_pnl_seen.get(lid, 0.0)
                # Solo actúa si el lote alguna vez llegó al mínimo de activación
                if pnl_max >= min_activacion and (pnl_max - pnl_now) >= retroceso:
                    trail_candidates.append((pnl_max - pnl_now, pnl_now, pnl_max, lid, lot_dict))
            # Vender el que más retrocedió primero
            trail_candidates.sort(key=lambda x: x[0], reverse=True)
            for retroceso_real, pnl_now, pnl_max, lid, lot in trail_candidates:
                sell_pct = float(tp_trail_cfg.get("sell_pct", 1.0))
                return _make_decision(
                    action="SELL_V1",
                    rule_id="V1-TRAILING-TP",
                    reason=f"Trailing TP: max={pnl_max:.2f}% → actual={pnl_now:.2f}% | retroceso={retroceso_real:.2f}% >= {retroceso}%",
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=regimen, direccion=direccion, conviccion=conviccion,
                    score_venta=round(pnl_now, 1),
                    lot_id=lid,
                    sell_pct=sell_pct,
                )

        # ── REGIME CHANGE TP ──────────────────────────────────────────────────
        # Si el régimen mejoró desde el loop anterior, evaluar TP escalado por salto.
        # Prioridad: antes de SL/TP estándar, porque el cambio de régimen ya confirmó el movimiento.
        # EXCEPCIÓN: lotes S2 (pacientes) ignoran este trigger — esperan trailing/hard TP (≥1.0%).
        if last_regimen and last_regimen != regimen:
            rc_threshold = _calc_regime_tp_threshold(last_regimen, regimen)
            if rc_threshold is not None:
                hierarchy = POLICIES.get("regime_change_tp", {}).get("hierarchy", {})
                jump = hierarchy.get(regimen, 0) - hierarchy.get(last_regimen, 0)
                _lot_strats = lot_strategies or {}
                rc_candidates = []
                for _, lot in open_lots.iterrows():
                    lot_dict = lot.to_dict()
                    lid      = str(lot_dict.get("lot_id", ""))
                    pnl      = float(lot_dict.get("pnl_pct_actual", 0.0) or 0.0)
                    # Saltarse lotes S2 — esperan TP más grande
                    if _lot_strats.get(lid, "S1") == "S2":
                        continue
                    if pnl >= rc_threshold:
                        rc_candidates.append((pnl, lot_dict))
                # Vender el lote MÁS CARO primero (menor pnl% = comprado más caro = el COMÚN)
                # El lote más barato (mayor pnl%) es el KEEPER — se queda para el movimiento grande
                rc_candidates.sort(key=lambda x: x[0])
                for pnl, lot in rc_candidates:
                    sell_pct = _get_sell_pct_tp(regimen)
                    return _make_decision(
                        action="SELL_V1",
                        rule_id="V1-REGIME-TP",
                        reason=f"Regime TP: {last_regimen}→{regimen} | salto={jump} | umbral={rc_threshold:.2f}% | pnl={pnl:.2f}% | sell={sell_pct*100:.0f}%",
                        ctx=ctx, snapshot_id=snapshot_id,
                        regimen=regimen, direccion=direccion, conviccion=conviccion,
                        score_venta=round(pnl, 1),
                        lot_id=str(lot.get("lot_id", "")),
                        sell_pct=sell_pct,
                    )
                
                # ── HARD TAKE PROFIT POR COMPORTAMIENTO DEL LOTE ─────────────────────  SmartBot GPT
        # Objetivo:
        # No dejar vivo un lote con ganancia clara solo porque el score_venta_tp
        # no alcanzó el umbral dinámico del régimen.
        #
        # Filosofía:
        # - Primero manda el lote (ganancia ya construida)
        # - Luego el mercado confirma de forma liviana
        #
        # Esto NO reemplaza trailing ni regime-change TP.
        # Es un fallback intermedio para liberar inventario en laterales/rebotes.

        hard_tp_cfg = POLICIES.get("hard_take_profit", {})

        if hard_tp_cfg.get("enabled", True):
            min_pnl_hard   = float(hard_tp_cfg.get("min_pnl_pct", 1.00))
            min_pos_1m     = float(hard_tp_cfg.get("min_pos_1m", 0.75))
            min_score_tp   = float(hard_tp_cfg.get("min_score_tp", 10))
            pullback_min   = float(hard_tp_cfg.get("pullback_from_max_pct", 0.20))
            sell_pct_hard  = float(hard_tp_cfg.get("sell_pct", 1.0))

            # Evitar cortar runners demasiado pronto en regímenes alcistas buenos
            allowed_regimes = set(hard_tp_cfg.get(
                "allowed_regimes",
                ["LATERAL_DEFINIDO", "LATERAL_DEBIL", "CORRECCION", "REBOTE_BAJISTA", "BAJISTA_MODERADO", "BAJISTA_DUDOSO"]
            ))

            if regimen in allowed_regimes:
                pos_1m_now = float(_get(ctx, "price_position_1m", 0.0) or 0.0)
                hard_tp_candidates = []

                for _, lot in open_lots.iterrows():
                    lot_dict = lot.to_dict()
                    lid      = str(lot_dict.get("lot_id", ""))
                    pnl_now  = float(lot_dict.get("pnl_pct_actual", 0.0) or 0.0)
                    _lot_strat_htp = (lot_strategies or {}).get(lid, "S1")
                    sc_tp    = _score_venta_tp(ctx, lot_dict, strategy=_lot_strat_htp)

                    # Si no vino max_pnl_seen, usar pnl actual como fallback seguro
                    pnl_max = float(max_pnl_seen.get(lid, pnl_now) if max_pnl_seen else pnl_now)
                    pullback = pnl_max - pnl_now

                    # Regla:
                    # Vender si ya hay ganancia suficiente y además se cumple al menos
                    # una señal de salida "suave":
                    #   A) precio alto dentro del rango corto
                    #   B) score_tp ya dio señales, aunque no llegue al umbral completo
                    #   C) ya hubo retroceso desde el máximo
                    if pnl_now >= min_pnl_hard and (
                        pos_1m_now >= min_pos_1m or
                        sc_tp >= min_score_tp or
                        pullback >= pullback_min
                    ):
                        # Orden: vender primero el lote más "común"/caro
                        # -> menor pnl% primero
                        hard_tp_candidates.append((pnl_now, sc_tp, pullback, lot_dict))

                hard_tp_candidates.sort(key=lambda x: x[0])

                for pnl_now, sc_tp, pullback, lot in hard_tp_candidates:
                    return _make_decision(
                        action="SELL_V1",
                        rule_id="V1-HARD-TP",
                        reason=(
                            f"Hard TP: pnl={pnl_now:.2f}% >= {min_pnl_hard:.2f}% "
                            f"| pos_1m={pos_1m_now:.2f} "
                            f"| sc_tp={sc_tp:.0f} "
                            f"| pullback={pullback:.2f}% "
                            f"| régimen={regimen}"
                        ),
                        ctx=ctx, snapshot_id=snapshot_id,
                        regimen=regimen, direccion=direccion, conviccion=conviccion,
                        score_venta=max(float(sc_tp), round(pnl_now, 1)),
                        lot_id=str(lot.get("lot_id", "")),
                        sell_pct=sell_pct_hard,
                    )

        # Ordenar: para SL usar peor lote primero, para TP mejor primero
        sl_candidates = []
        tp_candidates = []

        for _, lot in open_lots.iterrows():
            lot_dict = lot.to_dict()
            pnl_pct  = float(lot_dict.get("pnl_pct_actual", 0.0) or 0.0)
            lot_dict["pnl_pct_actual"] = pnl_pct
            _lid_tp   = str(lot_dict.get("lot_id", ""))
            _strat_tp = (lot_strategies or {}).get(_lid_tp, "S1")

            sc_sl = _score_venta_sl(ctx, lot_dict, regimen)
            sc_tp = _score_venta_tp(ctx, lot_dict, strategy=_strat_tp)

            if sc_sl > 0:   sl_candidates.append((sc_sl, lot_dict))
            if sc_tp > 0:   tp_candidates.append((sc_tp, lot_dict))

        # SL primero — ordenado por score descendente
        sl_candidates.sort(key=lambda x: x[0], reverse=True)
        for sc_sl, lot in sl_candidates:
            if sc_sl >= 30:  # umbral mínimo para ejecutar SL
                sell_pct = 1.0  # SL siempre 100%
                return _make_decision(
                    action="SELL_V2",
                    rule_id="V2-SL",
                    reason=f"SL: score={sc_sl:.0f} | pnl={lot.get('pnl_pct_actual',0):.2f}% | régimen={regimen}",
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=regimen, direccion=direccion, conviccion=conviccion,
                    score_venta=sc_sl,
                    lot_id=str(lot.get("lot_id", "")),
                    sell_pct=sell_pct,
                )

        # TP — ordenado por score descendente
        # Umbral dinámico por régimen (en policies) — más bajo en laterales donde el movimiento es suave
        _tp_umbrales = POLICIES.get("scores", {}).get("venta_tp", {}).get("umbral_por_regimen", {})
        umbral_tp    = int(_tp_umbrales.get(regimen, _tp_umbrales.get("_default", 35)))
        tp_candidates.sort(key=lambda x: x[0], reverse=True)
        for sc_tp, lot in tp_candidates:
            if sc_tp >= umbral_tp:
                sell_pct = _get_sell_pct_tp(regimen)
                return _make_decision(
                    action="SELL_V1",
                    rule_id="V1-TP",
                    reason=f"TP: score={sc_tp:.0f}>={umbral_tp} | pnl={lot.get('pnl_pct_actual',0):.2f}% | régimen={regimen} | sell={sell_pct*100:.0f}%",
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=regimen, direccion=direccion, conviccion=conviccion,
                    score_venta=sc_tp,
                    lot_id=str(lot.get("lot_id", "")),
                    sell_pct=sell_pct,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # PRIORIDAD 3 — COMPRA
    # ══════════════════════════════════════════════════════════════════════════

    # ── MAX LOTES GLOBAL ─────────────────────────────────────────────────────
    # Hard limit: máximo 2 lotes abiertos. Override sobre max_lotes por régimen.
    max_global = int(POLICIES["sizing"].get("max_lotes_global", 99))
    if n_lotes >= max_global:
        return _wait(
            f"max_lotes_global={max_global} alcanzado ({n_lotes} abiertos)",
            "WAIT-MAX-GLOBAL",
            regimen=regimen, direccion=direccion, conviccion=conviccion,
            score_compra=sc_compra,
        )

    # ── SOLO PROMEDIA HACIA ABAJO ─────────────────────────────────────────────
    # Si ya hay 1+ lotes abiertos, solo comprar si el precio actual es MENOR
    # que el precio del lote más barato. Nunca promediar hacia arriba.
    if n_lotes >= 1 and df_inventory is not None and not df_inventory.empty:
        import pandas as pd
        open_mask    = df_inventory["status"].astype(str) == "OPEN"
        open_lots_df = df_inventory[open_mask]
        if not open_lots_df.empty:
            min_lot_price = pd.to_numeric(open_lots_df["price_usdt"], errors="coerce").min()
            precio_actual = float(_get(ctx, "last_price", 0))
            if precio_actual >= min_lot_price:
                return _wait(
                    f"no_promedio_abajo: precio={precio_actual:,.0f} >= lote_mas_barato={min_lot_price:,.0f}",
                    "WAIT-NO-PROMEDIO",
                    regimen=regimen, direccion=direccion, conviccion=conviccion,
                    score_compra=sc_compra,
                )

    if guardian_effect == "BLOCK_BUY":
        return _wait(
            guardian_reason, "GUARDIAN-BLOCK-BUY",
            regimen=regimen, direccion=direccion, conviccion=conviccion,
            score_compra=sc_compra
        )

    # ── FILTROS DE CALIDAD DE ENTRADA ──────────────────────────────────────────
    # Regla A: trend_1m no confirma dip.
    #   "up"       → precio subiendo en 1m = rebote activo, no es dip para comprar.
    #   "flat-weak" + downside_1h > 0.20% → pausa sin dirección Y hay espacio real
    #               para caer más → entrada sin confirmación de soporte.
    # Excepción intencional: "flat-weak" con downside_1h <= 0.20% se permite
    # porque el soporte está tan cerca que el riesgo de caída adicional es mínimo.
    _trend_1m  = str(_get(ctx, "trend_1m", "down") or "down")
    _d1h       = float(_get(ctx, "downside_1h_pct", 0.0) or 0.0)
    _regla_a   = (_trend_1m == "up") or (_trend_1m == "flat-weak" and _d1h > 0.20)
    if _regla_a:
        return _wait(
            f"calidad_entrada_A: trend_1m={_trend_1m} d1h={_d1h:.3f}% — sin confirmacion de dip en 1m",
            "WAIT-CALIDAD-A",
            regimen=regimen, direccion=direccion, conviccion=conviccion,
            score_compra=sc_compra,
        )

    # Regla B: pnl_dia alto + pos_1h bajo = comprar tarde en el día sin impulso horario.
    #   pnl_dia > 1.5%  → el bot ya extrajo buen valor hoy, el mercado se extendió.
    #   pos_1h  < 0.75  → precio NO está en la parte alta del rango horario,
    #                      no hay momentum alcista que justifique nueva entrada.
    #   Si pos_1h >= 0.75 el precio muestra momentum real en la hora → se permite.
    _pnl_dia   = float(pnl_dia or 0.0)
    _pos_1h    = float(_get(ctx, "pos_1h", 1.0) or 1.0)
    _regla_b   = _pnl_dia > 1.5 and _pos_1h < 0.75
    if _regla_b:
        return _wait(
            f"calidad_entrada_B: pnl_dia={_pnl_dia:+.2f}% > 1.5% con pos_1h={_pos_1h:.2f} < 0.75 — sin momentum horario",
            "WAIT-CALIDAD-B",
            regimen=regimen, direccion=direccion, conviccion=conviccion,
            score_compra=sc_compra,
        )

    # Score suficiente para el régimen? — condición original O señal EMA sostenida
    sc_ema_cfg     = POLICIES.get("score_ema", {})
    sc_ema_enabled = sc_ema_cfg.get("enabled", False)
    sc_ema_delta   = float(sc_ema_cfg.get("delta_umbral", 6))
    sc_ema_delta2  = float(sc_ema_cfg.get("delta_score_actual", 12))
    # Condición EMA: score_ema >= umbral-delta Y score_actual >= umbral-delta2
    # Captura señales que bailan justo por debajo del umbral de forma sostenida
    ema_dispara = (
        sc_ema_enabled
        and score_compra_ema is not None
        and score_compra_ema >= (umbral_cmp - sc_ema_delta)
        and sc_compra >= (umbral_cmp - sc_ema_delta2)
    )
    if sc_compra >= umbral_cmp or ema_dispara:
        # ── Detectar estrategia S1 vs S2 ─────────────────────────────────────
        # S2 (paciente): soporte real + posición baja + volatilidad calmada.
        # Lotes S2 ignoran regime_change_tp y esperan trailing/hard TP (≥1.0%).
        # S1 (rápido): cualquier otra condición. Sale con regime_change_tp (~0.25%).
        _s2_pos1h = float(_get(ctx, "price_position_1h") or 0.5)
        _s2_pos1w = float(_get(ctx, "price_position_1w") or 0.5)
        _s2_ns    = bool(_get(ctx, "near_support", False))
        _s2_vol   = str(_get(ctx, "vol_state") or "normal")
        _is_s2    = (
            _s2_ns
            and _s2_pos1h < 0.40
            and _s2_pos1w < 0.70
            and _s2_vol in ("calm", "normal")
        )
        buy_strategy = "S2" if _is_s2 else "S1"

        # Calcular sizing
        min_trade   = POLICIES["sizing"].get("min_trade_usdt", 10.0)
        sizing_usdt = _get_sizing_compra(regimen, sc_compra, usdt_disp, min_trade)

        if sizing_usdt >= min_trade:
            # Auditor
            veredicto, sizing_final, razon_auditor = _auditor(
                accion="BUY",
                regimen=regimen,
                score_compra=sc_compra,
                umbral_compra=umbral_cmp,
                sizing_usdt=sizing_usdt,
                ctx=ctx,
                last_buy_ctx=last_buy_ctx or {},
                n_lotes=n_lotes,
                max_lotes=max_lotes,
                lotes_en_perdida=lotes_en_perdida,
                pnl_dia=pnl_dia,
            )

            if veredicto == "VETADO":
                return _wait(
                    f"Auditor VETADO: {razon_auditor}",
                    "AUDITOR-VETO",
                    regimen=regimen, direccion=direccion, conviccion=conviccion,
                    score_compra=sc_compra, veredicto_auditor=veredicto
                )

            buy_rule   = f"BUY-{regimen}" if sc_compra >= umbral_cmp else f"BUY-EMA-{regimen}"
            buy_reason = (
                f"score={sc_compra:.0f}>={umbral_cmp} | régimen={regimen} | dir={direccion:+.0f} | conv={conviccion:.0f}"
                f" | strategy={buy_strategy}"
                if sc_compra >= umbral_cmp else
                f"score_ema={score_compra_ema:.1f}>={umbral_cmp - sc_ema_delta:.0f} (sostenido) | score={sc_compra:.0f}"
                f" | régimen={regimen} | strategy={buy_strategy}"
            )
            return _make_decision(
                action="BUY",
                rule_id=buy_rule,
                reason=buy_reason,
                ctx=ctx, snapshot_id=snapshot_id,
                regimen=regimen, direccion=direccion, conviccion=conviccion,
                score_compra=sc_compra,
                qty_usdt=sizing_final,
                signal_strength="strong" if sc_compra >= 75 else "normal" if sc_compra >= 55 else "weak",
                veredicto_auditor=veredicto,
                strategy=buy_strategy,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # PRIORIDAD 4 — WAIT (default)
    # ══════════════════════════════════════════════════════════════════════════
    wait_reasons = []
    if sc_compra < umbral_cmp and not ema_dispara:
        ema_str = f" (ema={score_compra_ema:.1f})" if score_compra_ema is not None else ""
        wait_reasons.append(f"score_compra={sc_compra:.0f}{ema_str} < umbral={umbral_cmp} para {regimen}")
    if n_lotes >= max_lotes:
        wait_reasons.append(f"lotes={n_lotes} >= max={max_lotes} en {regimen}")

    return _wait(
        " | ".join(wait_reasons) or "sin señal",
        regimen=regimen, direccion=direccion, conviccion=conviccion,
        score_compra=sc_compra
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLASE DecisionEngine (interfaz para M6)
# ──────────────────────────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Interfaz principal de M3 para M6.
    Mantiene estado entre loops (last_buy_ctx para cambio mínimo).
    """

    def __init__(self, dry_run: bool = True):
        global DRY_RUN
        DRY_RUN = dry_run
        self._last_buy_ctx: Optional[Dict] = None
        self._last_regimen: Optional[str]  = None
        # EMA de direccion/conviccion — evita oscilación en fronteras de régimen
        self._dir_ema:  Optional[float] = None
        self._conv_ema: Optional[float] = None
        # EMA de score_compra — captura señales sostenidas que nunca tocan el umbral exacto
        self._score_compra_ema: Optional[float] = None
        # M3.5 — LotTracker + LotAnalytics + LotContext
        # Reemplaza self._max_pnl_seen (que vivía solo en RAM y se perdía en reinicios)
        from lot_tracker   import LotTracker
        from lot_analytics import LotAnalytics
        from lot_context   import LotContext
        self._tracker   = LotTracker()
        self._analytics = LotAnalytics()
        self._lot_ctx   = LotContext(self._tracker, self._analytics)

        # M3.6 — Libretita de mercado: últimos 20 ticks (≈3 min a 10s/loop)
        # Cada entrada: {vol_state, pos_1h, precio, ts_epoch}
        # Persiste entre loops en RAM — se reinicia solo al reiniciar el bot.
        self._tick_history: Deque[Dict] = deque(maxlen=20)

        log.info(f"DecisionEngine inicializado · dry_run={dry_run} · versión={POLICIES['meta']['version']}")

    def decide(
        self,
        ctx: Dict,
        df_inventory=None,
        df_ventas_hoy=None,
        snapshot_id: str = "",
    ) -> Dict:
        """Evalúa el contexto y retorna una decisión."""

        # ── EMA suavizado de direccion/conviccion ────────────────────────────
        # Calculamos raw aquí para actualizar el EMA antes de pasarlo al motor.
        # Esto evita que un solo tick en la frontera (ej: conv=41 vs 39) cambie
        # el régimen y su umbral de compra en cada loop.
        alpha     = float(POLICIES["regimen"].get("ema_alpha", 0.4))
        dir_raw   = _calc_direccion(ctx)
        conv_raw  = _calc_conviccion(ctx)

        if self._dir_ema is None:
            # Primer tick: inicializar con valor raw (sin suavizado todavía)
            self._dir_ema  = dir_raw
            self._conv_ema = conv_raw
        else:
            self._dir_ema  = round(alpha * dir_raw  + (1 - alpha) * self._dir_ema,  1)
            self._conv_ema = round(alpha * conv_raw + (1 - alpha) * self._conv_ema, 1)

        # ── EMA de score_compra ──────────────────────────────────────────────
        sc_raw = _score_compra(ctx)
        if self._score_compra_ema is None:
            self._score_compra_ema = sc_raw
        else:
            self._score_compra_ema = round(
                alpha * sc_raw + (1 - alpha) * self._score_compra_ema, 1
            )

        # ── Actualizar libretita de mercado (tick_history) ───────────────────
        price_now = float(ctx.get("last_price") or 0.0)
        self._tick_history.append({
            "vol_state": str(ctx.get("vol_state") or ""),
            "pos_1h":    float(ctx.get("price_position_1h") or 0.5),
            "precio":    price_now,
            "ts_epoch":  datetime.now(timezone.utc).timestamp(),
        })

        # ── Señales micro (últimos 2 min = 12 ticks) ─────────────────────────
        # Calculadas ANTES de llamar al motor para poder bloquear si es necesario.
        _hist = list(self._tick_history)        # snapshot inmutable
        _hist_2m = _hist[-12:]                  # últimos ~2 minutos

        # chaos_ticks_2min: cuántos ticks recientes tuvieron vol=chaos
        _chaos_ticks = sum(1 for t in _hist_2m if t["vol_state"] == "chaos")

        # pos1h_rising: ¿pos_1h subió en la última ventana? (trampa de rebote falso)
        _pos1h_vals = [t["pos_1h"] for t in _hist_2m if "pos_1h" in t]
        _pos1h_rising = (
            len(_pos1h_vals) >= 6 and _pos1h_vals[-1] > _pos1h_vals[-6]
        )

        # price_vel_2m: % cambio de precio en últimos 2 min
        _prices = [t["precio"] for t in _hist_2m if t["precio"] > 0]
        _price_vel_2m = (
            (_prices[-1] - _prices[0]) / _prices[0] * 100
            if len(_prices) >= 4 and _prices[0] > 0 else 0.0
        )

        # vol_escalating: el vol empezó bajo y escaló a chaos en la ventana
        _vol_order = {"calm": 0, "normal": 1, "high": 2, "chaos": 3}
        _vol_scores = [_vol_order.get(t["vol_state"], 0) for t in _hist_2m if t["vol_state"]]
        _vol_escalating = (
            len(_vol_scores) >= 6
            and _vol_scores[-1] > _vol_scores[0]
            and _vol_scores[-1] >= 3  # termina en chaos
        )

        # ── Contexto macro desde velas (responde "¿cuánto ya cayó?") ─────────
        _rsi_15m = float(ctx.get("rsi_14_15m") or 50.0)
        _rsi_5m  = float(ctx.get("rsi_14_5m")  or 50.0)
        # Sobrevendido = el precio ya cayó mucho → posible fondo legítimo
        _oversold = _rsi_15m < 32 or _rsi_5m < 28

        # ── UTC hour/minute para regla de medianoche ──────────────────────────
        _utc_now  = datetime.now(timezone.utc)
        _utc_h    = _utc_now.hour
        _utc_m    = _utc_now.minute
        _midnight = (_utc_h == 23 and _utc_m >= 45) or (_utc_h == 0 and _utc_m <= 15)

        # ══════════════════════════════════════════════════════════════════════
        # FILTROS DE CALIDAD DE ENTRADA — NIVEL 2 (contexto histórico)
        # Solo aplican a BUY. Evaluados ANTES del motor para ahorrar ciclos.
        # ══════════════════════════════════════════════════════════════════════
        # Solo bloquear si el motor efectivamente querría comprar (score alto o
        # la señal está lista). Usamos una verificación rápida del score.
        _quick_sc = _score_compra(ctx)
        _regimen_actual = _clasificar_regimen(self._dir_ema, self._conv_ema)
        _reg_cfg   = POLICIES.get("regimen", {}).get("clasificacion", {}).get(_regimen_actual, {})
        _umbral    = int(_reg_cfg.get("umbral_compra", 999)) if isinstance(_reg_cfg, dict) else 999
        _buy_likely = _quick_sc >= (_umbral - 10) or _quick_sc >= 70

        if _buy_likely:
            # Regla HR — Doble techo: precio en zona alta TANTO en 1h COMO en 1w
            # Si pos_1h > 0.75 Y pos_1w > 0.70 = comprando en cima de corto Y mediano plazo.
            # El scoring suave ya penaliza esto, pero la regla dura cierra cualquier
            # camino residual (ej. ALCISTA_FUERTE con umbral=35 donde el score aún llegaría).
            # Override RSI: un breakout genuino con RSI sobrevendido es teóricamente
            # posible pero extremadamente raro — si RSI está bajo y precio en techo semanal,
            # hay algo estructural mal y conviene esperar confirmación.
            _pos1h_hr  = float(ctx.get("price_position_1h") or 0.5)
            _pos1w_hr  = float(ctx.get("price_position_1w") or 0.5)
            _regla_hr  = _pos1h_hr > 0.70 and _pos1w_hr > 0.70
            if _regla_hr and not _oversold:
                return _make_decision(
                    action="WAIT", rule_id="WAIT-HIGH-RANGE",
                    reason=(
                        f"high_range: pos_1h={_pos1h_hr:.3f} > 0.70 "
                        f"Y pos_1w={_pos1w_hr:.3f} > 0.70 — "
                        f"comprando en doble techo (1h+1w) | "
                        f"rsi_15m={_rsi_15m:.1f} rsi_5m={_rsi_5m:.1f}"
                    ),
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=_regimen_actual, score_compra=_quick_sc,
                )

            # Regla TD — Tendencia bajista con precio alto: trend_1m=down + pos_1h > 0.65
            # Comprar en caída cuando ya estamos en zona alta = trampa de dip en techo.
            # No es soporte real — es el precio bajando desde un pico reciente.
            # Override si sobrevendido (RSI confirma que la caída ya fue grande).
            _trend_1m_now = str(_get(ctx, "trend_1m") or "flat-weak")
            _pos1h_now_td = float(ctx.get("price_position_1h") or 0.5)
            _regla_td = _trend_1m_now == "down" and _pos1h_now_td > 0.65
            if _regla_td and not _oversold:
                return _make_decision(
                    action="WAIT", rule_id="WAIT-TREND-DOWN-HIGH",
                    reason=(
                        f"trend_down_high: trend_1m=down Y pos_1h={_pos1h_now_td:.3f} > 0.65 — "
                        f"comprando caída desde zona alta (sin soporte real) | "
                        f"rsi_15m={_rsi_15m:.1f} rsi_5m={_rsi_5m:.1f} — no sobrevendido"
                    ),
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=_regimen_actual, score_compra=_quick_sc,
                )

            # Regla VC — Volatilidad caótica invalida near_support
            # vol=chaos significa movimiento desordenado sin dirección estable.
            # En chaos, "near_support" (mínimo de 1 vela) no es soporte real.
            # Si el score llegó al umbral SOLO por near_support (+15pts en score_compra
            # más +5 en dirección), el chaos lo invalida — el bot no debe comprar.
            # Override si sobrevendido (caída ya fue grande aunque haya chaos).
            _vol_now       = str(_get(ctx, "vol_state") or "normal")
            _ns_now        = bool(_get(ctx, "near_support", False))
            # Estimación aproximada: ¿cuánto aporta near_support al score actual?
            # Si sin near_support el score caería por debajo del umbral, bloqueamos.
            _ns_pts_est    = 15  # puntos que near_support aporta en _score_compra
            _score_sin_ns  = _quick_sc - _ns_pts_est if _ns_now else _quick_sc
            _umbral_now    = int(POLICIES.get("regimen", {}).get("clasificacion", {}).get(
                _regimen_actual, {}).get("umbral_compra", 999))
            _regla_vc = _vol_now == "chaos" and _ns_now and _score_sin_ns < _umbral_now
            if _regla_vc and not _oversold:
                return _make_decision(
                    action="WAIT", rule_id="WAIT-VOL-CHAOS-NS",
                    reason=(
                        f"vol_chaos_near_support: vol=chaos invalida near_support — "
                        f"score={_quick_sc:.0f} sin_ns≈{_score_sin_ns:.0f} < umbral={_umbral_now} | "
                        f"rsi_15m={_rsi_15m:.1f} rsi_5m={_rsi_5m:.1f} — no sobrevendido"
                    ),
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=_regimen_actual, score_compra=_quick_sc,
                )

            # Regla C — Ruido de medianoche UTC (cierre de vela diaria)
            # La liquidez cae y hay spike artificial. No hay override por RSI.
            if _midnight:
                regimen_actual = _regimen_actual
                return _make_decision(
                    action="WAIT", rule_id="WAIT-MIDNIGHT-UTC",
                    reason=(
                        f"midnight_utc: {_utc_h:02d}:{_utc_m:02d} UTC — "
                        f"ventana 23:45-00:15 bloqueada (ruido de cierre de vela diaria)"
                    ),
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=_regimen_actual, score_compra=_quick_sc,
                )

            # Regla F — Trampa de rebote: chaos sostenido + pos_1h sube
            # El bot ve un pequeño bounce y quiere comprar, pero el mercado sigue malo.
            # Override si RSI sobrevendido (precio ya cayó mucho = fondo posible).
            _regla_f = _chaos_ticks >= 8 and _pos1h_rising
            if _regla_f and not _oversold:
                return _make_decision(
                    action="WAIT", rule_id="WAIT-BOUNCE-TRAP",
                    reason=(
                        f"bounce_trap: chaos={_chaos_ticks}/12 ticks, pos_1h sube "
                        f"(rebote falso en mercado caótico) | "
                        f"rsi_15m={_rsi_15m:.1f} rsi_5m={_rsi_5m:.1f} — no sobrevendido"
                    ),
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=_regimen_actual, score_compra=_quick_sc,
                )

            # Regla WF — Waterfall activo: vol escaló + velocidad de caída alta
            # Override si RSI sobrevendido (caída ya consumió su energía).
            _regla_wf = _vol_escalating and _chaos_ticks >= 4 and _price_vel_2m < -0.40
            if _regla_wf and not _oversold:
                return _make_decision(
                    action="WAIT", rule_id="WAIT-WATERFALL",
                    reason=(
                        f"waterfall: vel_2m={_price_vel_2m:+.3f}% chaos={_chaos_ticks}/12 "
                        f"vol_escalando={_vol_escalating} | "
                        f"rsi_15m={_rsi_15m:.1f} rsi_5m={_rsi_5m:.1f} — no sobrevendido"
                    ),
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=_regimen_actual, score_compra=_quick_sc,
                )

            # Regla OI-LIQ — Cascada de liquidaciones en futuros
            # OI baja + precio baja = posiciones forzadas cerrando (liq cascade).
            # Dos umbrales: señal formal del módulo O caída moderada de OI (-0.5%)
            # combinada con precio cayendo (-0.20% en 2min) como señal temprana.
            # SIN override RSI: las cascadas de liquidaciones no respetan oversold —
            # cada liq fuerza el precio abajo y dispara más liqidaciones.
            _oi_sig  = str(ctx.get("oi_price_signal") or "stable")
            _oi_chg  = float(ctx.get("oi_change_pct")  or 0.0)
            _regla_oiliq = (
                _oi_sig == "liq_cascade"                          # módulo formal
                or (_oi_chg < -0.5 and _price_vel_2m < -0.20)    # señal temprana
            )
            if _regla_oiliq:
                return _make_decision(
                    action="WAIT", rule_id="WAIT-LIQ-CASCADE",
                    reason=(
                        f"liq_cascade: oi_signal={_oi_sig} oi_chg={_oi_chg:+.3f}% "
                        f"vel_2m={_price_vel_2m:+.3f}% | "
                        f"rsi_15m={_rsi_15m:.1f} — sin override (liquidaciones estructurales)"
                    ),
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=_regimen_actual, score_compra=_quick_sc,
                )

            # Regla SH — Mercado sin calmar (Sustained High Volatility)
            # Patrón: vol elevada sostenida (high/chaos) durante toda la ventana
            # mientras el precio sigue en zona baja de su rango 1h y no sube.
            # Diagnóstico: mercado todavía en estado post-crash, no recuperado.
            # Diferencia con WF: WF ve la ESCALADA hacia caos. SH ve el ESTADO
            # sostenido de alto vol — aunque no esté subiendo, tampoco se calmó.
            # Override RSI: si está en oversold extremo, puede ser fondo legítimo.
            _high_chaos_all = sum(
                1 for t in _hist if t["vol_state"] in ("high", "chaos")
            )
            _pos1h_now   = float(ctx.get("price_position_1h") or 0.5)
            _crash_zone  = _pos1h_now < 0.30          # tercio inferior del rango 1h
            _no_rally    = _price_vel_2m < +0.10       # precio no está subiendo
            _regla_sh    = (
                _high_chaos_all >= 16                  # ≥80% de los 20 ticks elevados
                and _crash_zone
                and _no_rally
            )
            if _regla_sh and not _oversold:
                return _make_decision(
                    action="WAIT", rule_id="WAIT-SUSTAINED-HIGH",
                    reason=(
                        f"sustained_high: {_high_chaos_all}/20 ticks en high/chaos "
                        f"pos_1h={_pos1h_now:.3f} (crash_zone) "
                        f"vel_2m={_price_vel_2m:+.3f}% | "
                        f"rsi_15m={_rsi_15m:.1f} rsi_5m={_rsi_5m:.1f} — no sobrevendido"
                    ),
                    ctx=ctx, snapshot_id=snapshot_id,
                    regimen=_regimen_actual, score_compra=_quick_sc,
                )

        # ── Actualizar LotTracker con tick actual ────────────────────────────
        regimen_actual = _regimen_actual
        try:
            self._tracker.update(df_inventory, price_now, regimen_actual)
        except Exception as _e:
            log.warning(f"LotTracker.update() error (no crítico): {_e}")

        max_pnl_seen  = self._tracker.get_all_max_pnl()
        lot_strategies = self._tracker.get_all_strategies()

        # ── Llamar motor de decisiones ───────────────────────────────────────
        decision = decide(
            ctx               = ctx,
            df_inventory      = df_inventory,
            df_ventas_hoy     = df_ventas_hoy,
            last_buy_ctx      = self._last_buy_ctx,
            last_regimen      = self._last_regimen,
            snapshot_id       = snapshot_id,
            dir_override      = self._dir_ema,
            conv_override     = self._conv_ema,
            score_compra_ema  = self._score_compra_ema,
            max_pnl_seen      = max_pnl_seen,
            lot_strategies    = lot_strategies,
        )

        # ── Actualizar estado interno ────────────────────────────────────────
        if decision.get("action") == "BUY":
            self._last_buy_ctx = ctx.copy()
            # Registrar estrategia para que LotTracker la asigne al nuevo lote
            buy_strat = decision.get("strategy", "S1")
            self._tracker.set_next_lot_strategy(buy_strat)
        self._last_regimen = decision.get("regimen") or regimen_actual

        # ── Imprimir decisión en terminal ────────────────────────────────────
        print_decision(decision, df_inventory=df_inventory)

        return decision

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE SOPORTE (llamados desde loop_principal.py)
    # ──────────────────────────────────────────────────────────────────────────

    def log_outcome(self, decision_id: str, outcome: dict) -> None:
        """
        Registra el outcome (resultado de venta) en el .jsonl mensual.
        Llamado por Accountant tras confirmar cierre/parcial de lote.
        """
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            import json as _json
            month    = datetime.now().strftime("%Y%m")
            log_path = _LOG_DIR / f"decisions_{month}.jsonl"
            entry    = {**outcome, "decision_id": decision_id, "event": "OUTCOME"}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            log.warning(f"log_outcome: no se pudo escribir outcome {decision_id}: {e}")

    def maybe_refresh_analytics(self) -> None:
        """
        Refresca LotAnalytics si han pasado suficientes minutos desde el último refresh.
        Llamar cada ~60s (en el bloque force_ohlcv de loop_principal).
        """
        try:
            self._analytics.maybe_refresh()
        except Exception as e:
            log.warning(f"maybe_refresh_analytics error (no crítico): {e}")

    def on_lot_closed(self) -> None:
        """
        Fuerza refresh de LotAnalytics al cerrar un lote.
        Llamar tras confirmar cierre de lote en loop_principal.
        """
        try:
            self._analytics.refresh_on_close()
        except Exception as e:
            log.warning(f"on_lot_closed error (no crítico): {e}")
          