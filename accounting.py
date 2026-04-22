# -*- coding: utf-8 -*-
"""
Módulo 5 — Contabilidad Post-Trade · accounting.py

Se ejecuta DESPUÉS de que M4 confirma una orden ejecutada.
Nunca toca el exchange — solo lee/escribe Google Sheets y el .jsonl.

Responsabilidades:
  - Post-compra  → crear lote nuevo en Sheets (lotes)
  - Post-venta   → actualizar qty_restante, calcular PnL, cerrar lote si corresponde
  - Ventas log   → escribir en Sheets (ventas, ventas_detalle)
  - Outcome log  → completar el .jsonl con event=OUTCOME para entrenamiento IA
  - Heartbeat    → actualizar status_hb en Sheets

Dependencias:
  M1 → inventory_status.py  (new_lot_id, append_row_dict, update_lote_fields,
                              snapshot_lote_history, write_log, write_status_hb)
  M3 → decision_engine.py   (log_outcome)
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — Config, imports de M1/M3 y helpers
# ──────────────────────────────────────────────────────────────────────────────

DRY_RUN: bool = os.getenv("DRY_RUN", "1") == "1"
SYMBOL:  str  = os.getenv("SYMBOLS", "BTC/USDT").split(",")[0].strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [M5] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("M5")

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _date_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _to_float(val, default: float = 0.0) -> float:
    """
    Convierte cualquier valor a float de forma segura.
    Maneja strings formateados de Google Sheets como '$66,729.58' o '0.00100000'.
    Nunca lanza excepción — retorna default si no puede convertir.
    """
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        try:
            return float(str(val).replace("$", "").replace(",", "").strip())
        except Exception:
            return default

def new_sale_id() -> str:
    return "SALE-" + uuid.uuid4().hex[:10].upper()


def _calc_pnl(
    qty_btc:       float,
    buy_price:     float,
    sell_price:    float,
    fee_buy_usdt:  float,
    fee_sell_usdt: float,
) -> Dict:
    """
    Calcula PnL de una venta usando FIFO.

    costo_usdt   = qty x precio_compra + fee_compra_proporcional
    ingreso_usdt = qty x precio_venta  - fee_venta
    pnl_usdt     = ingreso - costo
    pnl_pct      = pnl / costo x 100
    """
    costo_usdt   = (qty_btc * buy_price)  + fee_buy_usdt
    ingreso_usdt = (qty_btc * sell_price) - fee_sell_usdt
    pnl_usdt     = ingreso_usdt - costo_usdt
    pnl_pct      = (pnl_usdt / costo_usdt * 100) if costo_usdt > 0 else 0.0

    return {
        "costo_usdt":   round(costo_usdt,   4),
        "ingreso_usdt": round(ingreso_usdt,  4),
        "pnl_usdt":     round(pnl_usdt,      4),
        "pnl_pct":      round(pnl_pct,       4),
        "result":       "WIN" if pnl_usdt > 0 else ("LOSS" if pnl_usdt < 0 else "NEUTRAL"),
    }


def _holding_hours(date_open: str) -> Optional[float]:
    """Calcula horas desde apertura del lote hasta ahora."""
    try:
        opened = datetime.fromisoformat(str(date_open))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - opened
        return round(delta.total_seconds() / 3600, 2)
    except:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 2 — post_buy()
# ──────────────────────────────────────────────────────────────────────────────

def post_buy(
    execution: Dict,
    decision:  Dict,
    m1,
) -> Dict:
    """
    Registra una compra ejecutada en Google Sheets.
    Crea lote nuevo en la hoja 'lotes' con todos los campos canonicos.
    Si dry_run=True, loguea sin escribir en Sheets.
    """
    exe = execution
    if not exe.get("ok"):
        log.warning(f"post_buy ignorado — ejecucion fallida: {exe.get('error')}")
        return {"ok": False, "reason": "execution_failed"}

    lot_id       = m1.new_lot_id()
    filled_qty   = _to_float(exe.get("filled_qty", 0))
    avg_price    = _to_float(exe.get("avg_price",  0))
    fee_usdt     = _to_float(exe.get("fee_usdt",   0))
    capitalizado = round(filled_qty * avg_price + fee_usdt, 4)
    now_str      = _utc_now_str()
    date_str     = _date_str()

    lote_row = {
        "lot_id":            lot_id,
        "order_id":          str(exe.get("order_id",      "")),
        "client_order_id":   str(exe.get("execution_id",  "")),
        "bucket_id":         "1",
        "bucket_label":      "OPEN",
        "bucket_version":    "1",
        "last_touch_ts":     now_str,
        "source_route":      decision.get("rule_id", ""),
        "trade_ids":         "",
        "date":              date_str,
        "datetime":          now_str,
        "symbol":            SYMBOL,
        "side":              "buy",
        "type":              "market",
        "qty_inicial_btc":   str(filled_qty),
        "qty_restante_btc":  str(filled_qty),
        "price_usdt":        str(avg_price),
        "notional_usdt":     str(round(filled_qty * avg_price, 4)),
        "fee_amount":        str(fee_usdt),
        "fee_asset":         str(exe.get("fee_asset", "USDT")),
        "fee_usdt":          str(fee_usdt),
        "capitalizado_usdt": str(capitalizado),
        "exchange":          "binance",
        "status":            "OPEN",
        "policy":            decision.get("rule_id", ""),
        "comentario":        f"reg={decision.get('regimen','')} dir={decision.get('direccion','')} sc={decision.get('score_compra','')} sig={decision.get('signal_strength','')}",
        "decision_id":       decision.get("decision_id", ""),
        "rule_id_trigger":   decision.get("rule_id", ""),
        "ai_model_id":       "",
        # ── Contexto v2.0 — régimen, scores, dirección (para BiLSTM)
        "regimen":           decision.get("regimen", ""),
        "direccion":         str(decision.get("direccion", "")),
        "conviccion":        str(decision.get("conviccion", "")),
        "score_compra":      str(decision.get("score_compra", "")),
        "veredicto_auditor": decision.get("veredicto_auditor", ""),
    }

    if DRY_RUN or exe.get("dry_run"):
        log.info(
            f"[DRY RUN] post_buy: lote {lot_id} · "
            f"{filled_qty:.8f} BTC @ ${avg_price:,.2f} · cap=${capitalizado:.2f}"
        )
        return {"ok": True, "lot_id": lot_id, "dry_run": True, "lote_row": lote_row}

    try:
        m1.append_row_dict("lotes", lote_row)
        m1.write_log("INFO", f"Lote creado: {lot_id} · {filled_qty:.8f} BTC @ ${avg_price:,.2f}")
        log.info(f"Lote {lot_id} registrado en Sheets")
        return {"ok": True, "lot_id": lot_id, "dry_run": False}
    except Exception as e:
        log.error(f"Error escribiendo lote en Sheets: {e}")
        return {"ok": False, "reason": str(e), "lot_id": lot_id}


# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 3 — post_sell()
# ──────────────────────────────────────────────────────────────────────────────

def post_sell(
    execution:      Dict,
    decision:       Dict,
    lot:            Dict,
    m1,
    m3_log_outcome,
) -> Dict:
    """
    Registra una venta — actualiza inventario y calcula PnL.

    Flujo:
      1. Calcula qty vendida y qty restante
      2. Calcula PnL (FIFO con fee proporcional)
      3. Si qty_restante <= 0 → cierra el lote (status=CLOSED)
      4. Escribe ventas y ventas_detalle en Sheets
      5. Actualiza lote en Sheets
      6. Llama log_outcome() para completar el .jsonl
    """
    exe = execution
    if not exe.get("ok"):
        log.warning(f"post_sell ignorado — ejecucion fallida: {exe.get('error')}")
        return {"ok": False, "reason": "execution_failed"}

    sale_id      = new_sale_id()
    # exe viene de M4 — valores ya numéricos, _to_float por seguridad defensiva
    filled_qty   = _to_float(exe.get("filled_qty",       0))
    sell_price   = _to_float(exe.get("avg_price",        0))
    fee_sell     = _to_float(exe.get("fee_usdt",         0))
    # lot viene de Sheets vía df_inventory — puede traer strings formateados '$66,729.58'
    lot_id       = str(lot.get("lot_id",             ""))
    buy_price    = _to_float(lot.get("price_usdt",       0))
    qty_inicial  = _to_float(lot.get("qty_inicial_btc",  0))
    qty_restante = _to_float(lot.get("qty_restante_btc", 0))
    fee_buy_tot  = _to_float(lot.get("fee_usdt",         0))
    date_open    = str(lot.get("date",               ""))
    now_str      = _utc_now_str()
    date_str     = _date_str()

    # Validación: si datos críticos son cero después de conversión, loguear y fallar rápido
    if buy_price <= 0:
        log.error(f"post_sell: buy_price inválido para lote {lot_id} — valor raw='{lot.get('price_usdt')}'")
        return {"ok": False, "reason": f"buy_price_invalido:{lot.get('price_usdt')}", "lot_id": lot_id}
    if qty_restante <= 0:
        log.error(f"post_sell: qty_restante=0 para lote {lot_id} — valor raw='{lot.get('qty_restante_btc')}'")
        return {"ok": False, "reason": f"qty_restante_cero:{lot.get('qty_restante_btc')}", "lot_id": lot_id}
    if filled_qty <= 0:
        log.error(f"post_sell: filled_qty=0 en execution para lote {lot_id}")
        return {"ok": False, "reason": "filled_qty_cero", "lot_id": lot_id}

    # Fee de compra proporcional a la qty vendida
    fee_buy_prop = (fee_buy_tot * filled_qty / qty_inicial) if qty_inicial > 0 else 0.0

    # PnL de esta venta
    pnl = _calc_pnl(filled_qty, buy_price, sell_price, fee_buy_prop, fee_sell)

    # Nueva qty restante
    new_qty_rest = max(0.0, qty_restante - filled_qty)
    lote_cerrado = new_qty_rest <= 0.0001
    holding_h    = _holding_hours(date_open)

    venta_row = {
        "sale_id":            sale_id,
        "order_id":           str(exe.get("order_id",      "")),
        "client_order_id":    str(exe.get("execution_id",  "")),
        "date":               date_str,
        "datetime":           now_str,
        "symbol":             SYMBOL,
        "side":               "sell",
        "type":               "market",
        "avg_price_usdt":     str(sell_price),
        "qty_total_btc":      str(filled_qty),
        "ingreso_bruto_usdt": str(round(filled_qty * sell_price, 4)),
        "fee_total_amount":   str(fee_sell),
        "fee_total_asset":    str(exe.get("fee_asset", "USDT")),
        "fee_total_usdt":     str(fee_sell),
        "costo_asignado_usdt":str(pnl["costo_usdt"]),
        "pnl_usdt":           str(pnl["pnl_usdt"]),
        "pnl_pct":            str(pnl["pnl_pct"]),
        "exchange":           "binance",
        "fills_count":        str(exe.get("fills_count", 1)),
        "lotes_involucrados": lot_id,
        "policy":             decision.get("rule_id", ""),
        "catalyst":           decision.get("signal_strength", ""),
        "status":             "CLOSED" if lote_cerrado else "PARTIAL",
        "comment_full":       f"rule={decision.get('rule_id','')} pnl={pnl['pnl_pct']:.2f}% result={pnl['result']}",
        "decision_id":        decision.get("decision_id", ""),
        "rule_id_trigger":    decision.get("rule_id", ""),
        "risk_state":         decision.get("leverage_risk", "low"),
        "snapshot_id":        decision.get("snapshot_id", ""),
        "latency_ms":         "",
        # ── Contexto de decisión v2.0
        "regimen":            decision.get("regimen", ""),
        "direccion":          str(decision.get("direccion", "")),
        "conviccion":         str(decision.get("conviccion", "")),
        "score_venta":        str(decision.get("score_venta", "")),
        "holding_hours":      str(holding_h),
    }

    detalle_row = {
        "sale_id":          sale_id,
        "lot_id":           lot_id,
        "qty_usada_btc":    str(filled_qty),
        "costo_unit_usdt":  str(buy_price),
        "costo_usdt":       str(pnl["costo_usdt"]),
        "sold_price_usdt":  str(sell_price),
        "ingreso_usdt":     str(pnl["ingreso_usdt"]),
        "fee_prorrata_usdt":str(round(fee_buy_prop + fee_sell, 4)),
        "pnl_usdt":         str(pnl["pnl_usdt"]),
        "pnl_pct":          str(pnl["pnl_pct"]),
        "policy_venta":     decision.get("rule_id", ""),
        "decision_id":      decision.get("decision_id", ""),
        "rule_id_trigger":  decision.get("rule_id", ""),
        "comment_short":    f"{pnl['result']} {pnl['pnl_pct']:.2f}%",
    }

    if DRY_RUN or exe.get("dry_run"):
        log.info(
            f"[DRY RUN] post_sell: {lot_id} · {filled_qty:.8f} BTC @ ${sell_price:,.2f} "
            f"· PnL={pnl['pnl_pct']:.2f}% ({pnl['result']}) "
            f"· {'CERRADO' if lote_cerrado else 'PARCIAL'}"
        )
        m3_log_outcome(decision.get("decision_id", ""), {
            "closed_at":     now_str,
            "close_price":   sell_price,
            "pnl_usdt":      pnl["pnl_usdt"],
            "pnl_pct":       pnl["pnl_pct"],
            "holding_hours": holding_h,
            "result":        pnl["result"],
            "lot_closed":    lote_cerrado,
            "dry_run":       True,
        })
        return {
            "ok":           True,
            "dry_run":      True,
            "sale_id":      sale_id,
            "lot_id":       lot_id,
            "pnl_usdt":     pnl["pnl_usdt"],
            "pnl_pct":      pnl["pnl_pct"],
            "result":       pnl["result"],
            "lot_closed":   lote_cerrado,
            "new_qty_rest": new_qty_rest,
            "holding_hours":holding_h,
        }

    # ESCRITURAS REALES EN SHEETS
    try:
        m1.append_row_dict("ventas", venta_row)
        m1.append_row_dict("ventas_detalle", detalle_row)
        m1.snapshot_lote_history(lot, event="PRE_SELL")

        if lote_cerrado:
            m1.update_lote_fields(
                lot_id,
                status           = "CLOSED",
                qty_restante_btc = "0",
                last_touch_ts    = now_str,
                comentario       = f"Cerrado {decision.get('rule_id','')} PnL={pnl['pnl_pct']:.2f}%",
            )
            m1.snapshot_lote_history({**lot, "status": "CLOSED"}, event="POST_SELL_CLOSED")
        else:
            # Venta PARCIAL — reducir qty_restante en Sheets
            m1.update_lote_fields(
                lot_id,
                qty_restante_btc = str(new_qty_rest),
                last_touch_ts    = now_str,
                status           = "OPEN",  # sigue abierto pero con menos qty
                comentario       = f"Parcial {decision.get('rule_id','')} resta={new_qty_rest:.8f} vendido={filled_qty:.8f}",
            )
            m1.snapshot_lote_history(
                {**lot, "qty_restante_btc": new_qty_rest, "status": "OPEN"},
                event="POST_SELL_PARTIAL"
            )

        m1.write_log("INFO",
            f"Venta {sale_id}: {lot_id} {filled_qty:.8f} BTC @ ${sell_price:,.2f} "
            f"PnL={pnl['pnl_pct']:.2f}% ({pnl['result']})"
        )
        m3_log_outcome(decision.get("decision_id", ""), {
            "closed_at":     now_str,
            "close_price":   sell_price,
            "pnl_usdt":      pnl["pnl_usdt"],
            "pnl_pct":       pnl["pnl_pct"],
            "holding_hours": holding_h,
            "result":        pnl["result"],
            "lot_closed":    lote_cerrado,
            "dry_run":       False,
        })
        estado = "CERRADO" if lote_cerrado else f"PARCIAL (resta {new_qty_rest:.8f} BTC)"
        log.info(
            f"Venta {sale_id} registrada: {lot_id} PnL={pnl['pnl_pct']:.2f}% "
            f"({pnl['result']}) {estado}"
        )
        if lote_cerrado:
            log.info(f"✅ SELL PnL={pnl['pnl_pct']:.2f}% ({pnl['result']}) CERRADO")
        else:
            log.info(f"⚡ SELL PARCIAL PnL={pnl['pnl_pct']:.2f}% — resta {new_qty_rest:.8f} BTC")
        return {
            "ok":           True,
            "dry_run":      False,
            "sale_id":      sale_id,
            "lot_id":       lot_id,
            "pnl_usdt":     pnl["pnl_usdt"],
            "pnl_pct":      pnl["pnl_pct"],
            "result":       pnl["result"],
            "lot_closed":   lote_cerrado,
            "new_qty_rest": new_qty_rest,
            "holding_hours":holding_h,
            # Para que el loop sepa exactamente qué pasó
            "was_partial":  not lote_cerrado,
            "qty_vendida":  filled_qty,
            "qty_restante": new_qty_rest,
        }
    except Exception as e:
        log.error(f"Error en post_sell escribiendo Sheets: {e}")
        return {"ok": False, "reason": str(e), "sale_id": sale_id}


# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — Accountant class
# ──────────────────────────────────────────────────────────────────────────────

class Accountant:
    """
    Orquestador de M5.

    Uso desde M6:
        accountant = Accountant(m1_module, log_outcome_fn)
        result     = accountant.run(execution, decision, df_inventory)
    """

    def __init__(self, m1, m3_log_outcome):
        self.m1             = m1
        self.m3_log_outcome = m3_log_outcome
        log.info(f"Accountant iniciado · dry_run={DRY_RUN}")

    def _find_lot(self, df_inventory, lot_id: str) -> Optional[Dict]:
        if df_inventory is None or df_inventory.empty:
            return None
        rows = df_inventory[df_inventory["lot_id"].astype(str) == str(lot_id)]
        return rows.iloc[0].to_dict() if not rows.empty else None

    def heartbeat(self, snapshot_id: str, df_inventory, budget_reads_left: int,
                  comment: str = "", regimen: str = "", direccion: float = 0.0,
                  pnl_dia_pct: float = 0.0) -> None:
        """Escribe pulso en status_hb. Llamar al final de cada loop de M6."""
        if DRY_RUN:
            return
        try:
            comment_full = (
                f"{comment} | reg={regimen} dir={direccion:+.0f} pnl_dia={pnl_dia_pct:+.2f}%"
                if regimen else comment
            )
            self.m1.write_status_hb(snapshot_id, df_inventory, budget_reads_left, comment_full)
        except Exception as e:
            log.warning(f"Error escribiendo heartbeat: {e}")

    def run(self, execution: Dict, decision: Dict, df_inventory) -> Dict:
        """
        Punto de entrada principal — llamar desde M6 despues de M4.
        Retorna dict con resultado contable.
        """
        if not execution.get("ok"):
            log.warning(f"M5.run ignorado — ejecucion fallida: {execution.get('error')}")
            return {"ok": False, "reason": "execution_failed"}

        action = execution.get("action", decision.get("action", ""))

        if action == "BUY":
            result = post_buy(execution, decision, self.m1)
            if result.get("ok"):
                log.info(
                    f"{'[DRY RUN] ' if result.get('dry_run') else ''}"
                    f"Lote {result.get('lot_id')} registrado"
                )
            return result

        elif action in ("SELL_V1", "SELL_V2", "SELL_INTRA"):
            # lot_id debe venir de la decisión — es la fuente de verdad
            lot_id = decision.get("lot_id") or execution.get("lot_id")
            if not lot_id:
                return {"ok": False, "reason": "no_lot_id"}
            lot = self._find_lot(df_inventory, lot_id)
            if not lot:
                log.error(f"Lote {lot_id} no encontrado en inventario")
                return {"ok": False, "reason": f"lot_not_found:{lot_id}"}
            result = post_sell(execution, decision, lot, self.m1, self.m3_log_outcome)
            if result.get("ok"):
                emoji = "✅" if result.get("result") == "WIN" else "❌"
                log.info(
                    f"{'[DRY RUN] ' if result.get('dry_run') else ''}"
                    f"{emoji} {action} PnL={result.get('pnl_pct',0):.2f}% "
                    f"({result.get('result')}) "
                    f"{'CERRADO' if result.get('lot_closed') else 'PARCIAL'}"
                )
            return result

        return {"ok": False, "reason": f"action_desconocida:{action}"}


# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 5 — Ejecucion directa (prueba)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*72)
    print("  M5 - CONTABILIDAD  ·  Prueba con datos simulados")
    print("="*72)

    class MockM1:
        def new_lot_id(self):
            return "LOT-" + uuid.uuid4().hex[:10].upper()
        def append_row_dict(self, sheet, row):
            log.info(f"[MOCK] append → {sheet}: {row.get('lot_id') or row.get('sale_id','?')}")
        def update_lote_fields(self, lot_id, **fields):
            log.info(f"[MOCK] update_lote → {lot_id}: {list(fields.keys())}")
        def snapshot_lote_history(self, lot, event):
            log.info(f"[MOCK] snapshot → {lot.get('lot_id','?')} event={event}")
        def write_log(self, level, msg):
            log.info(f"[MOCK] write_log [{level}] {msg}")
        def write_status_hb(self, *a, **k):
            pass

    outcomes = []
    def mock_log_outcome(decision_id, outcome):
        outcomes.append({"decision_id": decision_id, **outcome})
        log.info(f"[MOCK] outcome → {decision_id} result={outcome.get('result')} pnl={outcome.get('pnl_pct'):.2f}%")

    accountant = Accountant(MockM1(), mock_log_outcome)

    df_inv = pd.DataFrame([{
        "lot_id":            "LOT-TEST001",
        "qty_inicial_btc":   0.001,
        "qty_restante_btc":  0.001,
        "price_usdt":        68000.0,
        "fee_usdt":          0.07,
        "capitalizado_usdt": 68.07,
        "status":            "OPEN",
        "date":              "2026-03-20 10:00:00",
    }])

    dec = {
        "decision_id":    "DEC-TEST",
        "rule_id":        "V1-TP2",
        "lot_id":         "LOT-TEST001",
        "signal_strength":"strong",
        "inv_state":      "balanced",
        "leverage_risk":  "low",
        "snapshot_id":    "SNP-TEST",
    }

    # Test 1: BUY
    print("\n  -- Test 1: Post-compra")
    exe_buy = {"ok":True,"dry_run":True,"action":"BUY","execution_id":"EXE-B1",
               "filled_qty":0.00071,"avg_price":70400.0,"fee_usdt":0.05,"fee_asset":"USDT"}
    r = accountant.run(exe_buy, {**dec, "action":"BUY","rule_id":"BUY-DIP-STRONG"}, df_inv)
    print(f"  ok={r['ok']} lot_id={r.get('lot_id')} dry_run={r.get('dry_run')}")

    # Test 2: SELL_V1 parcial
    print("\n  -- Test 2: Post-venta parcial (V1-TP2 · 40%)")
    exe_sell = {"ok":True,"dry_run":True,"action":"SELL_V1","execution_id":"EXE-S1",
                "lot_id":"LOT-TEST001","filled_qty":0.0004,"avg_price":70570.0,
                "fee_usdt":0.028,"fee_asset":"USDT","sell_pct":40.0}
    r = accountant.run(exe_sell, {**dec,"action":"SELL_V1"}, df_inv)
    print(f"  ok={r['ok']} pnl={r.get('pnl_pct',0):.2f}% result={r.get('result')} cerrado={r.get('lot_closed')}")

    # Test 3: SELL_V2 total (SL)
    print("\n  -- Test 3: Post-venta total (V2-SL · 100%)")
    exe_sl = {"ok":True,"dry_run":True,"action":"SELL_V2","execution_id":"EXE-SL",
              "lot_id":"LOT-TEST001","filled_qty":0.001,"avg_price":66640.0,
              "fee_usdt":0.067,"fee_asset":"USDT","sell_pct":100.0}
    r = accountant.run(exe_sl, {**dec,"action":"SELL_V2","rule_id":"V2-SL-NORMAL"}, df_inv)
    print(f"  ok={r['ok']} pnl={r.get('pnl_pct',0):.2f}% result={r.get('result')} cerrado={r.get('lot_closed')}")

    print(f"\n  Outcomes para IA: {len(outcomes)}")
    for o in outcomes:
        print(f"    {o['decision_id']} result={o['result']} pnl={o['pnl_pct']:.2f}% holding={o.get('holding_hours','?')}h")

    print("\n  M5 prueba completa.\n")
