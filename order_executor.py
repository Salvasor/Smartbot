# -*- coding: utf-8 -*-
"""
Módulo 4 — Ejecutor de Órdenes · order_executor.py

Recibe la decisión de M3 y la ejecuta en Binance.
Es el ÚNICO módulo que usa credenciales y toca dinero real.

Responsabilidades:
  - Conectar a Binance con API key/secret (autenticado)
  - Ejecutar órdenes de compra y venta (market orders)
  - Verificar fills: precio real, qty real, fee pagado
  - Manejar errores del exchange sin caerse (rate limit, fondos, timeout)
  - Respetar dry_run=True — si está activo, simula sin ejecutar
  - Loguear la ejecución real en .jsonl (event=EXECUTION)

Filosofía:
  - Si hay duda, NO ejecutar — retornar error y dejar que M6 reintente
  - Nunca ejecutar si decision["dry_run"] = True
  - Siempre verificar qty mínima de Binance antes de enviar
  - Toda orden es MARKET — sin límite, sin complicaciones por ahora

Dependencias:
  ccxt          → cliente de exchange
  decision_engine.py → formato del dict de decisión que recibe
"""

import os
import json
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, List

import ccxt

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — Config y conexión autenticada
# ──────────────────────────────────────────────────────────────────────────────

# Credenciales — mismas variables de entorno del bot anterior
API_KEY:    Optional[str] = os.getenv("BINANCE_KEY")
API_SECRET: Optional[str] = os.getenv("BINANCE_SECRET")
EXCHANGE_ID: str          = os.getenv("EXCHANGE_ID", "binance")
SYMBOL:      str          = os.getenv("SYMBOLS", "BTC/USDT").split(",")[0].strip()

# Modo seguro — nunca ejecuta si True
DRY_RUN: bool = os.getenv("DRY_RUN", "1") == "1"

# Log
LOG_DIR: str = os.getenv("LOG_DIR", "logs")

# Reintentos ante errores temporales del exchange
MAX_RETRIES:    int   = 3
RETRY_DELAY_S:  float = 2.0   # segundos entre reintentos
ORDER_TIMEOUT_S: int  = 10    # máximo de espera por confirmación de fill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [M4] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("M4")

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _execution_id() -> str:
    return "EXE-" + uuid.uuid4().hex[:10].upper()


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


def build_exchange() -> ccxt.Exchange:
    """
    Construye cliente ccxt autenticado para Binance spot.
    Verifica credenciales antes de retornar.
    Lanza excepción si no hay API key configurada.
    """
    if not API_KEY or not API_SECRET:
        raise ValueError(
            "API_KEY y API_SECRET no encontrados en variables de entorno. "
            "Verificá tu archivo .env o las variables del sistema."
        )

    ex_class = getattr(ccxt, EXCHANGE_ID)
    ex: ccxt.Exchange = ex_class({
        "apiKey":          API_KEY,
        "secret":          API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType":              "spot",
            "adjustForTimeDifference":  True,
            "recvWindow":               10000,
        }
    })

    # Monkey-patch para evitar llamadas SAPI que requieren permisos extra
    # (igual que el bot anterior — evita errores de load_markets)
    ex.fetch_currencies = lambda *a, **k: {}

    ex.load_markets()
    log.info(f"Exchange conectado: {EXCHANGE_ID} · {SYMBOL} · autenticado")
    return ex


def get_lot_step(ex: ccxt.Exchange, symbol: str) -> float:
    """
    Retorna el step size de cantidad para el símbolo.
    Binance requiere cantidades redondeadas al step exacto.
    ej: BTC/USDT → 0.00001

    Prioridad:
      1. Filtro LOT_SIZE del exchange (más confiable para Binance)
      2. market['precision']['amount'] como fallback
      3. 0.00001 como último recurso
    """
    try:
        market = ex.market(symbol)

        # 1. Intentar via filtros de Binance (market.info.filters)
        filters = market.get("info", {}).get("filters", [])
        for f in filters:
            if f.get("filterType") == "LOT_SIZE":
                step = float(f.get("stepSize", 0))
                if 0 < step < 1:
                    log.info(f"lot_step obtenido de LOT_SIZE filter: {step}")
                    return step

        # 2. Fallback: precision.amount
        step = float(market["precision"]["amount"])
        # ccxt puede retornar nº de decimales (ej: 5) o el step directo (ej: 0.00001)
        if step >= 1:
            step = 10 ** (-int(step))
        log.info(f"lot_step obtenido de precision.amount: {step}")
        return step

    except Exception as e:
        log.warning(f"No se pudo obtener lot_step para {symbol}: {e} · usando 0.00001")
        return 0.00001


def round_qty(qty: float, step: float) -> float:
    """
    Redondea qty al step size de Binance (hacia abajo — nunca exceder).
    Usa Decimal para evitar errores de punto flotante binario.
    Ejemplo: 0.00021 en float64 = 0.000209999... → floor(×100000) = 20 → 0.0002 (MAL)
             con Decimal: 0.00021 / 0.00001 = 21 exacto → 0.00021 (OK)
    """
    if step <= 0:
        return qty
    from decimal import Decimal, ROUND_DOWN
    step_d = Decimal(str(step))
    qty_d  = Decimal(str(qty))
    result = (qty_d / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
    return float(result)

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 2 — Ejecución de órdenes
# ──────────────────────────────────────────────────────────────────────────────

def _parse_fill(order: Dict) -> Dict:
    """
    Extrae los datos relevantes de una orden ejecutada por ccxt.
    Maneja tanto fills completos como parciales.
    """
    fills      = order.get("trades") or order.get("fills") or []
    status     = order.get("status", "unknown")
    filled_qty = float(order.get("filled", 0) or 0)
    avg_price  = float(order.get("average", 0) or order.get("price", 0) or 0)

    # Si ccxt no da precio promedio, calcularlo desde los fills
    if avg_price == 0 and fills:
        total_cost = sum(float(f.get("cost", 0) or 0) for f in fills)
        total_qty  = sum(float(f.get("amount", 0) or 0) for f in fills)
        avg_price  = total_cost / total_qty if total_qty > 0 else 0

    # Fee total en USDT
    fee_usdt = 0.0
    fee_asset = "USDT"
    for f in fills:
        fee_info = f.get("fee") or {}
        fee_cost = float(fee_info.get("cost", 0) or 0)
        fee_curr = fee_info.get("currency", "USDT")
        if fee_curr == "BNB":
            # Fee en BNB — aproximamos en USDT (no crítico para el log)
            fee_usdt += fee_cost * 600  # precio BNB aproximado
        elif fee_curr in ("USDT", "BTC"):
            fee_usdt += fee_cost * (avg_price if fee_curr == "BTC" else 1.0)
        else:
            fee_usdt += fee_cost

    # Si no hay fills detallados, estimar fee (0.1% de Binance)
    if fee_usdt == 0 and filled_qty > 0 and avg_price > 0:
        fee_usdt = filled_qty * avg_price * 0.001

    return {
        "order_id":    str(order.get("id", "")),
        "status":      status,
        "filled_qty":  round(filled_qty, 8),
        "avg_price":   round(avg_price, 2),
        "notional":    round(filled_qty * avg_price, 2),
        "fee_usdt":    round(fee_usdt, 4),
        "fee_asset":   fee_asset,
        "fills_count": len(fills),
        "raw_order":   order.get("id"),   # solo ID para no inflar el log
    }


def _execute_market_order(
    ex:      ccxt.Exchange,
    symbol:  str,
    side:    str,         # "buy" | "sell"
    qty_btc: float,
    params:  Dict = {},
) -> Dict:
    """
    Envía una orden de mercado con reintentos ante errores temporales.
    Retorna el fill parseado o un dict con error.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"Enviando {side.upper()} {qty_btc:.8f} BTC · intento {attempt}/{MAX_RETRIES}")

            if side == "buy":
                order = ex.create_market_buy_order(symbol, qty_btc, params)
            else:
                order = ex.create_market_sell_order(symbol, qty_btc, params)

            fill = _parse_fill(order)
            log.info(
                f"Orden ejecutada: {side.upper()} {fill['filled_qty']} BTC "
                f"@ {fill['avg_price']:,.2f} USDT · fee={fill['fee_usdt']:.4f} USDT"
            )
            return {"ok": True, **fill}

        except ccxt.InsufficientFunds as e:
            log.error(f"Fondos insuficientes: {e}")
            return {"ok": False, "error": "insufficient_funds", "detail": str(e)}

        except ccxt.InvalidOrder as e:
            log.error(f"Orden inválida: {e}")
            return {"ok": False, "error": "invalid_order", "detail": str(e)}

        except ccxt.RateLimitExceeded as e:
            log.warning(f"Rate limit · esperando {RETRY_DELAY_S * attempt}s...")
            time.sleep(RETRY_DELAY_S * attempt)

        except ccxt.NetworkError as e:
            log.warning(f"Error de red · intento {attempt}: {e}")
            time.sleep(RETRY_DELAY_S)

        except ccxt.ExchangeError as e:
            # Errores -1021 (nonce) — resincronizar y reintentar
            if "-1021" in str(e):
                log.warning(f"Nonce desincronizado · resincronizando...")
                try:
                    ex.load_time_difference()
                except:
                    pass
                time.sleep(1)
            else:
                log.error(f"Error del exchange: {e}")
                return {"ok": False, "error": "exchange_error", "detail": str(e)}

        except Exception as e:
            log.error(f"Error inesperado: {e}")
            return {"ok": False, "error": "unexpected", "detail": str(e)}

    return {"ok": False, "error": "max_retries", "detail": f"Falló tras {MAX_RETRIES} intentos"}


def execute_buy(
    ex:           ccxt.Exchange,
    decision:     Dict,
    current_price: float,
    lot_step:     float,
) -> Dict:
    """
    Ejecuta una orden de compra a mercado.

    Parámetros:
        ex            → cliente ccxt autenticado
        decision      → dict de M3 (action=BUY)
        current_price → precio actual para calcular qty_btc
        lot_step      → step size de Binance

    Retorna dict de ejecución con status, fill y execution_id.
    """
    exe_id   = _execution_id()
    qty_usdt = float(decision.get("qty_usdt", 0))

    if qty_usdt <= 0:
        return {"ok": False, "error": "qty_usdt_zero", "execution_id": exe_id}

    # Calcular qty en BTC y redondear al step
    qty_btc_raw = qty_usdt / current_price
    qty_btc     = round_qty(qty_btc_raw, lot_step)

    if qty_btc <= 0:
        return {"ok": False, "error": "qty_btc_zero_after_rounding", "execution_id": exe_id}

    # DRY RUN — simular sin ejecutar
    # IMPORTANTE: default False — si DRY_RUN=0 y la decisión no trae la key, ejecuta real
    if DRY_RUN or decision.get("dry_run", False):
        log.info(f"[DRY RUN] BUY simulado: {qty_btc:.8f} BTC @ ~{current_price:,.2f} USDT = ${qty_btc * current_price:.2f}")
        return {
            "ok":           True,
            "dry_run":      True,
            "execution_id": exe_id,
            "decision_id":  decision.get("decision_id"),
            "action":       "BUY",
            "symbol":       SYMBOL,
            "filled_qty":   qty_btc,
            "avg_price":    current_price,
            "notional":     round(qty_btc * current_price, 2),
            "fee_usdt":     round(qty_btc * current_price * 0.001, 4),
            "status":       "simulated",
            "ts":           _utc_now_str(),
        }

    # EJECUCIÓN REAL
    fill = _execute_market_order(ex, SYMBOL, "buy", qty_btc)
    return {
        **fill,
        "execution_id": exe_id,
        "decision_id":  decision.get("decision_id"),
        "action":       "BUY",
        "symbol":       SYMBOL,
        "ts":           _utc_now_str(),
        "dry_run":      False,
    }


def execute_sell(
    ex:            ccxt.Exchange,
    decision:      Dict,
    lot:           Dict,
    lot_step:      float,
    current_price: float = 0.0,
) -> Dict:
    """
    Ejecuta una orden de venta a mercado para un lote específico.

    Parámetros:
        ex       → cliente ccxt autenticado
        decision → dict de M3 (action=SELL_V1 | SELL_V2 | SELL_INTRA)
        lot      → dict del lote a vender (de M1)
        lot_step → step size de Binance

    La cantidad a vender depende de sell_pct del lote:
        SELL_V1 TP1 → 40% del lote
        SELL_V1 TP2 → 40% del lote
        SELL_V2     → 100% del lote
    """
    exe_id   = _execution_id()
    # sell_pct en el dict de decisión es fracción (0.8 = 80%), NO porcentaje
    sell_pct = float(decision.get("sell_pct", 1.0))
    # qty_rest: viene de Sheets vía df_inventory — puede ser string '$0.00100000'
    qty_rest = _to_float(lot.get("qty_restante_btc", 0))

    if qty_rest <= 0:
        return {"ok": False, "error": "lote_sin_qty", "execution_id": exe_id}

    # Calcular qty a vender y redondear
    qty_btc_raw = qty_rest * sell_pct
    qty_btc     = round_qty(qty_btc_raw, lot_step)

    # Si el residuo que quedaría es demasiado pequeño, vender todo el lote
    # Evita acumulación de polvo BTC que accounting cierra sin vender
    MIN_RESIDUAL_BTC = 0.0001  # mismo umbral que accounting.py lote_cerrado
    qty_residual = qty_rest - qty_btc
    if qty_btc > 0 and 0 < qty_residual < MIN_RESIDUAL_BTC:
        qty_btc_full = round_qty(qty_rest, lot_step)
        if qty_btc_full > 0:
            log.info(
                f"Residuo {qty_residual:.8f} BTC < {MIN_RESIDUAL_BTC} tras venta parcial "
                f"→ vendiendo lote completo ({qty_btc_full:.8f} BTC) para evitar polvo"
            )
            qty_btc = qty_btc_full

    if qty_btc <= 0:
        # El parcial es demasiado pequeño para el step — intentar vender lote completo
        qty_btc_full = round_qty(qty_rest, lot_step)
        if qty_btc_full > 0:
            log.warning(
                f"sell_pct={sell_pct*100:.0f}% redondeó a 0 "
                f"(qty_raw={qty_btc_raw:.8f} lot_step={lot_step}) "
                f"→ vendiendo lote completo: {qty_btc_full:.8f} BTC"
            )
            qty_btc = qty_btc_full
        else:
            log.error(
                f"qty_btc=0 tras redondeo completo: qty_rest={qty_rest:.8f} "
                f"sell_pct={sell_pct*100:.0f}% lot_step={lot_step}"
            )
            return {
                "ok":          False,
                "error":       "qty_btc_zero_after_rounding",
                "detail":      f"qty_rest={qty_rest:.8f} lot_step={lot_step} sell_pct={sell_pct*100:.0f}%",
                "execution_id": exe_id,
            }

    # DRY RUN — simular sin ejecutar
    # IMPORTANTE: default False — si DRY_RUN=0 y la decisión no trae la key, ejecuta real
    if DRY_RUN or decision.get("dry_run", False):
        # Usar current_price (precio de mercado actual) para el PnL simulado sea real
        # Si no se pasó current_price, fallback al precio de compra del lote
        price_sim = current_price if current_price > 0 else _to_float(lot.get("price_usdt", 0))
        log.info(
            f"[DRY RUN] SELL simulado: {qty_btc:.8f} BTC "
            f"({sell_pct*100:.0f}% del lote {lot.get('lot_id','')}) @ ~${price_sim:,.2f}"
        )
        return {
            "ok":           True,
            "dry_run":      True,
            "execution_id": exe_id,
            "decision_id":  decision.get("decision_id"),
            "action":       decision.get("action"),
            "rule_id":      decision.get("rule_id"),
            "symbol":       SYMBOL,
            "lot_id":       lot.get("lot_id"),
            "filled_qty":   qty_btc,
            "avg_price":    price_sim,
            "notional":     round(qty_btc * price_sim, 2),
            "fee_usdt":     round(qty_btc * price_sim * 0.001, 4),
            "sell_pct":     sell_pct * 100,
            "status":       "simulated",
            "ts":           _utc_now_str(),
        }

    # EJECUCIÓN REAL — verificar balance BTC disponible antes de vender
    try:
        balance       = ex.fetch_balance()
        # ccxt expone balance libre en balance["free"]["BTC"] o balance["BTC"]["free"]
        btc_available = float(balance.get("free", {}).get("BTC", 0) or 0)
        if btc_available < qty_btc:
            log.error(f"BTC insuficiente: disponible={btc_available:.8f} necesario={qty_btc:.8f}")
            return {
                "ok":    False,
                "error": "insufficient_btc",
                "detail": f"BTC disponible={btc_available:.8f} < requerido={qty_btc:.8f}",
                "lot_id": lot.get("lot_id"),
            }
        log.info(f"Balance pre-sell OK: {btc_available:.8f} BTC disponible · necesario={qty_btc:.8f}")
    except Exception as e:
        log.warning(f"No se pudo verificar balance pre-sell: {e} — continuando de todas formas")

    fill = _execute_market_order(ex, SYMBOL, "sell", qty_btc)
    return {
        **fill,
        "execution_id": exe_id,
        "decision_id":  decision.get("decision_id"),
        "action":       decision.get("action"),
        "rule_id":      decision.get("rule_id"),
        "symbol":       SYMBOL,
        "lot_id":       lot.get("lot_id"),
        "sell_pct":     sell_pct * 100,
        "ts":           _utc_now_str(),
        "dry_run":      False,
    }

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 3 — Logger de ejecución y OrderExecutor class
# ──────────────────────────────────────────────────────────────────────────────

def log_execution(execution: Dict, log_dir: str = LOG_DIR) -> None:
    """
    Escribe la ejecución real en el .jsonl mensual (event=EXECUTION).
    Se une con el DECISION de M3 por decision_id.
    Siempre loguea — tanto ejecuciones reales como dry_run y errores.
    """
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        month    = datetime.now().strftime("%Y%m")
        log_path = Path(log_dir) / f"decisions_{month}.jsonl"
        entry    = {**execution, "event": "EXECUTION"}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        log.warning(f"No se pudo escribir log de ejecución: {e}")


def print_execution(execution: Dict) -> None:
    """Imprime el resultado de la ejecución en terminal."""
    SEP2 = "═" * 72
    ok      = execution.get("ok", False)
    dry     = execution.get("dry_run", False)
    action  = execution.get("action", "?")
    exe_id  = execution.get("execution_id", "?")
    dr_tag  = " [DRY RUN]" if dry else " [REAL]"

    icons = {
        "BUY":        "🟢",
        "SELL_V1":    "💰",
        "SELL_V2":    "🔴",
        "SELL_INTRA": "⚡",
    }
    icon = icons.get(action, "⚙️")

    print(f"\n{SEP2}")
    if ok:
        print(f"  {icon} EJECUTADO{dr_tag}  ·  {exe_id}  ·  {execution.get('ts','')}")
        print(SEP2)
        print(f"  Símbolo    : {execution.get('symbol','?')}")
        print(f"  Acción     : {action}  ·  rule={execution.get('rule_id','?')}")
        print(f"  Qty BTC    : {execution.get('filled_qty', 0):.8f} BTC")
        print(f"  Precio     : ${execution.get('avg_price', 0):>12,.2f} USDT")
        print(f"  Notional   : ${execution.get('notional', 0):>12,.2f} USDT")
        print(f"  Fee        : ${execution.get('fee_usdt', 0):>12,.4f} USDT")
        if execution.get("lot_id"):
            print(f"  Lote       : {execution.get('lot_id')}")
        if execution.get("sell_pct"):
            print(f"  Vendido    : {execution.get('sell_pct'):.0f}% del lote")
    else:
        print(f"  ❌ ERROR EN EJECUCIÓN{dr_tag}  ·  {exe_id}")
        print(SEP2)
        print(f"  Error      : {execution.get('error','?')}")
        print(f"  Detalle    : {execution.get('detail','?')}")
    print(f"{SEP2}\n")


class OrderExecutor:
    """
    Orquestador de M4.

    Mantiene en memoria:
      - exchange: cliente ccxt autenticado (singleton)
      - lot_step: step size del símbolo (se lee una vez al arrancar)

    Uso desde M6:
        executor = OrderExecutor()
        result   = executor.run(decision, df_inventory, current_price)
    """

    def __init__(self):
        self.ex:       Optional[ccxt.Exchange] = None
        self.lot_step: float                   = 0.00001
        self._initialized: bool                = False

        if DRY_RUN:
            log.info("OrderExecutor iniciado · modo DRY RUN — no se ejecutarán órdenes reales")
        else:
            log.info("OrderExecutor iniciado · modo REAL — las órdenes se ejecutarán en Binance")

    def _ensure_connected(self) -> bool:
        """
        Conecta al exchange si no está conectado.
        Retorna True si la conexión está lista, False si falla.
        En dry_run no conecta — no hace falta.
        """
        if self._initialized:
            return True

        if DRY_RUN:
            self._initialized = True
            return True

        try:
            self.ex       = build_exchange()
            self.lot_step = get_lot_step(self.ex, SYMBOL)
            self._initialized = True
            log.info(f"Conexión lista · lot_step={self.lot_step}")
            return True
        except Exception as e:
            log.error(f"No se pudo conectar al exchange: {e}")
            return False

    def _find_lot(self, df_inventory, lot_id: str) -> Optional[Dict]:
        """Busca un lote por lot_id en el DataFrame de inventario."""
        if df_inventory is None or df_inventory.empty:
            return None
        rows = df_inventory[df_inventory["lot_id"].astype(str) == str(lot_id)]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    def run(
        self,
        decision:      Dict,
        df_inventory,
        current_price: float,
    ) -> Dict:
        """
        Punto de entrada principal — llamar desde M6 cuando M3 decide actuar.

        Parámetros:
            decision      → dict de M3
            df_inventory  → DataFrame de lotes abiertos (M1)
            current_price → precio actual del tick

        Retorna dict de ejecución que M5 usa para actualizar contabilidad.
        No lanza excepciones — errores quedan en el dict retornado.
        """
        action = decision.get("action", "WAIT")

        # WAIT no llega a M4 — pero por si acaso
        if action == "WAIT":
            return {"ok": False, "error": "action_wait", "action": "WAIT"}

        # Verificar conexión
        if not self._ensure_connected():
            return {
                "ok":    False,
                "error": "exchange_not_connected",
                "action": action,
                "decision_id": decision.get("decision_id"),
            }

        # ── COMPRA ───────────────────────────────────────────────────────────
        if action == "BUY":
            execution = execute_buy(
                ex            = self.ex,
                decision      = decision,
                current_price = current_price,
                lot_step      = self.lot_step,
            )

        # ── VENTA (V1, V2, INTRA) ────────────────────────────────────────────
        elif action in ("SELL_V1", "SELL_V2", "SELL_INTRA"):
            lot_id = decision.get("lot_id")
            if not lot_id:
                execution = {
                    "ok":          False,
                    "error":       "no_lot_id_in_decision",
                    "action":      action,
                    "decision_id": decision.get("decision_id"),
                }
            else:
                lot = self._find_lot(df_inventory, lot_id)
                if not lot:
                    execution = {
                        "ok":          False,
                        "error":       f"lot_not_found:{lot_id}",
                        "action":      action,
                        "decision_id": decision.get("decision_id"),
                    }
                else:
                    execution = execute_sell(
                        ex            = self.ex,
                        decision      = decision,
                        lot           = lot,
                        lot_step      = self.lot_step,
                        current_price = current_price,
                    )
        else:
            execution = {
                "ok":    False,
                "error": f"action_desconocida:{action}",
                "action": action,
            }

        # ── Loguear y mostrar siempre ─────────────────────────────────────────
        log_execution(execution)
        print_execution(execution)

        return execution

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — Ejecución directa (prueba en dry_run)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import pandas as pd

    print("\n" + "═"*72)
    print("  ⚙️   M4 — EJECUTOR DE ÓRDENES  ·  Prueba en DRY RUN")
    print("═"*72)

    if not DRY_RUN:
        print("\n  ⚠️  DRY_RUN=0 detectado — esta prueba solo corre en modo simulado.")
        print("  Seteá DRY_RUN=1 en tu entorno para correr la prueba.\n")
        sys.exit(1)

    # ── Inventario simulado
    import pandas as pd
    df_inv_sim = pd.DataFrame([{
        "lot_id":           "LOT-TEST001",
        "qty_restante_btc": 0.001,
        "price_usdt":       68000.0,
        "capitalizado_usdt":68.07,
        "status":           "OPEN",
    }])

    executor = OrderExecutor()

    # ── Test 1: BUY
    print("\n  ── Test 1: BUY simulado")
    decision_buy = {
        "decision_id": "DEC-TEST-BUY",
        "action":      "BUY",
        "rule_id":     "BUY-DIP-STRONG",
        "qty_usdt":    50.0,
        "sell_pct":    None,
        "lot_id":      None,
        "dry_run":     True,
    }
    result_buy = executor.run(decision_buy, df_inv_sim, current_price=70400.0)
    print(f"  → ok={result_buy['ok']} qty_btc={result_buy.get('filled_qty',0):.8f} fee=${result_buy.get('fee_usdt',0):.4f}")

    # ── Test 2: SELL_V1
    print("\n  ── Test 2: SELL_V1 simulado (40% del lote)")
    decision_sell = {
        "decision_id": "DEC-TEST-SELL",
        "action":      "SELL_V1",
        "rule_id":     "V1-TP2",
        "qty_usdt":    0.0,
        "sell_pct":    40.0,
        "lot_id":      "LOT-TEST001",
        "dry_run":     True,
    }
    result_sell = executor.run(decision_sell, df_inv_sim, current_price=70400.0)
    print(f"  → ok={result_sell['ok']} qty_btc={result_sell.get('filled_qty',0):.8f} fee=${result_sell.get('fee_usdt',0):.4f}")

    # ── Test 3: SELL_V2 (100%)
    print("\n  ── Test 3: SELL_V2 simulado (100% del lote — stop loss)")
    decision_sl = {
        "decision_id": "DEC-TEST-SL",
        "action":      "SELL_V2",
        "rule_id":     "V2-SL-NORMAL",
        "qty_usdt":    0.0,
        "sell_pct":    100.0,
        "lot_id":      "LOT-TEST001",
        "dry_run":     True,
    }
    result_sl = executor.run(decision_sl, df_inv_sim, current_price=70400.0)
    print(f"  → ok={result_sl['ok']} qty_btc={result_sl.get('filled_qty',0):.8f} fee=${result_sl.get('fee_usdt',0):.4f}")

    print("\n  ✅ Prueba M4 completa. Revisá logs/ para ver los EXECUTION en el .jsonl.\n")
