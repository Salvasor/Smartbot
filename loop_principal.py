# -*- coding: utf-8 -*-
"""
Módulo 6 — Loop Principal · loop_principal.py

Orquestador del bot. Une M1→M5 en un loop de 10 segundos.

Orden de ejecución por loop:
  1. Balance real de Binance (capital_usdt, capital_btc)
  2. Inventario M1 (con TTL y caché)
  3. Mercado M2 (ticks + OB cada loop, OHLCV cada 60s)
  4. Apalancamiento M2.5 (con TTL propio)
  5. Decisión M3
  6. Ejecución M4 (solo si action != WAIT)
  7. Contabilidad M5 (solo si M4 ok)
  8. Heartbeat → status_hb en Sheets

Filosofía:
  - Si un módulo falla, loguea y sigue — nunca se cae por un error aislado
  - DRY_RUN=1 por defecto — poné DRY_RUN=0 solo cuando estés listo
  - CTRL+C limpio — espera a que termine el loop actual antes de salir
  - Reconexión automática ante errores de red o exchange

Variables de entorno requeridas:
  API_KEY, API_SECRET          → Binance (solo para balance y órdenes)
  GOOGLE_APPLICATION_CREDENTIALS o service_account.json → Sheets
  SPREADSHEET_ID               → ID de tu Google Sheet
  DRY_RUN=1                    → modo simulado (default)
  SYMBOLS=BTC/USDT             → símbolo a operar
"""

import os
import sys
import time
import signal
import logging
import traceback
from datetime import datetime, timezone
from typing import Optional, Dict

# Cargar variables de entorno desde .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv no instalado — usar variables del sistema directamente

import pandas as pd
import ccxt

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — Config, logging y señales de sistema
# ──────────────────────────────────────────────────────────────────────────────

# Modo seguro — NUNCA cambies a 0 sin haber probado en dry_run primero
DRY_RUN: bool = os.getenv("DRY_RUN", "1") == "1"

# Ritmo del loop
LOOP_FAST_S:    int = int(os.getenv("LOOP_FAST_S",    "10"))   # loop principal
OHLCV_REFRESH_S: int = int(os.getenv("OHLCV_REFRESH_S", "60"))  # refresh de velas

# Símbolo
SYMBOL: str = os.getenv("SYMBOLS", "BTC/USDT").split(",")[0].strip()

# Credenciales Binance (solo para balance y M4)
API_KEY:    Optional[str] = os.getenv("BINANCE_KEY")
API_SECRET: Optional[str] = os.getenv("BINANCE_SECRET")
EXCHANGE_ID: str          = os.getenv("EXCHANGE_ID", "binance")

# Presupuesto de lecturas Sheets
SHEETS_BUDGET: int = int(os.getenv("SHEETS_BUDGET", "60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [M6] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("M6")

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# Flag para salida limpia con CTRL+C
_shutdown_requested = False

def _handle_signal(signum, frame):
    global _shutdown_requested
    log.info("Señal de cierre recibida — terminando al final del loop actual...")
    _shutdown_requested = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 2 — Inicialización de módulos
# ──────────────────────────────────────────────────────────────────────────────

def init_modules():
    """
    Importa e inicializa todos los módulos.
    Retorna dict con las instancias listas.
    Lanza excepción si algo crítico falla.
    """
    log.info("Inicializando módulos...")

    # ── M1 — Inventario
    import inventory_status as m1
    m1.ensure_base()
    budget = m1.Budget(reads_left=SHEETS_BUDGET)
    cache  = m1.InventoryCache()
    log.info("M1 OK — inventario conectado a Sheets")

    # ── M2 — Mercado
    from market_status import WorldReader, print_market_summary
    reader = WorldReader()
    log.info("M2 OK — WorldReader conectado a Binance (público)")

    # ── M2.5 — Apalancamiento
    from leverage_context import read_leverage_context
    log.info("M2.5 OK — leverage_context listo")

    # ── M3 — Motor de decisión
    from decision_engine import DecisionEngine, reload_policies
    engine = DecisionEngine(dry_run=DRY_RUN)
    log.info(f"M3 OK — DecisionEngine inicializado · versión=2.0")

    # ── M4 — Ejecutor de órdenes
    from order_executor import OrderExecutor
    executor = OrderExecutor()
    log.info("M4 OK — OrderExecutor inicializado")

    # ── M5 — Contabilidad
    from accounting import Accountant
    accountant = Accountant(m1, engine.log_outcome)
    log.info("M5 OK — Accountant inicializado")

    return {
        "m1":             m1,
        "budget":         budget,
        "cache":          cache,
        "reader":         reader,
        "print_mkt":      print_market_summary,
        "read_lev":       read_leverage_context,
        "engine":         engine,
        "executor":       executor,
        "accountant":     accountant,
        "reload_policies": reload_policies,
    }


def build_auth_exchange() -> Optional[ccxt.Exchange]:
    """
    Construye cliente ccxt autenticado para leer balance real.
    Solo necesario para obtener capital_usdt y capital_btc.
    En dry_run retorna None — usamos valores simulados.
    """
    if DRY_RUN:
        return None
    if not API_KEY or not API_SECRET:
        log.warning("API_KEY/API_SECRET no encontrados — no se puede leer balance real")
        return None
    try:
        ex_class = getattr(ccxt, EXCHANGE_ID)
        ex = ex_class({
            "apiKey":          API_KEY,
            "secret":          API_SECRET,
            "enableRateLimit": True,
            "options": {
                "defaultType":             "spot",
                "adjustForTimeDifference": True,
                "recvWindow":              10000,
            }
        })
        ex.fetch_currencies = lambda *a, **k: {}
        ex.load_markets()
        log.info("Exchange autenticado OK — balance real disponible")
        return ex
    except Exception as e:
        log.error(f"No se pudo conectar exchange autenticado: {e}")
        return None


def fetch_balance(ex: Optional[ccxt.Exchange], price_btc: float) -> tuple[float, float]:
    """
    Retorna (capital_usdt, capital_btc_en_usdt).
    En dry_run o sin conexión, retorna valores de prueba.
    """
    if ex is None or DRY_RUN:
        # Valores de prueba para dry_run — ajustá según tu capital real
        usdt_sim = float(os.getenv("SIM_CAPITAL_USDT", "70.0"))
        btc_sim  = float(os.getenv("SIM_CAPITAL_BTC",  "0.0"))
        return usdt_sim, btc_sim * price_btc

    try:
        balance  = ex.fetch_balance()
        usdt     = float(balance.get("USDT", {}).get("free", 0) or 0)
        btc_qty  = float(balance.get("BTC",  {}).get("total", 0) or 0)
        btc_usdt = btc_qty * price_btc if price_btc > 0 else 0.0
        return usdt, btc_usdt
    except Exception as e:
        log.warning(f"No se pudo leer balance: {e} · usando 0")
        return 0.0, 0.0

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 3 — Loop principal
# ──────────────────────────────────────────────────────────────────────────────

def run_loop(modules: Dict) -> None:
    """
    Loop principal del bot.
    Corre indefinidamente hasta CTRL+C o error crítico.
    """
    m1              = modules["m1"]
    budget          = modules["budget"]
    cache           = modules["cache"]
    reader          = modules["reader"]
    print_mkt       = modules["print_mkt"]
    read_lev        = modules["read_lev"]
    engine          = modules["engine"]
    executor        = modules["executor"]
    accountant      = modules["accountant"]
    reload_policies = modules["reload_policies"]

    # Exchange autenticado para balance
    ex_auth = build_auth_exchange()

    loop_idx       = 0
    last_ohlcv_ts  = 0.0   # timestamp del último refresh de OHLCV
    snap_counter   = 0
    _sold_this_cycle: set = set()  # anti-duplicado: lotes vendidos en este ciclo
    _sold_persistent: set = set()  # lotes cerrados confirmados — persiste entre loops

    SEP2 = "═" * 72
    log.info(f"Loop iniciado · símbolo={SYMBOL} · dry_run={DRY_RUN} · loop={LOOP_FAST_S}s")
    print(f"\n{SEP2}")
    print(f"  🤖  BOT BTC V2  ·  {_utc_now_str()}")
    print(f"  Símbolo  : {SYMBOL}")
    print(f"  Modo     : {'🟡 DRY RUN — sin órdenes reales' if DRY_RUN else '🔴 REAL — órdenes activas'}")
    print(f"  Loop     : cada {LOOP_FAST_S}s  ·  OHLCV refresh: cada {OHLCV_REFRESH_S}s")
    print(f"{SEP2}\n")

    while not _shutdown_requested:
        loop_start = time.time()
        loop_idx  += 1
        snap_id    = f"SNP-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{loop_idx:04d}"

        try:
            # ── 1. INVENTARIO (M1) ───────────────────────────────────────────
            # Resetear presupuesto si pasó 1h — evita congelamiento de caché
            # cuando budget se agota y nunca vuelve a refrescar por sí solo
            if budget.reset_if_needed():
                log.info(
                    f"Budget reseteado a {budget.reads_left} lecturas — "
                    f"caché expirada para forzar refresh inmediato"
                )
                cache.ts_loaded = 0  # expirar caché para que refresque en este loop

            # Primera carga obligatoria, luego solo si TTL venció
            if cache.df is None:
                if budget.can_read():
                    cache.refresh_full(budget)
                    log.info(f"Inventario cargado · {len(cache.df)} lotes abiertos · budget={budget.reads_left}")
                else:
                    log.error("Sin presupuesto para carga inicial de inventario")
                    time.sleep(LOOP_FAST_S)
                    continue
            else:
                m1.inventory_try_refresh(cache, budget)

            df_inventory = m1.inventory_get(cache)
            if df_inventory is None:
                df_inventory = pd.DataFrame(columns=m1.COLUMNS["lotes"])

            # ── 2. MERCADO (M2) ──────────────────────────────────────────────
            # Decidir si refrescamos OHLCV (cada 60s) o solo ticks+OB
            now_ts        = time.time()
            force_ohlcv   = (now_ts - last_ohlcv_ts) >= OHLCV_REFRESH_S
            force_flags   = {"orderbook": (loop_idx % 3 == 0)}  # OB cada 3 loops

            out = reader.read_once(
                snapshot_id      = snap_id,
                loop_idx         = loop_idx,
                watchlist_symbols= set([SYMBOL]),
                force_flags      = force_flags,
                bucket_focus     = 1,
                use_orderbook    = None,
            )

            if force_ohlcv:
                last_ohlcv_ts = now_ts
                try:
                    reload_policies()
                except Exception as e:
                    log.warning(f"reload_policies falló (no crítico): {e}")
                # M3.5 — LotAnalytics: recalcular stats históricas si pasaron 24h
                engine.maybe_refresh_analytics()

            world_summary = out.get("world_summary", pd.DataFrame())
            ttl           = out.get("ttl", {})
            errors_m2     = out.get("errors", [])

            if errors_m2:
                log.warning(f"M2 errores: {errors_m2}")

            # Precio actual
            price_now = None
            if world_summary is not None and not world_summary.empty:
                price_now = float(world_summary.iloc[0].get("last", 0) or 0)

            if not price_now or price_now <= 0:
                log.warning("Sin precio — saltando loop")
                time.sleep(LOOP_FAST_S)
                continue

            # ── 3. APALANCAMIENTO (M2.5) ─────────────────────────────────────
            lev_ctx = {}
            try:
                lev_ctx = read_lev(current_price=price_now)
                if lev_ctx.get("errors"):
                    log.warning(f"M2.5 errores: {lev_ctx['errors']}")
            except Exception as e:
                log.warning(f"M2.5 falló: {e} · usando leverage_risk=low")
                lev_ctx = {"leverage_risk": "low", "errors": [str(e)]}

            # ── 4. BALANCE REAL ──────────────────────────────────────────────
            capital_usdt, capital_btc_usdt = fetch_balance(ex_auth, price_now)

            if capital_usdt <= 0 and capital_btc_usdt <= 0:
                log.warning("Balance cero — verificá tu conexión o capital")

            # ── 5. DECISIÓN (M3) ─────────────────────────────────────────────
            # Ventas del día para pnl_dia_pct
            try:
                df_ventas_hoy = m1.read_ventas_hoy() if hasattr(m1, 'read_ventas_hoy') else None
            except Exception:
                df_ventas_hoy = None

            # Construir ctx desde world_summary + leverage + balance
            ctx = {}
            if world_summary is not None and not world_summary.empty:
                row = world_summary.iloc[0]
                ctx = {k: row.get(k) for k in row.index}
                ctx["last_price"]        = price_now
                ctx["vol_state"]         = row.get("vol_state_1m")
                ctx["speed_state"]       = row.get("speed_state_1m")
                ctx["price_position_1m"] = row.get("price_position_1m")
                ctx["price_position_1h"] = row.get("price_position_1h")

            # Agregar leverage (M2.5 — campos completos)
            ctx["leverage_risk"]   = lev_ctx.get("leverage_risk",   "low")
            ctx["funding_rate"]    = lev_ctx.get("funding_rate",    0.0)
            ctx["funding_signal"]  = lev_ctx.get("funding_signal",  "neutral")
            ctx["oi_price_signal"] = lev_ctx.get("oi_price_signal", "stable")
            ctx["oi_change_pct"]   = lev_ctx.get("oi_change_pct",   0.0)
            ctx["ls_signal"]       = lev_ctx.get("ls_signal",       "neutral")
            ctx["long_short_ratio"]= lev_ctx.get("long_short_ratio", 1.0)

            # Agregar capital e inventario
            total_capital = capital_usdt + capital_btc_usdt
            usdt_pct      = (capital_usdt / total_capital * 100) if total_capital > 0 else 0.0
            n_lotes_open  = len(df_inventory[df_inventory["status"].astype(str) == "OPEN"]) if df_inventory is not None and not df_inventory.empty else 0
            inv_st        = "light" if usdt_pct >= 60 else ("balanced" if usdt_pct >= 30 else "heavy")

            ctx.update({
                "budget_usdt":      capital_usdt,
                "usdt_reserve_pct": usdt_pct,
                "n_lotes_open":     n_lotes_open,
                "inv_state":        inv_st,
                "btc_disponible":   capital_btc_usdt / price_now if price_now > 0 else 0.0,
            })

            # ── Enriquecer df_inventory con pnl_pct_actual ──────────────────
            # El motor necesita pnl_pct_actual por lote para calcular score_venta_tp/sl.
            # No está en Sheets — se calcula aquí donde conviven price_now y df_inventory.
            if df_inventory is not None and not df_inventory.empty and price_now > 0:
                df_inventory = df_inventory.copy()

                raw_price = df_inventory["price_usdt"].astype(str)

                clean_price = (
                    raw_price
                    .str.replace(r"[^\d\.\-]", "", regex=True)
                    .str.strip()
                )

                precio_compra = pd.to_numeric(clean_price, errors="coerce")

                df_inventory["pnl_pct_actual"] = (
                    (price_now - precio_compra) / precio_compra * 100
                ).round(4)

            decision = engine.decide(
                ctx=ctx,
                df_inventory=df_inventory,
                df_ventas_hoy=df_ventas_hoy,
                snapshot_id=snap_id,
            )

            action = decision.get("action", "WAIT")
            

            # ── 6. EJECUCIÓN (M4) ────────────────────────────────────────────
            # Anti-duplicado: si este lote ya fue vendido en este ciclo o confirmado cerrado → skip
            lot_id_decision = decision.get("lot_id")
            if action in ("SELL_V1", "SELL_V2", "SELL_INTRA") and lot_id_decision:
                if lot_id_decision in _sold_this_cycle or lot_id_decision in _sold_persistent:
                    log.warning(f"Anti-dup: lote {lot_id_decision} ya cerrado — skip")
                    action = "WAIT"
                    decision["action"] = "WAIT"

            execution = None
            if action != "WAIT":
                try:
                    execution = executor.run(
                        decision      = decision,
                        df_inventory  = df_inventory,
                        current_price = price_now,
                    )
                except Exception as e:
                    log.error(f"M4 excepción inesperada: {e}")
                    log.error(traceback.format_exc())
                    execution = {"ok": False, "error": str(e), "action": action}

            # ── 7. CONTABILIDAD (M5) ─────────────────────────────────────────
            # Corre siempre después del bloque M4 — execution=None cuando action=WAIT,
            # los `if execution and ...` short-circuit sin ejecutar nada en ese caso.
            # Si insufficient_funds o insufficient_btc → marcar lote para no reintentar
            if execution and execution.get("error") in ("insufficient_funds", "insufficient_btc"):
                bad_lot = execution.get("lot_id") or lot_id_decision
                if bad_lot:
                    _sold_this_cycle.add(bad_lot)
                    _sold_persistent.add(bad_lot)
                    log.warning(f"{execution.get('error')} en {bad_lot} — bloqueado permanentemente")

            if execution and execution.get("ok"):
                try:
                    acc_result = accountant.run(
                        execution=execution,
                        decision=decision,
                        df_inventory=df_inventory,
                    )

                    print(f"\n[DEBUG M5 RESULT] {acc_result}")

                    if acc_result.get("ok"):
                        lot_id_acc = acc_result.get("lot_id") or lot_id_decision

                        if lot_id_acc:
                            _sold_this_cycle.add(lot_id_acc)

                            # Solo persistente si M5 confirmó que quedó cerrado
                            if acc_result.get("lot_closed", False):
                                _sold_persistent.add(lot_id_acc)
                                log.info(f"Lote {lot_id_acc} marcado como CERRADO en anti-dup")
                            else:
                                log.info(
                                    f"Lote {lot_id_acc} PARCIAL — sigue en inventario con "
                                    f"{acc_result.get('qty_restante', '?')} BTC"
                                )

                        # M3.5 — notificar al tracker solo si M5 confirmó cierre real
                        if acc_result.get("lot_closed", False):
                            engine.on_lot_closed()

                        # Refresh inmediato solo si M5 sí persistió bien
                        cache.mark_dirty()
                        try:
                            if budget.can_read():
                                cache.refresh_full(budget)
                                df_inventory = m1.inventory_get(cache)
                                if df_inventory is None:
                                    df_inventory = pd.DataFrame(columns=m1.COLUMNS["lotes"])
                                log.info(f"Cache post-trade refrescada · {len(df_inventory)} lotes en inventario")
                            else:
                                # Sin presupuesto: expirar TTL para forzar refresh en el próximo loop
                                cache.ts_loaded = 0
                                log.warning("Sin presupuesto para refresh post-trade · cache expirada para proximo loop")
                        except Exception as re:
                            log.warning(f"Refresh post-trade fallo (no critico): {re}")

                    else:
                        log.error(f"M5 devolvió fallo: {acc_result}")

                except Exception as e:
                    log.error(f"M5 excepcion inesperada: {e}")
                    log.error(traceback.format_exc())

            # ── 8. HEARTBEAT ─────────────────────────────────────────────────
            try:
                accountant.heartbeat(
                    snapshot_id=snap_id,
                    df_inventory=df_inventory,
                    budget_reads_left=budget.reads_left,
                    comment=f"loop={loop_idx} action={action}",
                )
            except AttributeError:
                pass  # versión local de Accountant sin heartbeat — OK
                
            # ── 9. FIN DE LOOP ───────────────────────────────────────────────
            m1.end_of_loop(cache)
            snap_counter += 1
            _sold_this_cycle.clear()  # limpiar ciclo — persistent se mantiene
            # Limpiar persistent cuando Sheets confirme que el lote ya no está OPEN
            # (el lote desaparece de df_inventory = Sheets lo marcó como CLOSED)
            # IMPORTANTE: mantener protección MIENTRAS Sheets sigue mostrándolo OPEN
            if df_inventory is not None:
                if not df_inventory.empty:
                    open_lot_ids = set(df_inventory["lot_id"].astype(str).tolist())
                    # Conservar en _sold_persistent solo los que aún aparecen en Sheets
                    # Los que ya desaparecieron de Sheets = confirmados cerrados = los limpiamos
                    _sold_persistent = {lid for lid in _sold_persistent if lid in open_lot_ids}
                elif cache.is_fresh():
                    # Caché fresca + inventario vacío = todos los lotes confirmados
                    # cerrados en Sheets → es seguro limpiar _sold_persistent completo.
                    # Si la caché fuera stale no entraríamos aquí (is_fresh() = False).
                    if _sold_persistent:
                        log.info(
                            f"_sold_persistent limpiado ({len(_sold_persistent)} lotes) "
                            f"— inventario vacío confirmado por caché fresca"
                        )
                    _sold_persistent.clear()

        except KeyboardInterrupt:
            break

        except Exception as e:
            log.error(f"Error en loop {loop_idx}: {e}")
            log.error(traceback.format_exc())
            # No rompemos el loop — solo esperamos y seguimos

        # ── Dormir el tiempo restante del loop
        elapsed   = time.time() - loop_start
        sleep_for = max(0.1, LOOP_FAST_S - elapsed)
        time.sleep(sleep_for)

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — Ejecución directa
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*72)
    print("  🤖  BOT BTC V2 — ARRANQUE")
    print("═"*72)
    print(f"  Modo     : {'DRY RUN 🟡' if DRY_RUN else 'REAL 🔴 — cuidado'}")
    print(f"  Símbolo  : {SYMBOL}")
    print(f"  Loop     : {LOOP_FAST_S}s")
    print(f"  OHLCV    : cada {OHLCV_REFRESH_S}s")

    if not DRY_RUN:
        print("\n  ⚠️  MODO REAL ACTIVADO — se ejecutarán órdenes reales en Binance")
        print("  Tenés 5 segundos para cancelar con CTRL+C...\n")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("  Cancelado.")
            sys.exit(0)

    print()

    try:
        modules = init_modules()
    except Exception as e:
        log.error(f"Error al inicializar módulos: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)

    run_loop(modules)
    log.info("Bot detenido limpiamente.")