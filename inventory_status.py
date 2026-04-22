# -*- coding: utf-8 -*-
"""
Módulo 1 — Inventario (Google Sheets)
Versión autónoma: conexión, lectura/escritura, caché y telemetría.

Este módulo concentra TODO lo relacionado a inventario:
- Conexión a Google Sheets con service_account.json
- Ensure de hojas y headers con columnas canónicas
- Lectura de 'lotes' (solo abiertos) y utilidades genéricas de lectura
- Escrituras atómicas (append / update por lot_id)
- Caché de inventario (TTL + dirty flag) y presupuesto de lecturas
- Logging y heartbeat (status_hb)

Requisitos:
    pip install gspread google-auth pandas
    GOOGLE_APPLICATION_CREDENTIALS o service_account.json junto al script
"""

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 0 — Imports y configuración
# ──────────────────────────────────────────────────────────────────────────────
import os
import json
import time
import uuid
import pandas as pd
import numpy as np
import gspread
from IPython.display import display
from datetime import datetime, timezone
from google.oauth2.service_account import Credentials

# Credenciales y Spreadsheet
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID", "1-F5Dcv6-VPKLxuwzpABXZoHLB7bbmeSoydUr3RzGTkc")
SHEET_PREFIX = os.getenv("SNAPSHOT_SHEET_PREFIX", "").strip()  # opcional, p.ej. "v4_"

def _full(name: str) -> str:
    """Aplica prefijo (si lo hay) al nombre de hoja lógico."""
    return f"{SHEET_PREFIX}{name}" if SHEET_PREFIX else name

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 1 — Esquemas canónicos de columnas (ACTUALIZADO)
# ──────────────────────────────────────────────────────────────────────────────
COLUMNS = {
    "lotes": [
        "lot_id", "order_id", "client_order_id",
        "bucket_id", "bucket_label", "bucket_version", "last_touch_ts", "source_route",
        "trade_ids", "date", "datetime", "symbol", "side", "type",
        "qty_inicial_btc", "qty_restante_btc", "price_usdt", "notional_usdt",
        "fee_amount", "fee_asset", "fee_usdt", "capitalizado_usdt",
        "exchange", "status", "policy", "comentario",
        "decision_id", "rule_id_trigger", "ai_model_id"
    ],
    "ventas": [
        "sale_id", "order_id", "client_order_id", "date", "datetime", "symbol", "side", "type",
        "avg_price_usdt", "qty_total_btc", "ingreso_bruto_usdt",
        "fee_total_amount", "fee_total_asset", "fee_total_usdt",
        "costo_asignado_usdt", "pnl_usdt", "pnl_pct",
        "exchange", "fills_count", "lotes_involucrados",
        "policy", "catalyst", "status", "comment_full",
        "decision_id", "rule_id_trigger", "risk_state", "snapshot_id", "latency_ms"
    ],
    "ventas_detalle": [
        "sale_id", "lot_id", "qty_usada_btc", "costo_unit_usdt", "costo_usdt",
        "sold_price_usdt", "ingreso_usdt", "fee_prorrata_usdt",
        "pnl_usdt", "pnl_pct", "policy_venta", "decision_id",
        "rule_id_trigger", "comment_short"
    ],
    "decisions_log": [
        "decision_id", "snapshot_id", "symbol", "route", "rule_id", "action", "qty", "price_ref",
        "why_applied", "why_not_applied", "next_triggers",
        "risk_state", "policy_version", "ai_model_id", "comment_short"
    ],
    "status_hb": [
        "snapshot_id", "ts", "symbol", "px", "spread_pct", "atr_pct",
        "exposure_pct", "n_lotes_open", "risk_flags", "cooldowns",
        "budget_reads_left", "decision_last", "comment"
    ],
    "append_only": [
        "transition_id", "ts", "lot_id",
        "from_bucket_id", "to_bucket_id",
        "reason", "rule_id_trigger", "snapshot_id",
        "actor", "notes", "prev_bucket_version", "new_bucket_version"
    ],
    "lotes_history": [
        "lot_id", "order_id", "client_order_id",
        "bucket_id", "bucket_label", "bucket_version", "last_touch_ts", "source_route",
        "trade_ids", "date", "datetime", "symbol", "side", "type",
        "qty_inicial_btc", "qty_restante_btc", "price_usdt", "notional_usdt",
        "fee_amount", "fee_asset", "fee_usdt", "capitalizado_usdt",
        "exchange", "status", "policy", "comentario",
        "decision_id", "rule_id_trigger", "ai_model_id",
        "history_ts", "history_event"
    ],
    "decisiones": [
        "decision_id", "date", "datetime", "symbol", "timeframe", "market_price",
        "inv_usdt_pct", "inv_btc_pct", "ind_rsi", "ind_ema5", "ind_ema12",
        "vol", "spread", "action_plan", "policy", "reason",
        "target_qty_btc", "target_price", "lot_ids_candidatos",
        "sale_id", "order_id", "status_decision", "blocked_by", "blocked_by_code", "result_pnl_usdt"
    ],
    "Dashboard": [],
    "logs": ["ts", "scope", "level", "message", "extra_json"],
}


# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 2 — Conexión singleton y ensure de headers
# ──────────────────────────────────────────────────────────────────────────────
_gc = None
_sh = None

def _client() -> gspread.client.Client:
    """Cliente gspread autorizado (singleton)."""
    global _gc
    if _gc is None:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scopes)
        _gc = gspread.authorize(creds)
    return _gc

def _sheet() -> gspread.Spreadsheet:
    """Spreadsheet abierto por ID (singleton)."""
    global _sh
    if _sh is None:
        _sh = _client().open_by_key(SPREADSHEET_ID)
    return _sh

def _get_or_create_worksheet(title: str, min_cols: int):
    """Obtiene/crea worksheet por título con al menos min_cols columnas."""
    sh = _sheet()
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=max(50, min_cols))
    return ws

def ensure_headers(sheet_name: str):
    """Asegura que exista la hoja y tenga headers canónicos en fila 1."""
    columns = COLUMNS[sheet_name]
    ws = _get_or_create_worksheet(_full(sheet_name), len(columns))
    if not ws.row_values(1):
        ws.insert_row(columns, 1)
    return ws

def ensure_base():
    """
    Asegura TODAS las hojas que este módulo usa/controla.
    Nota: 'Dashboard' queda con headers vacíos (matriz libre para KPIs).
    """
    for name in ("lotes", "ventas", "ventas_detalle", "decisions_log",
                 "status_hb", "append_only", "lotes_history", "decisiones",
                 "logs", "Dashboard"):
        ensure_headers(name)


# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 3 — Utilidades genéricas de lectura/escritura
# ──────────────────────────────────────────────────────────────────────────────
def _align_df_columns(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Normaliza DataFrame al orden canónico del sheet."""
    cols = COLUMNS[sheet_name]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]

def read_table_df(sheet_name: str) -> pd.DataFrame:
    """Lee una hoja por nombre lógico y devuelve DataFrame canónico."""
    ws = ensure_headers(sheet_name)
    rows = ws.get_all_records()  # desde fila 2, fila 1 = headers
    df = pd.DataFrame.from_records(rows)
    return _align_df_columns(df, sheet_name)

def append_row_dict(sheet_name: str, row: dict):
    """Inserta una fila respetando el orden canónico de columnas."""
    ws = ensure_headers(sheet_name)
    cols = COLUMNS[sheet_name]
    values = [row.get(c, "") for c in cols]
    ws.append_row(values, value_input_option="USER_ENTERED")

def _col_label(n: int) -> str:
    """Convierte 1->A, 26->Z, 27->AA... para rangos de update robustos."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def _find_row_by_key(ws: gspread.Worksheet, key_col_name: str, key_value: str) -> int | None:
    """
    Encuentra el índice de fila (1-based) donde la columna key_col_name == key_value.
    Devuelve None si no se encuentra. Optimiza leyendo la columna específica.
    """
    headers = ws.row_values(1)
    if key_col_name not in headers:
        return None
    col_idx = headers.index(key_col_name) + 1  # 1-based
    col_values = ws.col_values(col_idx)  # incluye header en posición 1
    for i, val in enumerate(col_values[1:], start=2):  # datos desde fila 2
        if str(val) == str(key_value):
            return i
    return None

def update_row_by_key(sheet_name: str, key_col_name: str, key_value: str, fields: dict):
    """
    Actualiza una fila (entera) identificada por key_col_name=key_value con merge de 'fields'.
    Escribe la fila completa respetando el orden canónico de columnas.
    """
    ws = ensure_headers(sheet_name)
    row_idx = _find_row_by_key(ws, key_col_name, key_value)
    if row_idx is None:
        raise ValueError(f"{sheet_name}: no se encontró {key_col_name}='{key_value}'")
    # Traer registro actual (como dict) desde la fila
    columns = COLUMNS[sheet_name]
    current = dict(zip(columns, ws.row_values(row_idx)))
    current.update({k: v for k, v in fields.items() if k in columns})
    # Ensamblar en orden y actualizar range exacto
    end_col = _col_label(len(columns))
    ws.update(f"A{row_idx}:{end_col}{row_idx}", [[current.get(c, "") for c in columns]])

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 4 — Lectura principal de inventario (solo lotes abiertos)
# ──────────────────────────────────────────────────────────────────────────────
def read_inventory_open_lotes() -> pd.DataFrame:
    """
    Devuelve DataFrame de lotes 'vivos':
      - status ∈ {'OPEN','PARTIAL'} (case-insensitive) O qty_restante_btc > 0.
    """
    df = read_table_df("lotes")
    df["qty_restante_btc"] = pd.to_numeric(df["qty_restante_btc"], errors="coerce")
    status_upper = df["status"].astype(str).str.upper()
    mask_open = status_upper.isin(["OPEN", "PARTIAL"])
    mask_qty  = df["qty_restante_btc"].fillna(0) > 0
    #return df.loc[mask_open | mask_qty].copy()
    return df.loc[mask_open].copy()

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 5 — Escrituras de inventario (append/update), logging, heartbeat
# ──────────────────────────────────────────────────────────────────────────────
def write_log(level: str, message: str, extra: dict | None = None, scope: str = "mod1.inventory"):
    """Escribe una línea de auditoría en la hoja 'logs'."""
    append_row_dict("logs", {
        "ts": _utc_now_str(),
        "scope": scope,
        "level": level.upper(),
        "message": message,
        "extra_json": json.dumps(extra or {}, ensure_ascii=False),
    })

def update_lote_fields(lot_id: str, **fields):
    """Actualiza campos de un lote por lot_id (escritura atómica de fila completa)."""
    update_row_by_key("lotes", "lot_id", lot_id, fields)

def close_lote_if_empty_in_sheet(lot_id: str):
    """Cierra un lote en Sheets (status='CLOSED')."""
    update_lote_fields(lot_id, status="CLOSED")

def read_ventas_hoy() -> pd.DataFrame:
    """
    Lee las ventas del día de hoy desde la hoja 'ventas'.
    Filtra por fecha UTC del día actual.
    Retorna DataFrame con columnas canónicas de ventas (puede estar vacío).
    Usado por decision_engine para calcular pnl_dia_pct del Guardián G4
    y los multiplicadores del Auditor.
    """
    try:
        df = read_table_df("ventas")
        if df is None or df.empty:
            return pd.DataFrame(columns=COLUMNS["ventas"])
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # La columna 'date' puede tener formato "YYYY-MM-DD" o "YYYY-MM-DD HH:MM:SS"
        df["_date_prefix"] = df["date"].astype(str).str[:10]
        df_hoy = df[df["_date_prefix"] == today_str].drop(columns=["_date_prefix"])
        return df_hoy.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=COLUMNS["ventas"])

def new_decision_id() -> str:
    return "DEC-" + uuid.uuid4().hex[:10].upper()

def new_lot_id() -> str:
    return "LOT-" + uuid.uuid4().hex[:10].upper()

def write_status_hb(snapshot_id: str, df_inv: pd.DataFrame, budget_reads_left: int, comment: str = ""):
    """
    Escribe un pulso mínimo en status_hb (conteo de lotes abiertos).
    Los campos de mercado (px, spread, etc.) se llenarán desde el módulo de mercado.
    """
    # calcular n_lotes_open
    status_upper = df_inv["status"].astype(str).str.upper()
    n_open = int((status_upper.isin(["OPEN", "PARTIAL"]) | 
                  (pd.to_numeric(df_inv["qty_restante_btc"], errors="coerce").fillna(0) > 0)).sum())
    row = {
        "snapshot_id": snapshot_id,
        "ts": _utc_now_str(),
        "symbol": "", "px": "", "spread_pct": "", "atr_pct": "", "exposure_pct": "",
        "n_lotes_open": n_open,
        "risk_flags": "", "cooldowns": "",
        "budget_reads_left": budget_reads_left,
        "decision_last": "",
        "comment": comment,
    }
    append_row_dict("status_hb", row)

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 5bis — Transiciones de bucket: append → update → snapshot (opcional)
# ──────────────────────────────────────────────────────────────────────────────

def append_transition(lot_id: str,
                      from_bucket_id: int,
                      to_bucket_id: int,
                      *,
                      reason: str,
                      rule_id_trigger: str,
                      snapshot_id: str,
                      actor: str = "bot",
                      notes: str = "",
                      prev_bucket_version: int = 1,
                      new_bucket_version: int = 2) -> str:
    """
    Registra una transición de bucket en 'append_only' (append-first).
    Devuelve transition_id (UUID corto) para trazabilidad e idempotencia externa.
    """
    tid = "TRN-" + uuid.uuid4().hex[:10].upper()
    append_row_dict("append_only", {
        "transition_id": tid,
        "ts": _utc_now_str(),
        "lot_id": lot_id,
        "from_bucket_id": str(from_bucket_id),
        "to_bucket_id": str(to_bucket_id),
        "reason": reason,
        "rule_id_trigger": rule_id_trigger,
        "snapshot_id": snapshot_id,
        "actor": actor,
        "notes": notes,
        "prev_bucket_version": str(prev_bucket_version),
        "new_bucket_version": str(new_bucket_version),
    })
    return tid

def update_lote_bucket(lot_id: str,
                       to_bucket_id: int,
                       to_bucket_label: str,
                       new_bucket_version: int,
                       *,
                       source_route: str,
                       decision_id: str | None,
                       rule_id_trigger: str | None):
    """
    Actualiza el estado actual del lote en 'lotes' tras una transición.
    Es una escritura atómica de la fila (merge por columnas existentes).
    """
    fields = {
        "bucket_id": str(to_bucket_id),
        "bucket_label": to_bucket_label,
        "bucket_version": str(new_bucket_version),
        "last_touch_ts": _utc_now_str(),
        "source_route": source_route,
    }
    if decision_id is not None:
        fields["decision_id"] = decision_id
    if rule_id_trigger is not None:
        fields["rule_id_trigger"] = rule_id_trigger
    update_lote_fields(lot_id, **fields)

def snapshot_lote_history(lote_row: dict, event: str):
    """
    Crea un snapshot del registro completo de 'lotes' en 'lotes_history' para auditoría.
    'lote_row' debe contener al menos los campos definidos en COLUMNS['lotes'].
    """
    hist = {k: lote_row.get(k, "") for k in COLUMNS["lotes"]}
    hist["history_ts"] = _utc_now_str()
    hist["history_event"] = event
    append_row_dict("lotes_history", hist)


# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 6 — Caché (TTL + dirty flag) y presupuesto de lecturas
# ──────────────────────────────────────────────────────────────────────────────
INVENTORY_TTL_SECONDS = int(os.getenv("INVENTORY_TTL_SECONDS", "600"))  # 10 min por defecto

class Budget:
    """
    Presupuesto sencillo de lecturas FULL (se descuenta solo en refresh_full).
    Se resetea automáticamente cada reset_every_s segundos (por defecto 1 hora).
    Esto evita que el caché quede congelado indefinidamente cuando el budget
    se agota — después del intervalo, vuelven a estar disponibles las lecturas.
    """
    def __init__(self, reads_left: int, reset_every_s: int = 3600):
        self.reads_left      = int(reads_left)
        self._initial        = int(reads_left)
        self._reset_every_s  = int(reset_every_s)
        self._last_reset_ts  = time.time()

    def can_read(self) -> bool:
        return self.reads_left > 0

    def consume(self, n: int = 1) -> None:
        self.reads_left = max(0, self.reads_left - n)

    def reset_if_needed(self) -> bool:
        """
        Resetea reads_left a su valor inicial si pasó reset_every_s desde el
        último reset. Retorna True si efectivamente hizo reset.
        Llamar al inicio de cada loop para garantizar que el presupuesto
        se recargue periódicamente.
        """
        if time.time() - self._last_reset_ts >= self._reset_every_s:
            self.reads_left     = self._initial
            self._last_reset_ts = time.time()
            return True
        return False

class InventoryCache:
    """
    Caché de inventario abierto:
      - df: DataFrame en memoria
      - ts_loaded: epoch seconds del último refresh FULL
      - dirty: True si hubo mutación local post-trade (bloquea refresh en el mismo loop)
      - ttl_s: segundos de validez antes de evaluar refresh
    """
    def __init__(self, ttl_s: int = INVENTORY_TTL_SECONDS):
        self.df: pd.DataFrame | None = None
        self.ts_loaded: float | None = None
        self.dirty: bool = False
        self.ttl_s: int = int(ttl_s)

    def age_s(self) -> float:
        if self.ts_loaded is None:
            return float("inf")
        return max(0.0, time.time() - self.ts_loaded)

    def is_fresh(self) -> bool:
        return (self.df is not None) and (self.age_s() < self.ttl_s)

    def mark_dirty(self) -> None:
        self.dirty = True

    def clear_dirty(self) -> None:
        self.dirty = False

    def should_refresh(self, budget: Budget, initial_required: bool = False) -> bool:
        if initial_required and self.df is None:
            return budget.can_read()
        if self.dirty:
            return False
        if self.age_s() >= self.ttl_s and budget.can_read():
            return True
        return False

    def refresh_full(self, budget: Budget) -> None:
        """Lee 'lotes' (solo abiertos) y actualiza caché + timestamp + presupuesto."""
        df_new = read_inventory_open_lotes()
        if not isinstance(df_new, pd.DataFrame):
            raise TypeError("La lectura de inventario debe devolver pandas.DataFrame")
        self.df = df_new
        self.ts_loaded = time.time()
        self.dirty = False
        budget.consume(1)

def inventory_get(cache: InventoryCache) -> pd.DataFrame | None:
    """Devuelve el DataFrame de inventario actual (o None si aún no hay refresh inicial)."""
    return cache.df

def inventory_try_refresh(cache: InventoryCache, budget: Budget) -> None:
    """Intenta refrescar inventario si TTL venció y no está dirty (y hay presupuesto)."""
    if cache.should_refresh(budget, initial_required=False):
        cache.refresh_full(budget)

def inventory_post_trade_update(cache: InventoryCache, mutate_fn) -> None:
    """Mutación local de la caché tras un trade; prohíbe refresco en el mismo loop."""
    if cache.df is None:
        cache.mark_dirty()
        return
    mutate_fn(cache.df)  # el caller define cómo mutar (restar qty, etc.)
    cache.mark_dirty()

def end_of_loop(cache: InventoryCache) -> None:
    """Llamar al final del loop para liberar el dirty y permitir refresh en el próximo."""
    cache.clear_dirty()

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 7 — Helpers de mutación post-trade (en memoria)
# ──────────────────────────────────────────────────────────────────────────────
def mutate_cache_consume_qty(df_inv: pd.DataFrame, lot_id: str, qty_delta_btc: float, comentario: str | None = None):
    """
    Resta qty_restante_btc en el DataFrame en memoria; si queda <= 0, marca status='CLOSED'.
    NO persiste en Sheets; para persistir, usar update_lote_fields fuera del loop.
    """
    idx = df_inv.index[df_inv["lot_id"].astype(str) == str(lot_id)]
    if len(idx) == 0:
        return
    i = idx[0]
    cur_qty = pd.to_numeric(df_inv.at[i, "qty_restante_btc"], errors="coerce")
    new_qty = float(cur_qty) - float(qty_delta_btc)
    df_inv.at[i, "qty_restante_btc"] = max(0.0, new_qty)
    if comentario:
        base = str(df_inv.at[i, "comentario"] or "").strip()
        df_inv.at[i, "comentario"] = (base + " | " if base else "") + comentario
    if new_qty <= 0:
        df_inv.at[i, "status"] = "CLOSED"

# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 8 — Resumen visual de inventario en terminal
# ──────────────────────────────────────────────────────────────────────────────

def print_inventory_summary(df: pd.DataFrame) -> None:
    """
    Imprime en terminal un resumen claro y legible del inventario abierto.
    Muestra por lote: fecha de compra, precio, cantidad, fee y valor capitalizado.
    Al final incluye totales consolidados.
    """
    SEP  = "─" * 80
    SEP2 = "═" * 80

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{SEP2}")
    print(f"  📦  INVENTARIO ABIERTO  ·  {now_str}")
    print(SEP2)

    if df is None or df.empty:
        print("  ⚠️  Sin lotes abiertos en este momento.")
        print(SEP2 + "\n")
        return

    # ── Conversión de tipos numéricos ──────────────────────────────────────────
    for col in ("price_usdt", "qty_restante_btc", "qty_inicial_btc",
                "fee_usdt", "capitalizado_usdt", "notional_usdt"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    total_lotes     = len(df)
    total_btc       = df["qty_restante_btc"].sum()
    total_invertido = df["capitalizado_usdt"].sum()   # precio_compra × qty + fee
    total_fee       = df["fee_usdt"].sum()

    # ── Cabecera de tabla ──────────────────────────────────────────────────────
    print(f"\n  {'#':<4} {'LOT_ID':<16} {'FECHA':<12} {'SYMBOL':<10} "
          f"{'PRECIO COMPRA':>14} {'QTY RESTANTE':>14} {'QTY INICIAL':>13} "
          f"{'FEE (USDT)':>11} {'CAPITALIZADO':>13}  STATUS")
    print(f"  {SEP}")

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        lot_id      = str(row.get("lot_id",  ""))[:14]
        fecha       = str(row.get("date",    ""))[:10]
        symbol      = str(row.get("symbol",  ""))[:8]
        status      = str(row.get("status",  ""))
        price       = row["price_usdt"]
        qty_rest    = row["qty_restante_btc"]
        qty_ini     = row["qty_inicial_btc"]
        fee         = row["fee_usdt"]
        capitaliz   = row["capitalizado_usdt"]

        # Icono de estado
        icon = "🟢" if status.upper() == "OPEN" else "🟡" if status.upper() == "PARTIAL" else "⚪"

        print(f"  {idx:<4} {lot_id:<16} {fecha:<12} {symbol:<10} "
              f"{price:>14,.2f} {qty_rest:>14,.8f} {qty_ini:>13,.8f} "
              f"{fee:>11,.4f} {capitaliz:>13,.2f}  {icon} {status}")

    # ── Totales ────────────────────────────────────────────────────────────────
    print(f"  {SEP}")
    print(f"\n  RESUMEN  ({total_lotes} lote{'s' if total_lotes != 1 else ''})")
    print(f"  {'Total BTC disponible':<30}  {total_btc:>18,.8f} BTC")
    print(f"  {'Total fees pagados':<30}  {total_fee:>18,.4f} USDT")
    print(f"  {'Total invertido (capitalizado)':<30}  {total_invertido:>18,.2f} USDT")

    # Precio promedio ponderado
    if total_btc > 0:
        precio_prom = (df["price_usdt"] * df["qty_restante_btc"]).sum() / total_btc
        print(f"  {'Precio promedio ponderado':<30}  {precio_prom:>18,.2f} USDT/BTC")

    print(f"\n{SEP2}\n")


# ──────────────────────────────────────────────────────────────────────────────
# BLOQUE 9 — Demo controlada (ejecución directa)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ensure_base()
    write_log("INFO", "Arranque módulo de inventario")

    # presupuesto (ejemplo): 60 lecturas FULL por hora
    budget = Budget(reads_left=60)
    cache = InventoryCache(ttl_s=INVENTORY_TTL_SECONDS)

    # primera lectura obligatoria
    if cache.should_refresh(budget, initial_required=True):
        cache.refresh_full(budget)
    else:
        raise RuntimeError("No hay presupuesto para lectura inicial de inventario.")

    # loop simulado (3 iteraciones)
    for i in range(3):
        inventory_try_refresh(cache, budget)  # solo refresca si TTL venció y no está dirty

        df_inv = inventory_get(cache)
        print(f"[loop {i}] lotes vivos: {0 if df_inv is None else len(df_inv)}")

        # ejemplo post-trade (comentar si no aplica):
        # if i == 1 and df_inv is not None and len(df_inv) > 0:
        #     lot_id_demo = str(df_inv.iloc[0]["lot_id"])
        #     inventory_post_trade_update(cache, lambda df: mutate_cache_consume_qty(df, lot_id_demo, 0.001, "venta demo"))
        #     # fuera del loop (otro momento) persistir:
        #     # update_lote_fields(lot_id_demo, qty_restante_btc=str(max(0.0, float(df_inv.iloc[0]['qty_restante_btc']))))

        end_of_loop(cache)

    df_final = inventory_get(cache)

    if df_final is None or df_final.empty:
        df_final = pd.DataFrame(columns=COLUMNS["lotes"])

    print_inventory_summary(df_final)


