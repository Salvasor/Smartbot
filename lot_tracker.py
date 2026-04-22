# -*- coding: utf-8 -*-
"""
lot_tracker.py — M3.5a  Bot BTC V2
Tracking en tiempo real de lotes abiertos.

Responsabilidades:
- Recibe df_inventory + precio_actual cada loop (10s)
- Actualiza max_pnl_seen por lote Y LO PERSISTE EN DISCO → sobrevive reinicios
- Escribe snapshots JSONL cada 60s o en eventos clave:
    open, close, regime_change, pnl_threshold
- Calcula pnl_velocity_10m, age_minutes, time_since_peak_minutes

Formato en disco (subcarpeta lot_history/):
  open/lot_{lot_id}.jsonl         → snapshots del lote abierto
  open/lot_{lot_id}_state.json    → estado actual (max_pnl, timestamps, etc.)
  closed/lot_{lot_id}.jsonl       → archivado al cierre
  closed/lot_{lot_id}_state.json  → estado final archivado

API pública:
  tracker.update(df_inventory, price_now, regimen_actual)
  tracker.get_lot_context(lot_id) → dict
  tracker.get_max_pnl(lot_id)     → float
  tracker.get_all_max_pnl()       → Dict[str, float]
"""

from __future__ import annotations

import json
import time
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Optional, Set, Tuple

log = logging.getLogger("M35.LotTracker")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
_SNAPSHOT_INTERVAL_S   = 60         # snapshot periódico mínimo
_PNL_VELOCITY_WINDOW   = 60         # ticks de 10s = 10 minutos de memoria
_PNL_THRESHOLDS        = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 3.0]  # % de PnL


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _to_float(val) -> float:
    """
    Convierte a float tolerando el formato de Google Sheets: '$66,729.58' → 66729.58
    También maneja valores ya numéricos o None.
    """
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        try:
            return float(str(val).replace("$", "").replace(",", "").strip())
        except Exception:
            return 0.0


def _parse_ts(ts_str: str) -> Optional[datetime]:
    """Parsea string de timestamp a datetime UTC. Retorna None si falla."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(str(ts_str).replace(" ", "T"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        return None


def _age_minutes(ts_str: str) -> int:
    """Minutos desde ts_str hasta ahora."""
    dt = _parse_ts(ts_str)
    if dt is None:
        return 0
    return max(0, int((datetime.now(timezone.utc) - dt).total_seconds() / 60))


# ──────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

class LotTracker:
    """
    Tracking en tiempo real de lotes abiertos con persistencia a disco.

    Uso típico (en DecisionEngine):
        self._tracker = LotTracker()
        # cada loop, antes de decide():
        self._tracker.update(df_inventory, price_now, regimen_now)
        max_pnl = self._tracker.get_all_max_pnl()
    """

    def __init__(self, data_dir: Optional[Path] = None):
        base             = data_dir or (Path(__file__).parent / "lot_history")
        self._open_dir   = base / "open"
        self._closed_dir = base / "closed"
        self._open_dir.mkdir(parents=True, exist_ok=True)
        self._closed_dir.mkdir(parents=True, exist_ok=True)

        # Estado en memoria por lote
        self._state:         Dict[str, dict]              = {}
        self._pnl_hist:      Dict[str, Deque[Tuple]]      = {}  # (ts_epoch, pnl)
        self._last_snap_ts:  Dict[str, float]             = {}
        self._known_lots:    Set[str]                     = set()

        # Estrategia del próximo lote detectado (S1 o S2).
        # Se setea cuando decide() aprueba un BUY; se consume en _handle_open().
        self._next_lot_strategy: str = "S1"

        # Restaurar estado persistido desde disco
        self._restore_from_disk()
        log.info(f"LotTracker inicializado · {len(self._state)} lote(s) restaurados desde disco")

    # ──────────────────────────────────────────────────────────────────────────
    # PERSISTENCIA
    # ──────────────────────────────────────────────────────────────────────────

    def _state_path(self, lot_id: str, closed: bool = False) -> Path:
        d = self._closed_dir if closed else self._open_dir
        return d / f"lot_{lot_id}_state.json"

    def _snap_path(self, lot_id: str, closed: bool = False) -> Path:
        d = self._closed_dir if closed else self._open_dir
        return d / f"lot_{lot_id}.jsonl"

    def _save_state(self, lot_id: str) -> None:
        """Persiste el estado del lote (max_pnl, timestamps, etc.) en disco."""
        try:
            state = self._state.get(lot_id, {})
            self._state_path(lot_id).write_text(
                json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"No se pudo guardar estado de {lot_id}: {e}")

    def _append_snapshot(self, lot_id: str, snap: dict, closed: bool = False) -> None:
        """Escribe una línea de snapshot al JSONL del lote."""
        try:
            path = self._snap_path(lot_id, closed)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snap, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            log.warning(f"No se pudo escribir snapshot de {lot_id}: {e}")

    def _restore_from_disk(self) -> None:
        """Al arranque: restaura max_pnl_seen y metadatos de lotes abiertos desde disco."""
        for state_file in self._open_dir.glob("lot_*_state.json"):
            try:
                state  = json.loads(state_file.read_text(encoding="utf-8"))
                lot_id = state.get("lot_id", "")
                if not lot_id:
                    continue
                self._state[lot_id]        = state
                self._pnl_hist[lot_id]     = deque(maxlen=_PNL_VELOCITY_WINDOW)
                self._last_snap_ts[lot_id] = 0.0   # forzar snapshot pronto
                self._known_lots.add(lot_id)
                log.info(
                    f"  ↺ Restaurado {lot_id} · "
                    f"max_pnl={state.get('max_pnl_seen', 0):.3f}% · "
                    f"abierto={state.get('open_ts', '?')}"
                )
            except Exception as e:
                log.warning(f"Error restaurando {state_file.name}: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # UPDATE — LLAMAR CADA LOOP (10s)
    # ──────────────────────────────────────────────────────────────────────────

    def update(self, df_inventory, price_now: float, regimen_actual: str) -> None:
        """
        Actualiza el estado de todos los lotes OPEN.

        Llamar cada loop (10s) DESPUÉS de que df_inventory tenga pnl_pct_actual.
        No lanza excepciones — errores se loguean y el bot continúa.
        """
        now_ts  = time.time()
        now_str = _utc_now_str()

        try:
            import pandas as pd

            if df_inventory is None or df_inventory.empty:
                # Si no hay inventario, cerrar todos los que estaban abiertos
                for lid in list(self._known_lots):
                    self._handle_close(lid, price_now, regimen_actual, now_str, now_ts)
                self._known_lots.clear()
                return

            open_mask   = df_inventory["status"].astype(str).str.upper().isin(["OPEN", "PARTIAL"])
            open_df     = df_inventory[open_mask]
            current_ids = set(open_df["lot_id"].astype(str))

            # Detectar cierres (lotes que ya no están en el inventario)
            closed_ids = self._known_lots - current_ids
            for lid in closed_ids:
                self._handle_close(lid, price_now, regimen_actual, now_str, now_ts)

            # Actualizar lotes activos
            for _, row in open_df.iterrows():
                lid      = str(row.get("lot_id", ""))
                pnl_now  = _to_float(row.get("pnl_pct_actual", 0.0))
                entry_px = _to_float(row.get("price_usdt", 0.0))
                open_dt  = str(row.get("datetime") or row.get("date") or "")

                if lid not in self._known_lots:
                    self._handle_open(lid, pnl_now, entry_px, open_dt, regimen_actual, now_str, now_ts)
                else:
                    self._handle_tick(lid, pnl_now, entry_px, price_now, regimen_actual, now_str, now_ts)

            self._known_lots = current_ids

        except Exception as e:
            log.error(f"LotTracker.update() falló (no crítico): {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # HANDLERS DE EVENTOS
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_open(
        self,
        lot_id: str,
        pnl: float,
        entry_px: float,
        open_date: str,
        regimen: str,
        now_str: str,
        now_ts: float,
    ) -> None:
        """Nuevo lote detectado por primera vez."""
        # Consumir la estrategia pendiente (seteada en DecisionEngine cuando aprobó el BUY)
        lot_strategy             = self._next_lot_strategy
        self._next_lot_strategy  = "S1"  # reset para el siguiente lote

        state = {
            "lot_id":             lot_id,
            "strategy":           lot_strategy,
            "max_pnl_seen":       round(pnl, 4),
            "max_pnl_ts":         now_str,
            "open_ts":            open_date or now_str,
            "open_price":         entry_px,
            "last_pnl":           round(pnl, 4),
            "last_update_ts":     now_str,
            "last_regimen":       regimen,
            "regimen_enter_ts":   now_str,
            "crossed_thresholds": [],
            "status":             "OPEN",
        }
        self._state[lot_id]        = state
        self._pnl_hist[lot_id]     = deque(maxlen=_PNL_VELOCITY_WINDOW)
        self._pnl_hist[lot_id].append((now_ts, pnl))
        self._last_snap_ts[lot_id] = now_ts

        self._save_state(lot_id)
        self._append_snapshot(lot_id, self._build_snap(lot_id, pnl, entry_px, regimen, "open", now_str, now_ts))
        log.info(f"LotTracker: ✚ nuevo lote {lot_id} [{lot_strategy}] · entrada=${entry_px:,.0f} · régimen={regimen}")

    def _handle_tick(
        self,
        lot_id: str,
        pnl: float,
        entry_px: float,
        price_now: float,
        regimen: str,
        now_str: str,
        now_ts: float,
    ) -> None:
        """Tick de actualización para un lote ya conocido."""
        state = self._state.setdefault(lot_id, {})

        # Actualizar max_pnl_seen (y persistir si cambió)
        prev_max = state.get("max_pnl_seen", -999.0)
        if pnl > prev_max:
            state["max_pnl_seen"] = round(pnl, 4)
            state["max_pnl_ts"]   = now_str

        # Historial para velocidad
        hist = self._pnl_hist.setdefault(lot_id, deque(maxlen=_PNL_VELOCITY_WINDOW))
        hist.append((now_ts, pnl))

        # Detectar cambio de régimen
        prev_regimen   = state.get("last_regimen", regimen)
        regime_changed = prev_regimen != regimen
        if regime_changed:
            state["last_regimen"]   = regimen
            state["regimen_enter_ts"] = now_str

        # Detectar cruce de umbral PnL
        threshold_crossed = self._check_threshold(lot_id, pnl, state)

        # Actualizar estado base
        state["last_pnl"]       = round(pnl, 4)
        state["last_update_ts"] = now_str
        self._state[lot_id]     = state

        # Decidir si es momento de hacer snapshot
        elapsed    = now_ts - self._last_snap_ts.get(lot_id, 0)
        should_snap = elapsed >= _SNAPSHOT_INTERVAL_S or regime_changed or threshold_crossed

        if should_snap:
            if regime_changed:
                event = "regime_change"
            elif threshold_crossed:
                event = "pnl_threshold"
            else:
                event = "tick"

            self._append_snapshot(
                lot_id,
                self._build_snap(lot_id, pnl, entry_px, regimen, event, now_str, now_ts)
            )
            self._save_state(lot_id)
            self._last_snap_ts[lot_id] = now_ts

    def _handle_close(
        self,
        lot_id: str,
        price_now: float,
        regimen: str,
        now_str: str,
        now_ts: float,
    ) -> None:
        """Lote cerrado — archivar en closed/ y limpiar memoria."""
        state = self._state.get(lot_id)
        if not state:
            return

        entry_px  = _to_float(state.get("open_price", 0.0))
        pnl_close = _to_float(state.get("last_pnl",   0.0))

        # Snapshot de cierre (incluye max_pnl_seen final)
        snap               = self._build_snap(lot_id, pnl_close, entry_px, regimen, "close", now_str, now_ts)
        snap["final_max_pnl_seen"] = state.get("max_pnl_seen", 0.0)

        # Escribir snapshot final en open antes de archivar
        self._append_snapshot(lot_id, snap, closed=False)

        # Mover JSONL y state a closed/
        try:
            open_snap  = self._snap_path(lot_id, closed=False)
            open_state = self._state_path(lot_id, closed=False)

            if open_snap.exists():
                open_snap.rename(self._snap_path(lot_id, closed=True))

            if open_state.exists():
                state["closed_ts"] = now_str
                state["status"]    = "CLOSED"
                self._state_path(lot_id, closed=True).write_text(
                    json.dumps(state, ensure_ascii=False, default=str), encoding="utf-8"
                )
                open_state.unlink()

        except Exception as e:
            log.warning(f"Error archivando lote {lot_id}: {e}")

        # Limpiar de memoria
        self._state.pop(lot_id, None)
        self._pnl_hist.pop(lot_id, None)
        self._last_snap_ts.pop(lot_id, None)
        log.info(
            f"LotTracker: ✖ lote {lot_id} cerrado · "
            f"max_pnl={snap.get('final_max_pnl_seen', 0):.3f}% · "
            f"pnl_final={pnl_close:.3f}%"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS INTERNOS
    # ──────────────────────────────────────────────────────────────────────────

    def _check_threshold(self, lot_id: str, pnl: float, state: dict) -> bool:
        """
        Verifica si se cruzó un nuevo umbral de PnL.
        Retorna True si se cruzó al menos uno que no estaba marcado.
        """
        crossed = state.get("crossed_thresholds", [])
        for thr in _PNL_THRESHOLDS:
            if thr not in crossed:
                if (thr >= 0 and pnl >= thr) or (thr < 0 and pnl <= thr):
                    crossed.append(thr)
                    state["crossed_thresholds"] = crossed
                    return True
        return False

    def _build_snap(
        self,
        lot_id: str,
        pnl: float,
        entry_px: float,
        regimen: str,
        event: str,
        now_str: str,
        now_ts: float,
    ) -> dict:
        """Construye el dict de snapshot para guardar en JSONL."""
        state          = self._state.get(lot_id, {})
        age_min        = _age_minutes(state.get("open_ts", now_str))
        age_in_reg_min = _age_minutes(state.get("regimen_enter_ts", now_str))

        return {
            "lot_id":                  lot_id,
            "ts":                      now_str,
            "event":                   event,
            "pnl_pct":                 round(pnl, 4),
            "max_pnl_seen":            round(state.get("max_pnl_seen", pnl), 4),
            "price_entry":             entry_px,
            "regimen":                 regimen,
            "age_minutes":             age_min,
            "age_in_regimen_minutes":  age_in_reg_min,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def get_max_pnl(self, lot_id: str) -> float:
        """Máximo PnL histórico para un lote. 0.0 si no hay datos."""
        return float(self._state.get(lot_id, {}).get("max_pnl_seen", 0.0))

    def get_all_max_pnl(self) -> Dict[str, float]:
        """
        Dict {lot_id: max_pnl_seen} para pasar a decide() como max_pnl_seen.
        Reemplaza self._max_pnl_seen de DecisionEngine.
        """
        return {lid: float(s.get("max_pnl_seen", 0.0)) for lid, s in self._state.items()}

    def get_lot_context(self, lot_id: str) -> dict:
        """
        Contexto enriquecido del lote para decision_engine.

        Retorna:
            age_minutes            — tiempo desde apertura del lote
            max_pnl_seen           — máximo PnL histórico (persiste reinicios)
            pnl_velocity_10m       — % por minuto en últimos 10 min (+ sube, - baja)
            age_ratio              — None aquí; se completa en lot_context.py con analytics
            time_since_peak_minutes— minutos desde el pico máximo de PnL
            last_regimen           — régimen actual del lote
            crossed_thresholds     — umbrales de PnL que ya cruzó
        """
        state = self._state.get(lot_id)
        if not state:
            return {}

        # age_minutes
        age_min = _age_minutes(state.get("open_ts", ""))

        # pnl_velocity_10m — (pnl_ahora - pnl_más_antiguo_en_ventana) / minutos_transcurridos
        velocity = 0.0
        hist     = self._pnl_hist.get(lot_id)
        if hist and len(hist) >= 2:
            oldest_ts, oldest_pnl = hist[0]
            newest_ts, newest_pnl = hist[-1]
            elapsed_min = (newest_ts - oldest_ts) / 60.0
            if elapsed_min > 0:
                velocity = round((newest_pnl - oldest_pnl) / elapsed_min, 4)

        # time_since_peak_minutes
        time_since_peak = _age_minutes(state.get("max_pnl_ts", ""))

        return {
            "age_minutes":             age_min,
            "max_pnl_seen":            state.get("max_pnl_seen", 0.0),
            "pnl_velocity_10m":        velocity,
            "age_ratio":               None,   # completado por lot_context.py
            "time_since_peak_minutes": time_since_peak,
            "last_regimen":            state.get("last_regimen", ""),
            "crossed_thresholds":      state.get("crossed_thresholds", []),
        }

    def n_open(self) -> int:
        """Cantidad de lotes que el tracker tiene en memoria."""
        return len(self._state)

    # ──────────────────────────────────────────────────────────────────────────
    # GESTIÓN DE ESTRATEGIA S1/S2
    # ──────────────────────────────────────────────────────────────────────────

    def set_next_lot_strategy(self, strategy: str) -> None:
        """
        Registra la estrategia ("S1" o "S2") que se asignará al PRÓXIMO lote
        que sea detectado como nuevo en _handle_open().

        Llamar desde DecisionEngine.decide() justo después de que el motor
        apruebe un BUY, antes de que el lote aparezca en df_inventory.
        """
        self._next_lot_strategy = strategy

    def get_all_strategies(self) -> Dict[str, str]:
        """
        Dict {lot_id: strategy} para pasar a decide() como lot_strategies.
        Lotes sin campo strategy (ej. creados antes de esta versión) → "S1".
        """
        return {lid: str(s.get("strategy", "S1")) for lid, s in self._state.items()}
