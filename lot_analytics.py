# -*- coding: utf-8 -*-
"""
lot_analytics.py — M3.5b  Bot BTC V2
Estadísticas históricas de lotes cerrados, por régimen de apertura.

Responsabilidades:
- Lee todos los lotes cerrados en lot_history/closed/
- Calcula percentiles de lifetime, time_to_peak, pnl_peak y outcomes por régimen
- Cachea resultados en lot_history/lot_analytics_cache.json
- Se recalcula: al cerrar un lote (refresh_on_close) o cada 24h (maybe_refresh)

Modo aprendizaje:
- Si n_lotes < MIN_LOTS_FOR_STATS para un régimen → has_enough_data=False
  y lot_context.py no usará esas stats para influir en decisiones

API pública:
  analytics.maybe_refresh()        — recalcular si pasaron 24h
  analytics.refresh_on_close()     — recalcular inmediatamente (al cerrar lote)
  analytics.get_regime_stats(reg)  — dict con stats o None si sin datos
  analytics.get_avg_lifetime(reg)  — float minutos o None
"""

from __future__ import annotations

import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("M35.LotAnalytics")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
_CACHE_FILE            = "lot_analytics_cache.json"
_REFRESH_INTERVAL_S    = 86_400   # 24 horas
_MIN_LOTS_FOR_STATS    = 5        # mínimo para que las stats sean significativas


def _parse_ts(ts_str: str) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(str(ts_str).replace(" ", "T"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        return None


def _percentile(data: List[float], p: float) -> float:
    """Percentil simple sin numpy (para no añadir dependencia)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n    = len(sorted_data)
    idx  = (p / 100.0) * (n - 1)
    lo   = int(idx)
    hi   = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


# ──────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

class LotAnalytics:
    """
    Estadísticas históricas de lotes cerrados agrupadas por régimen de apertura.

    Uso típico:
        analytics = LotAnalytics()
        # cada 60s (force_ohlcv):
        analytics.maybe_refresh()
        # al cerrar un lote (desde accounting o loop):
        analytics.refresh_on_close()
    """

    def __init__(self, data_dir: Optional[Path] = None):
        base              = data_dir or (Path(__file__).parent / "lot_history")
        self._closed_dir  = base / "closed"
        self._cache_path  = base / _CACHE_FILE
        self._cache:      Dict = {}
        self._last_refresh_ts: float = 0.0

        self._closed_dir.mkdir(parents=True, exist_ok=True)
        self._load_cache()
        log.info(
            f"LotAnalytics inicializado · {len(self._cache)} régimen(es) en caché · "
            f"n_lotes_cerrados={self._count_closed()}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # CACHÉ
    # ──────────────────────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        try:
            if self._cache_path.exists():
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                self._cache            = data.get("stats", {})
                self._last_refresh_ts  = float(data.get("refresh_ts", 0.0))
        except Exception as e:
            log.warning(f"No se pudo cargar analytics cache: {e}")

    def _save_cache(self) -> None:
        try:
            data = {
                "refresh_ts": self._last_refresh_ts,
                "stats":      self._cache,
            }
            self._cache_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8"
            )
        except Exception as e:
            log.warning(f"No se pudo guardar analytics cache: {e}")

    def _count_closed(self) -> int:
        if not self._closed_dir.exists():
            return 0
        return sum(1 for _ in self._closed_dir.glob("lot_*_state.json"))

    # ──────────────────────────────────────────────────────────────────────────
    # REFRESCO
    # ──────────────────────────────────────────────────────────────────────────

    def maybe_refresh(self) -> None:
        """Recalcular si pasaron más de 24h desde el último refresh."""
        if time.time() - self._last_refresh_ts >= _REFRESH_INTERVAL_S:
            self.refresh()

    def refresh_on_close(self) -> None:
        """Llamar cada vez que se cierra un lote para actualizar inmediatamente."""
        self.refresh()

    def refresh(self) -> None:
        """
        Lee todos los lotes cerrados y recalcula estadísticas por régimen.
        Guarda resultado en caché.
        """
        if not self._closed_dir.exists():
            log.info("LotAnalytics.refresh(): carpeta closed/ vacía, nada que calcular")
            return

        stats_by_regime: Dict[str, List[dict]] = {}
        n_procesados = 0
        n_errores    = 0

        for state_file in self._closed_dir.glob("lot_*_state.json"):
            try:
                state  = json.loads(state_file.read_text(encoding="utf-8"))
                lot_id = state.get("lot_id", "")
                if not lot_id:
                    continue

                # Buscar el JSONL del lote
                snap_file = self._closed_dir / f"lot_{lot_id}.jsonl"
                if not snap_file.exists():
                    continue

                snaps = []
                with open(snap_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                snaps.append(json.loads(line))
                            except Exception:
                                pass

                if not snaps:
                    continue

                # Régimen de apertura = primer snapshot con event='open'
                open_snap      = next((s for s in snaps if s.get("event") == "open"), snaps[0])
                regime_apertura = open_snap.get("regimen", "_unknown")

                # Timestamps
                open_ts_str   = state.get("open_ts",   snaps[0].get("ts", ""))
                closed_ts_str = state.get("closed_ts", snaps[-1].get("ts", ""))
                max_pnl_ts_str = state.get("max_pnl_ts", "")

                open_dt   = _parse_ts(open_ts_str)
                closed_dt = _parse_ts(closed_ts_str)
                max_pnl_dt = _parse_ts(max_pnl_ts_str)

                # Lifetime total en minutos
                if open_dt and closed_dt:
                    lifetime_min = max(0, int((closed_dt - open_dt).total_seconds() / 60))
                else:
                    lifetime_min = 0

                # Tiempo hasta pico en minutos
                if open_dt and max_pnl_dt:
                    time_to_peak_min = max(0, int((max_pnl_dt - open_dt).total_seconds() / 60))
                else:
                    time_to_peak_min = lifetime_min // 2

                # Outcome: buscar rule_id en snapshot de cierre
                close_snap = next((s for s in snaps if s.get("event") == "close"), snaps[-1])
                outcome    = close_snap.get("rule_id", state.get("rule_id_trigger", "UNKNOWN"))

                lot_data = {
                    "lot_id":           lot_id,
                    "lifetime_min":     lifetime_min,
                    "time_to_peak_min": time_to_peak_min,
                    "max_pnl":          float(state.get("max_pnl_seen", 0.0)),
                    "final_pnl":        float(state.get("last_pnl", 0.0)),
                    "outcome":          outcome,
                }

                stats_by_regime.setdefault(regime_apertura, []).append(lot_data)
                n_procesados += 1

            except Exception as e:
                n_errores += 1
                log.warning(f"Error procesando {state_file.name}: {e}")

        # ── Calcular estadísticas por régimen ──────────────────────────────────
        new_cache: Dict = {}
        for regime, lots in stats_by_regime.items():
            n = len(lots)
            if n < 1:
                continue

            lifetimes      = [l["lifetime_min"]     for l in lots]
            time_to_peaks  = [l["time_to_peak_min"] for l in lots]
            max_pnls       = [l["max_pnl"]           for l in lots]

            outcomes: Dict[str, int] = {}
            for l in lots:
                outcomes[l["outcome"]] = outcomes.get(l["outcome"], 0) + 1

            n_success = sum(1 for l in lots if l["final_pnl"] > 0)

            new_cache[regime] = {
                "n_lotes":              n,
                "has_enough_data":      n >= _MIN_LOTS_FOR_STATS,
                # Lifetime
                "lifetime_median_min":  int(_percentile(lifetimes,     50)),
                "lifetime_p75_min":     int(_percentile(lifetimes,     75)),
                "lifetime_p90_min":     int(_percentile(lifetimes,     90)),
                # Tiempo hasta pico
                "time_to_peak_p50_min": int(_percentile(time_to_peaks, 50)),
                "time_to_peak_p75_min": int(_percentile(time_to_peaks, 75)),
                # PnL en pico
                "pnl_peak_p50":         round(_percentile(max_pnls,    50), 3),
                "pnl_peak_p75":         round(_percentile(max_pnls,    75), 3),
                # Resultados
                "success_rate":         round(n_success / n, 3),
                "outcomes":             outcomes,
            }

        self._cache            = new_cache
        self._last_refresh_ts  = time.time()
        self._save_cache()

        log.info(
            f"LotAnalytics recalculado · {len(new_cache)} régimen(es) · "
            f"{n_procesados} lotes procesados · {n_errores} errores"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def get_regime_stats(self, regime: str) -> Optional[dict]:
        """
        Estadísticas para un régimen.
        Retorna None si no hay datos suficientes (n < MIN_LOTS_FOR_STATS).
        """
        stats = self._cache.get(regime)
        if stats and stats.get("has_enough_data", False):
            return stats
        return None

    def get_avg_lifetime(self, regime: str) -> Optional[float]:
        """
        Lifetime mediano (en minutos) para el régimen.
        Retorna None si no hay suficientes datos.
        """
        stats = self.get_regime_stats(regime)
        if stats:
            return float(stats.get("lifetime_median_min", 0))
        return None

    def summary(self) -> str:
        """String resumen para logs."""
        lines = [f"LotAnalytics ({len(self._cache)} regímenes):"]
        for regime, s in sorted(self._cache.items()):
            n     = s.get("n_lotes", 0)
            flag  = "✓" if s.get("has_enough_data") else "⚠ (pocos datos)"
            med   = s.get("lifetime_median_min", 0)
            peak  = s.get("pnl_peak_p50", 0)
            srate = s.get("success_rate", 0)
            lines.append(
                f"  {regime:<22} n={n:>3} {flag} · "
                f"lifetime_med={med:>4}min · pnl_peak_p50={peak:.2f}% · success={srate:.0%}"
            )
        return "\n".join(lines)
