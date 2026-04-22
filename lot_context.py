# -*- coding: utf-8 -*-
"""
lot_context.py — M3.5c  Bot BTC V2
Thin wrapper que combina LotTracker + LotAnalytics para decision_engine.

Responsabilidades:
- Expone una API limpia con un solo método útil: get_lot_enriched()
- Combina contexto en tiempo real (tracker) con estadísticas históricas (analytics)
- Calcula age_ratio: qué tan "viejo" está el lote vs. el promedio histórico de su régimen

API pública:
  ctx = LotContext(tracker, analytics)
  ctx.get_lot_enriched(lot_id, regimen) → dict listo para decision_engine
  ctx.get_max_pnl(lot_id)              → float
  ctx.get_all_max_pnl()                → Dict[str, float]

Campos que retorna get_lot_enriched():
  age_minutes             — minutos desde apertura del lote
  max_pnl_seen            — máximo PnL histórico (persiste reinicios del bot)
  pnl_velocity_10m        — variación de PnL por minuto en últimos 10 min
  age_ratio               — age_minutes / lifetime_mediano_del_régimen
                            None si no hay suficientes datos históricos
  time_since_peak_minutes — minutos desde el pico máximo registrado
  last_regimen            — régimen actual del lote
  crossed_thresholds      — lista de umbrales PnL% que ya cruzó el lote
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from lot_tracker   import LotTracker
from lot_analytics import LotAnalytics

log = logging.getLogger("M35.LotContext")


class LotContext:
    """
    API de contexto enriquecido por lote para decision_engine.

    Inicialización (en DecisionEngine.__init__):
        self._lot_ctx = LotContext(self._tracker, self._analytics)

    Uso (en DecisionEngine.decide() o en decide() funcional):
        ctx_lote = self._lot_ctx.get_lot_enriched(lot_id, regimen)
        age_ratio = ctx_lote.get("age_ratio")
    """

    def __init__(self, tracker: LotTracker, analytics: LotAnalytics):
        self._tracker   = tracker
        self._analytics = analytics

    # ──────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────────────

    def get_lot_enriched(self, lot_id: str, regimen: str) -> dict:
        """
        Retorna contexto completo del lote listo para decision_engine.

        Si el lote no existe en el tracker (poco probable pero posible
        si el tracker acaba de arrancar), retorna dict vacío.

        El campo age_ratio es None si analytics no tiene suficientes
        datos históricos para el régimen (n < 5 lotes cerrados).
        En ese caso, las reglas que dependen de age_ratio deben saltearse.
        """
        raw = self._tracker.get_lot_context(lot_id)
        if not raw:
            return {}

        # Completar age_ratio con datos históricos
        age_minutes  = raw.get("age_minutes", 0)
        avg_lifetime = self._analytics.get_avg_lifetime(regimen)

        if avg_lifetime and avg_lifetime > 0:
            age_ratio = round(age_minutes / avg_lifetime, 2)
        else:
            age_ratio = None   # sin datos suficientes → no influye en decisiones

        raw["age_ratio"] = age_ratio
        return raw

    def get_max_pnl(self, lot_id: str) -> float:
        """Máximo PnL visto para el lote. 0.0 si no hay datos."""
        return self._tracker.get_max_pnl(lot_id)

    def get_all_max_pnl(self) -> Dict[str, float]:
        """
        Dict {lot_id: max_pnl_seen} para pasar directamente a decide()
        como argumento max_pnl_seen. Reemplaza self._max_pnl_seen del motor.
        """
        return self._tracker.get_all_max_pnl()
