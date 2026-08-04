import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

BIAS_MAX_AGE_SECONDS = 24 * 60 * 60  # 24 hours


class DexterGate:
    """
    Simple hook to allow/deny trades based on a bias file produced by Dexter.
    If dexter_bias.json exists and marks a ticker as 'avoid', we block the trade.
    """

    def __init__(self, bias_path: str = "dexter_bias.json"):
        self.bias_path = Path(bias_path)
        self.bias_mtime = None
        self._stale_warned = False
        self.bias = self._load_bias()

    def _load_bias(self) -> Dict[str, Any]:
        if self.bias_path.exists():
            try:
                self.bias_mtime = self.bias_path.stat().st_mtime
                return json.loads(self.bias_path.read_text())
            except Exception:
                return {}
        return {}

    def _is_stale(self) -> bool:
        if self.bias_mtime is None:
            return False
        return (time.time() - self.bias_mtime) > BIAS_MAX_AGE_SECONDS

    def refresh(self):
        self.bias = self._load_bias()

    def get_bias(self, ticker: str) -> Dict[str, Any]:
        """
        Get the full bias data for a ticker (for confidence scoring).
        Returns dict with 'bias', 'reasoning', 'fundamentals', etc.
        """
        bias_data = self.bias.get(ticker)
        if bias_data is None:
            return {}
        if isinstance(bias_data, str):
            return {'bias': bias_data}
        if isinstance(bias_data, dict):
            return bias_data
        return {}

    def should_allow(self, ticker: str, trades_remaining: int = 1, context: Dict[str, Any] = None) -> bool:
        """
        Basic gating: if no trades remaining, block; if Dexter says 'avoid', block.
        """
        if trades_remaining <= 0:
            return False
        if self._is_stale():
            if not self._stale_warned:
                logger.warning(
                    "Dexter bias file %s is older than 24 hours — "
                    "ignoring stale bias and allowing trades (fail-open).",
                    self.bias_path,
                )
                self._stale_warned = True
            return True
        bias = self.bias.get(ticker) or {}
        if isinstance(bias, str) and bias.lower().strip() == "avoid":
            return False
        if isinstance(bias, dict):
            if bias.get("decision", "").lower() == "avoid":
                return False
            if bias.get("allow") is False:
                return False
        return True
