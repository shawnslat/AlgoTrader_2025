"""
Signal Confidence Scoring System

Calculates trade confidence based on agreement between multiple signal sources:
- Claude sentiment (primary AI)
- Grok bias (secondary AI)
- News sentiment
- Technical indicators
- FinViz fundamentals
- Insider/Congressional trading activity

When signals agree → High confidence → Full position size
When signals conflict → Low confidence → Reduced position or skip
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1


@dataclass
class SignalSource:
    """Individual signal from a data source."""
    name: str
    direction: SignalDirection
    strength: float  # 0.0 to 1.0
    reason: str


@dataclass
class ConfidenceScore:
    """Overall confidence score for a trading decision."""
    ticker: str
    score: float  # 0.0 to 1.0
    direction: SignalDirection
    position_multiplier: float  # How much to scale position (0.0 to 1.0)
    signals: list
    agreement_pct: float
    recommendation: str  # "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "SKIP"


def calculate_confidence(
    ticker: str,
    grok_bias: Optional[Dict] = None,
    sentiment_score: Optional[float] = None,  # -1.0 to 1.0
    technical_signal: Optional[str] = None,  # "BUY", "SELL", "HOLD"
    ml_prediction: Optional[int] = None,  # 1 = up, 0 = down
    finviz_risk: Optional[Dict] = None,
    insider_data: Optional[Dict] = None,
    config: Optional[Dict] = None,
    ma_crossover: Optional[int] = None,  # +1 (MA10>MA50) or -1 (MA10<MA50)
    ibs: Optional[float] = None,  # 0.0-1.0 Internal Bar Strength
    adx: Optional[float] = None,  # ADX value for regime detection
    options_flow: Optional[Dict] = None,  # {direction, strength, reason} from options_flow.get_options_flow
) -> ConfidenceScore:
    """
    Calculate confidence score based on signal agreement.

    Args:
        ticker: Stock ticker symbol
        grok_bias: Grok bias dict with 'direction' key ("bullish", "bearish", "neutral") and 'score' float
        sentiment_score: Aggregated sentiment from Claude (-1.0 bearish to 1.0 bullish)
        technical_signal: Technical indicator signal
        ml_prediction: ML model prediction (1=up, 0=down)
        finviz_risk: FinViz risk assessment with 'risk_score' and 'flags'
        config: Configuration dict with confidence settings

    Returns:
        ConfidenceScore with overall assessment
    """
    config = config or {}
    conf_cfg = config.get('confidence', {})

    # Weights for each signal source (configurable)
    weights = {
        'grok_bias': conf_cfg.get('weight_grok_bias', 0.20),
        'sentiment': conf_cfg.get('weight_sentiment', 0.15),
        'technical': conf_cfg.get('weight_technical', 0.15),
        'ml': conf_cfg.get('weight_ml', 0.10),
        'fundamental': conf_cfg.get('weight_fundamental', 0.05),
        'insider': conf_cfg.get('weight_insider', 0.05),
        'ma_crossover': conf_cfg.get('weight_ma_crossover', 0.15),
        'ibs': conf_cfg.get('weight_ibs', 0.15),
        'options_flow': conf_cfg.get('weight_options_flow', 0.10),
    }

    # Regime detection: adjust momentum vs mean-reversion weights based on ADX
    # ADX > 25 = trending → momentum (MA crossover) signals are reliable, suppress IBS
    # ADX < 15 = choppy/ranging → mean-reversion (IBS) signals are reliable, suppress MA crossover
    # This prevents applying the wrong strategy to the wrong market condition
    regime = 'MIXED'
    if adx is not None:
        base_ma = weights['ma_crossover']
        base_ibs = weights['ibs']
        total_strategy_weight = base_ma + base_ibs  # Keep total constant

        if adx > 25:
            regime = 'TRENDING'
            # Shift weight toward momentum
            weights['ma_crossover'] = total_strategy_weight * 0.85
            weights['ibs'] = total_strategy_weight * 0.15
        elif adx < 15:
            regime = 'MEAN_REVERTING'
            # Shift weight toward mean-reversion
            weights['ma_crossover'] = total_strategy_weight * 0.15
            weights['ibs'] = total_strategy_weight * 0.85
        # else MIXED: keep default 50/50 split

        logger.debug(f"[{ticker}] Regime: {regime} (ADX={adx:.1f}), "
                    f"MA weight={weights['ma_crossover']:.2f}, IBS weight={weights['ibs']:.2f}")

    signals = []

    # 1. Grok Bias Signal (secondary AI opinion from xAI)
    if grok_bias:
        bias_dir = grok_bias.get('direction', 'neutral').lower()
        bias_score = grok_bias.get('score', 0.0)  # -1.0 to 1.0
        if bias_dir == 'bullish' or bias_score > 0.2:
            direction = SignalDirection.BULLISH
            strength = min(1.0, abs(bias_score)) if bias_score else 0.6
        elif bias_dir == 'bearish' or bias_score < -0.2:
            direction = SignalDirection.BEARISH
            strength = min(1.0, abs(bias_score)) if bias_score else 0.6
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.3

        signals.append(SignalSource(
            name='grok_bias',
            direction=direction,
            strength=strength,
            reason=grok_bias.get('reasoning', f"Grok: {bias_dir}")[:100]
        ))

    # 2. Sentiment Signal
    if sentiment_score is not None:
        if sentiment_score > 0.2:
            direction = SignalDirection.BULLISH
            strength = min(1.0, abs(sentiment_score))
        elif sentiment_score < -0.2:
            direction = SignalDirection.BEARISH
            strength = min(1.0, abs(sentiment_score))
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.3

        signals.append(SignalSource(
            name='sentiment',
            direction=direction,
            strength=strength,
            reason=f"Sentiment score: {sentiment_score:.2f}"
        ))

    # 3. Technical Signal
    if technical_signal:
        sig_upper = technical_signal.upper()
        if sig_upper == 'BUY':
            direction = SignalDirection.BULLISH
            strength = 0.7
        elif sig_upper == 'SELL':
            direction = SignalDirection.BEARISH
            strength = 0.7
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.3

        signals.append(SignalSource(
            name='technical',
            direction=direction,
            strength=strength,
            reason=f"Technical: {technical_signal}"
        ))

    # 4. ML Prediction Signal
    if ml_prediction is not None:
        if ml_prediction == 1:
            direction = SignalDirection.BULLISH
            strength = 0.6  # ML gets moderate weight
        else:
            direction = SignalDirection.BEARISH
            strength = 0.6

        signals.append(SignalSource(
            name='ml',
            direction=direction,
            strength=strength,
            reason=f"ML predicts: {'UP' if ml_prediction == 1 else 'DOWN'}"
        ))

    # 5. Fundamental/Risk Signal (from FinViz)
    if finviz_risk:
        risk_score = finviz_risk.get('risk_score', 0)
        flags = finviz_risk.get('risk_flags', finviz_risk.get('flags', []))

        # High risk = bearish signal, low risk = neutral (not bullish by itself)
        # risk_score is on a 0-100 scale
        if risk_score >= 40:  # High risk
            direction = SignalDirection.BEARISH
            strength = min(1.0, risk_score / 100.0)
        elif risk_score <= 1:  # Low risk
            direction = SignalDirection.NEUTRAL  # Low risk doesn't mean buy
            strength = 0.2
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.3

        signals.append(SignalSource(
            name='fundamental',
            direction=direction,
            strength=strength,
            reason=f"Risk: {risk_score}, Flags: {', '.join(flags[:3])}" if flags else f"Risk score: {risk_score}"
        ))

    # 6. Insider/Congress Signal
    if insider_data:
        direction_str = (insider_data.get('direction') or 'NEUTRAL').upper()
        if direction_str == 'BULLISH':
            direction = SignalDirection.BULLISH
        elif direction_str == 'BEARISH':
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        strength = insider_data.get('strength', 0.3)

        signals.append(SignalSource(
            name='insider',
            direction=direction,
            strength=strength,
            reason=insider_data.get('reason', 'Insider/Congress activity')[:100]
        ))

    # 7. MA Crossover Signal (Strategy 3.12: Two Moving Averages)
    #    MA10 > MA50 = uptrend (golden cross) → BULLISH
    #    MA10 < MA50 = downtrend (death cross) → BEARISH
    if ma_crossover is not None:
        if ma_crossover > 0:
            direction = SignalDirection.BULLISH
            strength = 0.8  # Strong trend signal
        else:
            direction = SignalDirection.BEARISH
            strength = 0.8

        signals.append(SignalSource(
            name='ma_crossover',
            direction=direction,
            strength=strength,
            reason=f"MA10 {'>' if ma_crossover > 0 else '<'} MA50 ({'uptrend' if ma_crossover > 0 else 'downtrend'})"
        ))

    # 8. IBS Mean-Reversion Signal (Strategy 4.4: Internal Bar Strength)
    #    IBS < 0.2 → closed near daily low → oversold → BULLISH (buy the dip)
    #    IBS > 0.8 → closed near daily high → overbought → BEARISH (avoid/sell)
    #    Middle range → NEUTRAL
    #    ONLY valid in ranging/choppy markets (ADX < 25). In strong trends,
    #    mean-reversion is wrong — dips keep dipping, breakouts keep running.
    if ibs is not None:
        ibs_adx = adx if adx is not None else 20  # Default to moderate if unknown
        if ibs_adx >= 25:
            # Strong trend — suppress mean-reversion, treat as neutral
            direction = SignalDirection.NEUTRAL
            strength = 0.1
            ibs_reason = f"IBS={ibs:.2f} (suppressed, ADX={ibs_adx:.0f} trending)"
        elif ibs < 0.2:
            direction = SignalDirection.BULLISH
            strength = 0.7 + (0.2 - ibs)
            ibs_reason = f"IBS={ibs:.2f} (oversold, ADX={ibs_adx:.0f} ranging)"
        elif ibs > 0.8:
            direction = SignalDirection.BEARISH
            strength = 0.7 + (ibs - 0.8)
            ibs_reason = f"IBS={ibs:.2f} (overbought, ADX={ibs_adx:.0f} ranging)"
        else:
            direction = SignalDirection.NEUTRAL
            strength = 0.2
            ibs_reason = f"IBS={ibs:.2f} (neutral)"

        signals.append(SignalSource(
            name='ibs',
            direction=direction,
            strength=min(1.0, strength),
            reason=ibs_reason
        ))

    # 9. Options Flow Signal (front-month P/C ratio + call/put volume share)
    #    Bullish if low P/C with elevated call vol; bearish if high P/C with elevated put vol.
    #    Falls back to None for crypto / illiquid tickers — those signals are simply absent.
    if options_flow:
        of_dir = (options_flow.get('direction') or 'neutral').lower()
        if of_dir == 'bullish':
            direction = SignalDirection.BULLISH
        elif of_dir == 'bearish':
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL
        of_strength = float(options_flow.get('strength', 0.3))
        signals.append(SignalSource(
            name='options_flow',
            direction=direction,
            strength=min(1.0, of_strength),
            reason=options_flow.get('reason', f"Options flow: {of_dir}")[:100],
        ))

    # Calculate agreement and confidence
    if not signals:
        return ConfidenceScore(
            ticker=ticker,
            score=0.0,
            direction=SignalDirection.NEUTRAL,
            position_multiplier=0.0,
            signals=[],
            agreement_pct=0.0,
            recommendation="SKIP"
        )

    # Count directions
    bullish_count = sum(1 for s in signals if s.direction == SignalDirection.BULLISH)
    bearish_count = sum(1 for s in signals if s.direction == SignalDirection.BEARISH)
    neutral_count = sum(1 for s in signals if s.direction == SignalDirection.NEUTRAL)

    total_signals = len(signals)

    # Determine dominant direction from WEIGHTED sums so direction and
    # weighted score can never contradict each other
    bullish_weight = sum(weights.get(s.name, 0.1) * s.strength
                         for s in signals if s.direction == SignalDirection.BULLISH)
    bearish_weight = sum(weights.get(s.name, 0.1) * s.strength
                         for s in signals if s.direction == SignalDirection.BEARISH)

    if bullish_weight > bearish_weight:
        dominant_direction = SignalDirection.BULLISH
        agreement_count = bullish_count
    elif bearish_weight > bullish_weight:
        dominant_direction = SignalDirection.BEARISH
        agreement_count = bearish_count
    else:
        dominant_direction = SignalDirection.NEUTRAL
        agreement_count = neutral_count

    agreement_pct = agreement_count / total_signals

    # Calculate weighted confidence score
    # Each signal contributes: +weight*strength if agrees, -weight*strength if disagrees, +weight*0.1 if neutral
    # Final score is normalized to 0-1 without artificial inflation
    weighted_score = 0.0
    total_weight = 0.0

    for signal in signals:
        weight = weights.get(signal.name, 0.1)
        if signal.direction == dominant_direction:
            weighted_score += weight * signal.strength
        elif signal.direction == SignalDirection.NEUTRAL:
            weighted_score += weight * 0.1  # Neutral contributes very little
        else:
            weighted_score -= weight * signal.strength  # Full penalty for disagreement
        total_weight += weight

    # Normalize: ratio of actual score to max possible score (all signals agree at strength 1.0)
    if total_weight > 0:
        confidence_score = max(0.0, min(1.0, weighted_score / total_weight))
    else:
        confidence_score = 0.0

    # Determine position multiplier based on confidence
    min_confidence = conf_cfg.get('min_confidence_to_trade', 0.4)
    full_confidence = conf_cfg.get('full_confidence_threshold', 0.7)

    if confidence_score < min_confidence:
        position_multiplier = 0.0  # Don't trade
    elif confidence_score >= full_confidence:
        position_multiplier = 1.0  # Full position
    else:
        # Linear scale between min and full
        position_multiplier = (confidence_score - min_confidence) / (full_confidence - min_confidence)

    # Generate recommendation
    if position_multiplier == 0.0:
        recommendation = "SKIP"
    elif dominant_direction == SignalDirection.BULLISH:
        if confidence_score >= 0.7:
            recommendation = "STRONG_BUY"
        else:
            recommendation = "BUY"
    elif dominant_direction == SignalDirection.BEARISH:
        if confidence_score >= 0.7:
            recommendation = "STRONG_SELL"
        else:
            recommendation = "SELL"
    else:
        recommendation = "HOLD"

    result = ConfidenceScore(
        ticker=ticker,
        score=confidence_score,
        direction=dominant_direction,
        position_multiplier=position_multiplier,
        signals=signals,
        agreement_pct=agreement_pct,
        recommendation=recommendation
    )

    logger.info(
        f"[{ticker}] Confidence: {confidence_score:.2f}, "
        f"Agreement: {agreement_pct:.0%}, "
        f"Direction: {dominant_direction.name}, "
        f"Position: {position_multiplier:.0%}, "
        f"Recommendation: {recommendation}"
    )

    return result


def get_aggregated_sentiment(sentiment_results: list, ticker: str) -> Optional[float]:
    """
    Aggregate sentiment results for a specific ticker.

    Args:
        sentiment_results: List of sentiment dicts with 'ticker', 'label', 'magnitude', 'recency_weight'
        ticker: Ticker to filter for

    Returns:
        Aggregated sentiment score from -1.0 (bearish) to 1.0 (bullish)
    """
    ticker_sentiments = [s for s in sentiment_results if s.get('ticker') == ticker]

    if not ticker_sentiments:
        return None

    weighted_sum = 0.0
    total_weight = 0.0

    for s in ticker_sentiments:
        label = s.get('label', 'Neutral')
        magnitude = s.get('magnitude', 1.0)
        recency = s.get('recency_weight', 0.5)

        # Convert label to numeric
        if label == 'Positive':
            value = 1.0
        elif label == 'Negative':
            value = -1.0
        else:
            value = 0.0

        # Weight by magnitude and recency
        weight = magnitude * recency
        weighted_sum += value * weight
        total_weight += weight

    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def format_confidence_report(scores: list) -> str:
    """
    Format confidence scores into a readable report.

    Args:
        scores: List of ConfidenceScore objects

    Returns:
        Formatted string report
    """
    if not scores:
        return "No confidence scores calculated."

    lines = ["=" * 60, "SIGNAL CONFIDENCE REPORT", "=" * 60, ""]

    # Sort by confidence score descending
    sorted_scores = sorted(scores, key=lambda x: x.score, reverse=True)

    for cs in sorted_scores:
        direction_emoji = "🟢" if cs.direction == SignalDirection.BULLISH else "🔴" if cs.direction == SignalDirection.BEARISH else "⚪"

        lines.append(f"{direction_emoji} {cs.ticker}: {cs.recommendation}")
        lines.append(f"   Confidence: {cs.score:.0%} | Agreement: {cs.agreement_pct:.0%} | Position: {cs.position_multiplier:.0%}")

        # Show individual signals
        for sig in cs.signals:
            sig_icon = "↑" if sig.direction == SignalDirection.BULLISH else "↓" if sig.direction == SignalDirection.BEARISH else "→"
            lines.append(f"   {sig_icon} {sig.name}: {sig.reason[:50]}")

        lines.append("")

    # Summary
    buy_count = sum(1 for cs in scores if cs.recommendation in ["BUY", "STRONG_BUY"])
    sell_count = sum(1 for cs in scores if cs.recommendation in ["SELL", "STRONG_SELL"])
    skip_count = sum(1 for cs in scores if cs.recommendation in ["SKIP", "HOLD"])

    lines.append("-" * 60)
    lines.append(f"Summary: {buy_count} BUY | {sell_count} SELL | {skip_count} SKIP/HOLD")
    lines.append("=" * 60)

    return "\n".join(lines)
