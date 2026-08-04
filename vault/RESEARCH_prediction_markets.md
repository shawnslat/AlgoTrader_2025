# Prediction-Market Strategy Research (distilled from collected articles, 2026-07)

Distillation of Shawn's collected material on profitable Polymarket/Kalshi bots (esp. short-duration BTC Up/Down markets) plus the open-source landscape. Kept here as the design reference for SEER upgrades.

## The canonical decision chain

market data → signal → fair value → executable edge → position structure → execution → risk

The core insight: the edge is rarely "predicting BTC." It's recognizing when an outcome's *posted price* no longer reflects the latest information, and knowing whether the displayed price is actually executable at size.

## Fair value

- Bayesian updating: posterior P(Up) from prior × P(signal|Up) / normalizer. SEER's `probability.py` intentionally returns the mid (no model = no directional trades) — correct discipline until a real model exists.
- Signal double-counting warning: a BTC spike creates volume + aggressive buys + bid-side depth + correlated ETH/SOL moves — four features, one event. Weight by incremental information, not count. (Same disease as the trader's news cluster.)

## Executable edge (not displayed edge)

net_edge = fair_value − expected VWAP fill − fees − slippage − safety buffer.
Evaluate at volume-weighted fill price for the intended size, not top-of-book. An arb only exists if EVERY leg fills: Up 46¢ + Down 48¢ = 94¢ pair is 6¢ gross — but if Down actually fills at 53¢ the pair costs 99¢ before fees.
**Status in SEER:** fee model is good (rate·P·(1−P)); depth-awareness is missing everywhere; the live Poly US path detected on midpoints until the 2026-07-24 fix.

## Five position structures (from the $50k/mo bot analysis)

1. **Temporal arbitrage** — build the two legs at different moments (buy Down cheap on a spike, buy Up cheap on the retrace; pair costs <$1 without both prices ever coexisting). Risk: the second leg never comes → naked inventory. Build in small alternating blocks.
2. **Hedged directional** — matched inventory core + small directional excess sized to signal strength. Risk: pair cost >$1 means the excess must overcome a structural deficit.
3. **Inventory market making** — manage avg cost per side across multiple markets; sell the 98¢ favorite early to recycle capital; buy 2¢ tail protection. Risk: neutral inventory still loses if built above $1/pair.
4. **Near-resolution capture** — buy the near-certain side at 98–99¢ for the last cent. High win rate but one bad fill erases dozens of wins; demands perfect resolution-source understanding. (SEER's crypto bot does a naive version — it decides the "winner" from Binance spot while markets settle on their own index: adverse selection.)
5. **Dynamic rotation** — re-aim exposure at whichever side currently has net edge; each rotation must clear spread+slippage costs or noise churns you to death.

## Signals worth implementing

- **Order-book imbalance**: (bid_vol − ask_vol) / total. Not predictive alone; combine with underlying move + trade flow + time remaining. SEER already fetches the sizes and discards them.
- **Cross-window dislocation**: z-score of the spread between related markets (5m vs 15m vs next-window) vs its history; high z = someone hasn't repriced. Requires independent fair value per window first (different opens/time-remaining can justify gaps).
- **Stale-quote sniping**: after an external move, resting orders reflect pre-move information for seconds. This is the core "repricing" edge; demands fast data and FAK/IOC execution.

## Execution rules

- Arb legs atomic: FOK/FAK, never sequential GTC (SEER's current defect D3).
- Inventory-adjusted reservation price: holding too much Up ⇒ lower your Up bid, raise your Down bid.
- Maker-vs-taker: saving 1¢ passively is wrong when the edge dies in seconds.
- Split size across levels; cancel stale orders aggressively.

## Risk rules

- Fractional Kelly (~25%) on validated edges only; hard caps per market, per underlying (BTC 5m + BTC 15m = ONE exposure), unhedged inventory limit, daily loss limit, data-quality kill switch.
- Backtests must model fees, spread, slippage, partial fills, latency, cancels — "profitable if every order fills at best displayed price" = not a strategy.

## Open-source landscape (for reference / possible adoption)

Frameworks: Freqtrade (crypto, FreqAI), vn.py, **NautilusTrader** (multi-asset incl. prediction markets, Rust core — best graduation target), Backtrader, QuantConnect LEAN, FinRL (RL research).
Prediction-market repos worth mining: CloddsBot (118 strategies incl. Binance-Poly latency), HarrierOnChain toolkit (Poly-Kalshi arb, whale tracking), polymarket_lp_tool (LP reward optimization), Composio poly-kalshi arb bot, TauricResearch/TradingAgents (multi-agent analysis), polymarket-mcp-server (Claude ↔ Polymarket live), weather bots (GFS ensemble / NWS + Kelly, forecast-error learning), SII-WANGZJ dataset (107GB, 1.1B trades) + prediction-market-backtesting simulator for validating any of the above.
