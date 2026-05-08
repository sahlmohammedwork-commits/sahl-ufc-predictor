"""Quick smoke test - imports server module, hits handler directly, no HTTP."""
import sys, json
sys.path.insert(0, ".")
from server import PredictorBundle, FightInput, _predict_one
import server as srv

srv.bundle = PredictorBundle()

# Test 1 - clear favorite + market odds
f = FightInput(
    fighter_a="Fighter_007",
    fighter_b="Fighter_042",
    weight_class="Lightweight",
    odds_a=-150, odds_b=130,
    bankroll=1000,
)
p = _predict_one(f)
print("=== Test 1: favorite vs underdog ===")
print(f"Pick: {p.pick}  Confidence: {p.confidence:.3f}")
print(f"P(A)={p.p_a:.3f}  Elo {p.elo_a:.0f} vs {p.elo_b:.0f}")
print(f"Market P(A)={p.market_p_a:.3f}  Edge_A={p.edge_a*100:+.1f}pts")
print(f"EV_A={p.ev_a*100:+.1f}%  Kelly stake A=${p.kelly_stake_a}")
print(f"Recommendation: {p.recommendation}")
print("Top 5 SHAP contributions:")
for e in p.explanations[:5]:
    print(f"  {e['feature']:25s} shap={e['shap']:+.3f}  val={e['value']:+.2f}")

# Test 2 - even matchup
print()
print("=== Test 2: even matchup ===")
f2 = FightInput(
    fighter_a="Fighter_001", fighter_b="Fighter_002",
    odds_a=-110, odds_b=-110, bankroll=1000,
)
p2 = _predict_one(f2)
print(f"Pick: {p2.pick}  P(A)={p2.p_a:.3f}")
print(f"Recommendation: {p2.recommendation}")

# Test 3 - card prediction
from server import CardInput, predict_card
print()
print("=== Test 3: full card ===")
card = CardInput(
    fights=[
        FightInput(fighter_a="Fighter_007", fighter_b="Fighter_042", odds_a=-150, odds_b=130),
        FightInput(fighter_a="Fighter_010", fighter_b="Fighter_055", odds_a=200, odds_b=-250),
        FightInput(fighter_a="Fighter_001", fighter_b="Fighter_002", odds_a=-110, odds_b=-110),
    ],
    bankroll=1000,
)
result = predict_card(card)
for item in result["card"]:
    pr = item["prediction"]
    print(f"  {pr['fighter_a']} vs {pr['fighter_b']}: pick={pr['pick']}  EV={item['best_ev']*100:+.1f}%")

print("\nAll tests passed!")
