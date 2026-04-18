import sys
sys.path.insert(0, 'backend')
import numpy as np

# ── BOCPD Debug ──────────────────────────────────────────────────────────────
print("=== BOCPD Debug ===")
from agents.market_regime_agent import StudentTBOCPD
bocpd = StudentTBOCPD(hazard_rate=1/252)
for _ in range(100):
    bocpd.update(0.001)
r = bocpd.update(0.50)
print("CP prob after 50pct shock:", r["changepoint_probability"])
print("n_processed:", bocpd._n_processed)
print("run_length_probs[:5]:", bocpd.run_length_probs[:5])
print("max prob location:", int(np.argmax(bocpd.run_length_probs)))
print("len run_length_probs:", len(bocpd.run_length_probs))

# ── HMM dofs Debug ───────────────────────────────────────────────────────────
print()
print("=== HMM dofs_ Debug ===")
from agents.market_regime_agent import StudentTHMM
rng = np.random.RandomState(0)
returns = rng.standard_t(df=3, size=400) * 0.02
vols    = abs(returns) * 0.8 + 0.005
zvols   = (vols - vols.mean()) / (vols.std() + 1e-8)
X = np.column_stack([returns, vols, zvols])
model = StudentTHMM(n_components=3, max_iter=40).fit(X)
print("dofs_:", model.dofs_)
print("any < 15:", any(model.dofs_[k] < 15.0 for k in range(3)))
print("any < 20:", any(model.dofs_[k] < 20.0 for k in range(3)))

# ── Check what CP prob looks like for different shocks ───────────────────────
print()
print("=== BOCPD CP probs for varying shock sizes ===")
for shock in [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]:
    b2 = StudentTBOCPD(hazard_rate=1/252)
    for _ in range(50):
        b2.update(0.001)
    r2 = b2.update(shock)
    print("shock =", shock, "-> CP prob =", round(r2["changepoint_probability"], 6))
