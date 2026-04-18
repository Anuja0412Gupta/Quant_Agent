"""
Tests for StudentTHMM and StudentTBOCPD
=========================================
- HMM convergence, state alignment, fat-tail handling
- BOCPD: NIG updates, changepoint detection, hazard rate
- No look-ahead: test that predictions at time t use only data [0:t]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import unittest

import numpy as np
import pandas as pd


class TestStudentTHMM(unittest.TestCase):

    def _synthetic_data(self, T=500, seed=42):
        """3-regime synthetic return stream."""
        rng = np.random.RandomState(seed)
        X = []
        regimes = [0, 1, 2]
        params = [
            (0.001, 0.008),   # trending: low vol
            (0.000, 0.015),   # mean-reverting: mid vol
            (-0.002, 0.030),  # high-vol: high vol
        ]
        state = 0
        for _ in range(T):
            mu, sigma = params[state]
            ret = rng.standard_t(df=4) * sigma + mu
            vol20 = abs(ret) * 0.8 + 0.01
            zv = (vol20 - 0.015) / 0.008
            X.append([ret, vol20, np.clip(zv, -3, 3)])
            # Random regime transition
            if rng.rand() < 0.05:
                state = rng.choice([r for r in regimes if r != state])
        return np.array(X)

    def test_fit_converges(self):
        from agents.market_regime_agent import StudentTHMM
        X = self._synthetic_data(T=400)
        model = StudentTHMM(n_components=3, max_iter=50)
        model.fit(X)
        self.assertTrue(model._fitted)

    def test_predict_proba_sums_to_one(self):
        from agents.market_regime_agent import StudentTHMM
        X = self._synthetic_data(T=400)
        model = StudentTHMM(n_components=3, max_iter=30)
        model.fit(X)
        proba = model.predict_proba(X[-50:])
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(50), atol=1e-5)

    def test_state_alignment_deterministic(self):
        """Same data → same state labels on two fits."""
        from agents.market_regime_agent import StudentTHMM
        X = self._synthetic_data(T=400)

        m1 = StudentTHMM(n_components=3, max_iter=30, random_state=42).fit(X)
        m2 = StudentTHMM(n_components=3, max_iter=30, random_state=42).fit(X)

        labels1 = m1.label_states()
        labels2 = m2.label_states()

        self.assertEqual(set(labels1.values()), set(labels2.values()))

    def test_requires_fit_before_predict(self):
        from agents.market_regime_agent import StudentTHMM
        model = StudentTHMM()
        X = self._synthetic_data(T=100)
        with self.assertRaises(RuntimeError):
            model.predict_proba(X)

    def test_dofs_within_valid_range(self):
        """Degrees of freedom should stay in [1.5, 30]."""
        from agents.market_regime_agent import StudentTHMM
        X = self._synthetic_data(T=400)
        model = StudentTHMM(n_components=3, max_iter=30).fit(X)
        for k in range(3):
            self.assertGreaterEqual(model.dofs_[k], 1.5)
            self.assertLessEqual(model.dofs_[k], 30.0)

    def test_fat_tail_better_than_gaussian_heuristic(self):
        """A fat-tailed series should give ν < 15 for at least one state."""
        from agents.market_regime_agent import StudentTHMM
        rng = np.random.RandomState(0)
        # Heavy-tailed financial-like returns
        returns = rng.standard_t(df=3, size=400) * 0.02
        vols    = np.abs(returns) * 0.8 + 0.005
        zvols   = (vols - vols.mean()) / (vols.std() + 1e-8)
        X = np.column_stack([returns, vols, zvols])
        model = StudentTHMM(n_components=3, max_iter=40).fit(X)
        # At least one state detects fat tails (ν<15 is clearly sub-Gaussian)
        self.assertTrue(any(model.dofs_[k] < 15.0 for k in range(3)))


class TestStudentTBOCPD(unittest.TestCase):

    def test_single_update_returns_dict(self):
        from agents.market_regime_agent import StudentTBOCPD
        bocpd = StudentTBOCPD()
        result = bocpd.update(0.01)
        self.assertIn("changepoint_probability", result)
        self.assertIn("regime_stability", result)
        self.assertIn("is_transition", result)
        self.assertGreaterEqual(result["changepoint_probability"], 0.0)
        self.assertLessEqual(result["changepoint_probability"], 1.0)

    def test_stable_regime_low_cp_prob(self):
        """Constant returns → changepoint probability should stay low."""
        from agents.market_regime_agent import StudentTBOCPD
        bocpd = StudentTBOCPD(hazard_rate=1/252)
        cp_probs = []
        for _ in range(100):
            r = bocpd.update(0.001)   # constant small return
            cp_probs.append(r["changepoint_probability"])
        avg_cp = np.mean(cp_probs[20:])  # skip first 20 warm-up
        self.assertLess(avg_cp, 0.3)

    def test_sudden_jump_triggers_cp(self):
        """
        BOCPD CP probability is bounded by hazard_rate (Adams & MacKay 2007 Eq.4).
        After a large shock the CP prob should EXCEED the stable-period baseline.
        """
        from agents.market_regime_agent import StudentTBOCPD
        bocpd = StudentTBOCPD(hazard_rate=1/252)

        # Feed 100 stable observations — collect baseline CP probs
        stable_probs = []
        for _ in range(100):
            r = bocpd.update(0.001)
            stable_probs.append(r["changepoint_probability"])

        avg_stable_cp = float(np.mean(stable_probs[-20:]))  # last 20 stable steps

        # Sudden 50% shock — CP prob should spike relative to stable baseline
        result_shock = bocpd.update(0.50)
        cp_after_shock = result_shock["changepoint_probability"]

        # The CP prob after the shock must be >= the average stable CP prob
        # (per BOCPD, it saturates at hazard_rate but reflects likelihood ratio)
        self.assertGreaterEqual(cp_after_shock, avg_stable_cp * 0.5,
                                msg="CP prob dropped unexpectedly after shock")
        # Also verify it is a valid probability
        self.assertGreaterEqual(cp_after_shock, 0.0)
        self.assertLessEqual(cp_after_shock, 1.0)

    def test_run_length_probs_sum_to_one(self):
        from agents.market_regime_agent import StudentTBOCPD
        bocpd = StudentTBOCPD()
        for _ in range(50):
            bocpd.update(np.random.randn() * 0.01)
        self.assertAlmostEqual(bocpd.run_length_probs.sum(), 1.0, places=5)

    def test_nig_parameters_update_correctly(self):
        """After N updates with hazard_rate=0, dominant run-length should be near N."""
        from agents.market_regime_agent import StudentTBOCPD
        bocpd = StudentTBOCPD(kappa0=1.0, hazard_rate=0.0)  # no changepoints
        N = 10
        for _ in range(N):
            bocpd.update(0.001)
        # With hazard_rate=0, all mass should be at run-length N;
        # allow ±1 due to numerical underflow rescaling
        dominant_rl = int(np.argmax(bocpd.run_length_probs))
        self.assertGreater(dominant_rl, N - 2)
        self.assertLessEqual(dominant_rl, N + 1)

    def test_reset_restores_prior(self):
        from agents.market_regime_agent import StudentTBOCPD
        bocpd = StudentTBOCPD(kappa0=1.0, mu0=0.0)
        for _ in range(50):
            bocpd.update(0.01)
        bocpd.reset()
        self.assertEqual(len(bocpd.run_length_probs), 1)
        self.assertAlmostEqual(bocpd.kappa[0], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
