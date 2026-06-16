"""
Unit tests for pricing_service.py inference functions.
Run: pytest test_pricing_service.py
"""
import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ── Minimal stub to import pricing_service without loading models ───────────

# Stub out anthropic before importing so no API key is needed
anthropic_stub = types.ModuleType("anthropic")
anthropic_stub.AsyncAnthropic = MagicMock
sys.modules.setdefault("anthropic", anthropic_stub)

# Stub cachetools if not installed in test env
try:
    from cachetools import LRUCache
except ImportError:
    cachetools_stub = types.ModuleType("cachetools")
    cachetools_stub.LRUCache = dict
    sys.modules["cachetools"] = cachetools_stub

import pricing_service as svc


# Seed state.meta with the values the functions need
svc.state.meta = {
    "deadline_map":             {"As soon as possible": 4, "Within 1-2 weeks": 2, "I'm flexible": 0, "": 0},
    "complexity_map":           {"low": 0, "medium": 1, "high": 2},
    "categories":               ["Cleaning", "Handyman", "HVAC", "Plumbing"],
    "production_categories":    ["Cleaning", "Electrical", "Exterior", "HVAC", "Handyman", "Landscaping", "Pest Control", "Plumbing"],
    "well_priced_categories":   ["Cleaning", "HVAC"],
    "training_median_interval": 200,
}


# ── fix_intervals ────────────────────────────────────────────────────────────

class TestFixIntervals(unittest.TestCase):
    def test_already_ordered(self):
        lo, mid, hi = svc.fix_intervals(100, 200, 300)
        self.assertEqual((lo, mid, hi), (100.0, 200.0, 300.0))

    def test_lo_above_mid_is_clamped(self):
        lo, mid, hi = svc.fix_intervals(250, 200, 300)
        self.assertLessEqual(lo, mid)

    def test_hi_below_mid_is_raised(self):
        lo, mid, hi = svc.fix_intervals(100, 200, 150)
        self.assertGreaterEqual(hi, mid)

    def test_equal_values(self):
        lo, mid, hi = svc.fix_intervals(200, 200, 200)
        self.assertEqual(lo, mid)
        self.assertEqual(mid, hi)


# ── _resolve_category ────────────────────────────────────────────────────────

class TestResolveCategory(unittest.TestCase):
    def test_prd_kebab_slug_resolves_to_title_case(self):
        self.assertEqual(svc._resolve_category("electrical"),        "Electrical")
        self.assertEqual(svc._resolve_category("pest-control"),      "Pest Control")
        self.assertEqual(svc._resolve_category("indoor-cleaning"),   "Cleaning")
        self.assertEqual(svc._resolve_category("exterior-cleaning"), "Exterior")
        self.assertEqual(svc._resolve_category("landscaping-lawn"),  "Landscaping")
        self.assertEqual(svc._resolve_category("hvac"),              "HVAC")
        self.assertEqual(svc._resolve_category("handyman"),          "Handyman")
        self.assertEqual(svc._resolve_category("plumbing"),          "Plumbing")

    def test_prd_absent_verticals_resolve_to_parent_category(self):
        self.assertEqual(svc._resolve_category("irrigation"),              "Landscaping")
        self.assertEqual(svc._resolve_category("tick-mosquito-treatment"), "Pest Control")

    def test_title_case_passes_through_unchanged(self):
        self.assertEqual(svc._resolve_category("Handyman"),    "Handyman")
        self.assertEqual(svc._resolve_category("Pest Control"),"Pest Control")
        self.assertEqual(svc._resolve_category("Cleaning"),    "Cleaning")
        self.assertEqual(svc._resolve_category("HVAC"),        "HVAC")

    def test_unknown_category_passes_through_unchanged(self):
        self.assertEqual(svc._resolve_category("Remodeling"),   "Remodeling")
        self.assertEqual(svc._resolve_category("SomethingNew"), "SomethingNew")

    def test_strips_whitespace(self):
        self.assertEqual(svc._resolve_category("  handyman  "), "Handyman")


# ── compute_confidence ───────────────────────────────────────────────────────

class TestComputeConfidence(unittest.TestCase):
    def test_mid_zero_returns_floor(self):
        result = svc.compute_confidence(0, 100, 0, "Handyman")
        self.assertEqual(result, 0.3)

    def test_narrow_interval_high_confidence(self):
        result = svc.compute_confidence(190, 210, 200, "Handyman")
        self.assertGreater(result, 0.8)

    def test_ood_cap_high_price(self):
        result = svc.compute_confidence(4000, 8000, 6000, "Handyman")
        self.assertLessEqual(result, 0.40)

    def test_ood_cap_wide_interval(self):
        # interval = 800, median_interval = 200 → 800 > 3*200 → cap 0.45
        result = svc.compute_confidence(100, 900, 500, "Handyman")
        self.assertLessEqual(result, 0.45)

    def test_ood_cap_non_production_category(self):
        result = svc.compute_confidence(100, 200, 150, "Remodeling")
        self.assertLessEqual(result, 0.40)

    def test_prd_kebab_slug_not_capped_as_ood(self):
        # PRD-format slugs must resolve to production categories, not get OOD-capped.
        for slug in ("electrical", "handyman", "plumbing", "pest-control",
                     "hvac", "indoor-cleaning", "irrigation", "tick-mosquito-treatment"):
            result = svc.compute_confidence(100, 200, 150, slug)
            self.assertGreater(result, 0.40,
                msg=f"'{slug}' should not be OOD-capped but got {result}")

    def test_all_three_ood_caps_combined(self):
        result = svc.compute_confidence(1000, 9000, 6000, "Remodeling")
        self.assertLessEqual(result, 0.40)

    def test_result_in_range(self):
        result = svc.compute_confidence(100, 200, 150, "Handyman")
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


# ── build_feature_vector ─────────────────────────────────────────────────────

class TestBuildFeatureVector(unittest.TestCase):
    def _make_req(self, **kwargs):
        defaults = dict(
            job_id="test", service_category="Handyman", zip_code="78704",
            job_description="fix faucet", original_estimate=200.0,
            original_estimate_lo=None, original_estimate_hi=None,
            deadline="Within 1-2 weeks", service_subtype=None,
            booking_month=None, job_status=None,
        )
        defaults.update(kwargs)
        return MagicMock(**defaults)

    def test_known_category_has_one_hot(self):
        req = self._make_req(service_category="Handyman")
        X = svc.build_feature_vector(req, {"labor_only": False, "task_count": 1,
                                             "complexity_tier": "medium", "has_area_measure": False})
        cats = svc.state.meta["categories"]
        handyman_idx = cats.index("Handyman")
        self.assertEqual(X[0, 7 + handyman_idx], 1.0)

    def test_ood_category_all_zero_category_vec(self):
        req = self._make_req(service_category="WindowCleaning")
        X = svc.build_feature_vector(req, {"labor_only": False, "task_count": 1,
                                             "complexity_tier": "medium", "has_area_measure": False})
        self.assertTrue(all(X[0, 7:] == 0.0))

    def test_task_count_capped_at_20(self):
        req = self._make_req()
        X = svc.build_feature_vector(req, {"labor_only": False, "task_count": 999,
                                             "complexity_tier": "medium", "has_area_measure": False})
        self.assertEqual(X[0, 2], 20.0)

    def test_labor_only_flag(self):
        req = self._make_req()
        X = svc.build_feature_vector(req, {"labor_only": True, "task_count": 1,
                                             "complexity_tier": "low", "has_area_measure": False})
        self.assertEqual(X[0, 0], 1.0)


# ── route_predict ─────────────────────────────────────────────────────────────

class TestRoutePredict(unittest.TestCase):
    def _make_req(self, category, original_estimate=None):
        return MagicMock(service_category=category, original_estimate=original_estimate,
                         original_estimate_lo=None, original_estimate_hi=None)

    def test_well_priced_with_estimate_uses_baseline(self):
        req = self._make_req("HVAC", original_estimate=350.0)
        lo, mid, hi = svc.route_predict(req, np.zeros((1, 10), dtype=np.float32))
        self.assertEqual(mid, 350.0)

    def test_well_priced_without_estimate_falls_to_model(self):
        req = self._make_req("HVAC", original_estimate=None)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([300.0])
        svc.state.models = {0.05: mock_model, 0.5: mock_model, 0.95: mock_model}
        lo, mid, hi = svc.route_predict(req, np.zeros((1, 10), dtype=np.float32))
        mock_model.predict.assert_called()

    def test_hard_category_uses_model(self):
        req = self._make_req("Handyman", original_estimate=500.0)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([400.0])
        svc.state.models = {0.05: mock_model, 0.5: mock_model, 0.95: mock_model}
        lo, mid, hi = svc.route_predict(req, np.zeros((1, 10), dtype=np.float32))
        mock_model.predict.assert_called()


# ── extract_scope fallback ────────────────────────────────────────────────────

class TestExtractScopeDefaults(unittest.IsolatedAsyncioTestCase):
    def _desc_key(self, description):
        import hashlib
        return hashlib.sha256(description.encode()).hexdigest()

    @patch("pricing_service.asyncio.sleep")
    async def test_defaults_returned_on_double_failure(self, mock_sleep):
        mock_sleep.return_value = None
        svc.state.cache = {}
        svc.state.client = MagicMock()
        svc.state.client.messages.create.side_effect = Exception("API down")
        result = await svc.extract_scope("any description")
        self.assertEqual(result, svc.EXTRACTION_DEFAULTS)

    @patch("pricing_service.asyncio.sleep")
    async def test_failure_not_cached(self, mock_sleep):
        mock_sleep.return_value = None
        svc.state.cache = {}
        svc.state.client = MagicMock()
        svc.state.client.messages.create.side_effect = Exception("API down")
        await svc.extract_scope("unique description xyz")
        self.assertNotIn(
            self._desc_key("unique description xyz"),
            svc.state.cache,
            "Defaults should not be written to cache on failure",
        )


# ── Model regression benchmarks ──────────────────────────────────────────────

class TestModelBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("eval_results.json") as f:
            cls.r = json.load(f)

    def test_routed_beats_baseline_mape(self):
        routed   = self.r["routed"]["MAPE"]
        baseline = self.r["baseline"]["MAPE"]
        self.assertLess(routed, baseline,
            f"Routed MAPE {routed:.2f}% regressed past baseline {baseline:.2f}%")

    def test_handyman_model_beats_baseline_mape(self):
        model    = self.r["handyman_model"]["MAPE"]
        baseline = self.r["handyman_baseline"]["MAPE"]
        self.assertLess(model, baseline,
            f"Handyman model MAPE {model:.2f}% regressed past baseline {baseline:.2f}%")

    def test_coverage_above_floor(self):
        coverage = self.r["routed"]["Coverage"]
        self.assertGreater(coverage, 90.0,
            f"Prediction interval coverage {coverage:.1f}% dropped below 90% floor")

    def test_blended_mape_below_hard_ceiling(self):
        mape = self.r["routed"]["MAPE"]
        self.assertLess(mape, 15.0,
            f"Blended MAPE {mape:.2f}% exceeds 15% hard ceiling — model may be broken")


if __name__ == "__main__":
    unittest.main()
