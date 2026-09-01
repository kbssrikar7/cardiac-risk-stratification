import { test } from "node:test";
import assert from "node:assert/strict";
import { displayRiskLabel, riskAccent, shapRiskMismatchNote } from "./risk.ts";

test("displayRiskLabel maps known ordinal codes from the stacked ensemble", () => {
  assert.equal(displayRiskLabel("0"), "Low Risk");
  assert.equal(displayRiskLabel("2"), "High Risk (Chronic Heart Failure)");
});

test("displayRiskLabel falls through unrecognized input (clinical-fallback path already returns readable labels)", () => {
  assert.equal(displayRiskLabel("Low Risk"), "Low Risk");
  assert.equal(displayRiskLabel("Something Unmapped"), "Something Unmapped");
});

test("riskAccent picks the red 'very high' branch, not the orange 'high' branch, for ordinal code 3", () => {
  const accent = riskAccent("3");
  assert.equal(accent.bar, "bg-red-500");
});

test("riskAccent picks the orange 'high' branch for plain high risk", () => {
  const accent = riskAccent("2");
  assert.equal(accent.bar, "bg-orange-500");
});

test("riskAccent picks amber for moderate and emerald for low", () => {
  assert.equal(riskAccent("1").bar, "bg-amber-500");
  assert.equal(riskAccent("0").bar, "bg-emerald-500");
});

test("shapRiskMismatchNote is null when the explainer and headline predictions agree", () => {
  assert.equal(shapRiskMismatchNote("3", "3"), null);
});

test("shapRiskMismatchNote is null when there is no headline prediction yet", () => {
  assert.equal(shapRiskMismatchNote("3", undefined), null);
});

test("shapRiskMismatchNote flags a disagreement with both readable labels, e.g. /predict's stacked ensemble vs /shap's best_xgb model can diverge for the same patient", () => {
  const note = shapRiskMismatchNote("0", "3");
  assert.match(note ?? "", /Low Risk/);
  assert.match(note ?? "", /Very High Risk \(Acute Cardiac Event\)/);
});
