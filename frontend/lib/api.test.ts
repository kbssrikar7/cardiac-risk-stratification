import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import { predict, ApiError, type ClinicalInput } from "./api.ts";

const input: ClinicalInput = { age: 60, lvef: 45, troponin: 0.02, ntprobnp: 500 };

afterEach(() => {
  delete (globalThis as { fetch?: unknown }).fetch;
});

test("predict() surfaces the backend's JSON {detail} message as ApiError.message and preserves status", async () => {
  globalThis.fetch = (async () =>
    new Response(JSON.stringify({ detail: "age must be between 0 and 120" }), {
      status: 422,
      statusText: "Unprocessable Entity",
    })) as typeof fetch;

  await assert.rejects(
    () => predict(input),
    (err: unknown) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.status, 422);
      assert.equal(err.message, "age must be between 0 and 120");
      return true;
    },
  );
});

test("predict() falls back to statusText when the error body isn't JSON", async () => {
  globalThis.fetch = (async () =>
    new Response("<html>Bad Gateway</html>", {
      status: 502,
      statusText: "Bad Gateway",
    })) as typeof fetch;

  await assert.rejects(
    () => predict(input),
    (err: unknown) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.status, 502);
      assert.equal(err.message, "Bad Gateway");
      return true;
    },
  );
});

test("predict() returns the parsed JSON body on a successful response", async () => {
  const body = {
    risk_class: "1",
    probabilities: { "0": 0.1, "1": 0.7, "2": 0.15, "3": 0.05 },
    model_used: "stacked_ensemble",
    rule_based_risk: "Moderate Risk",
    rule_based_reasoning: "...",
  };
  globalThis.fetch = (async () => new Response(JSON.stringify(body), { status: 200 })) as typeof fetch;

  const result = await predict(input);
  assert.deepEqual(result, body);
});
