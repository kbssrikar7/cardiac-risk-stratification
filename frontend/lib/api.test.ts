import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import { predict, fetchGradcam, ApiError, type ClinicalInput } from "./api.ts";

const input: ClinicalInput = { age: 60, lvef: 45, troponin: 0.02, ntprobnp: 500 };

afterEach(() => {
  delete (globalThis as { fetch?: unknown }).fetch;
});

// fetchGradcam's retry backoff uses the real setTimeout; stub it to resolve
// immediately so retry tests don't have to wait out multi-second delays.
function withImmediateTimers<T>(fn: () => Promise<T>): Promise<T> {
  const original = globalThis.setTimeout;
  (globalThis as { setTimeout: unknown }).setTimeout = ((cb: () => void) => {
    cb();
    return 0 as unknown as ReturnType<typeof setTimeout>;
  }) as typeof setTimeout;
  return fn().finally(() => {
    globalThis.setTimeout = original;
  });
}

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

test("fetchGradcam() retries on 503 and returns the result once the backend recovers", async () => {
  const body = { overlay_png_base64: "abc", slice_index: 5, num_slices: 12 };
  let call = 0;
  globalThis.fetch = (async () => {
    call += 1;
    if (call < 3) return new Response("", { status: 503, statusText: "Service Unavailable" });
    return new Response(JSON.stringify(body), { status: 200 });
  }) as typeof fetch;

  const file = new File([new Uint8Array(4)], "sample.nii.gz");
  const retries: Array<[number, number]> = [];
  const result = await withImmediateTimers(() =>
    fetchGradcam(file, (attempt, maxAttempts) => retries.push([attempt, maxAttempts])),
  );

  assert.deepEqual(result, body);
  assert.equal(call, 3);
  assert.deepEqual(retries, [[1, 3], [2, 3]]);
});

test("fetchGradcam() throws after exhausting retries on repeated 503s", async () => {
  globalThis.fetch = (async () => new Response("", { status: 503, statusText: "Service Unavailable" })) as typeof fetch;

  const file = new File([new Uint8Array(4)], "sample.nii.gz");
  await assert.rejects(
    () => withImmediateTimers(() => fetchGradcam(file)),
    (err: unknown) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.status, 503);
      return true;
    },
  );
});

test("fetchGradcam() does not retry on a non-503 error", async () => {
  let call = 0;
  globalThis.fetch = (async () => {
    call += 1;
    return new Response(JSON.stringify({ detail: "file too large" }), { status: 413, statusText: "Payload Too Large" });
  }) as typeof fetch;

  const file = new File([new Uint8Array(4)], "sample.nii.gz");
  await assert.rejects(
    () => fetchGradcam(file),
    (err: unknown) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.status, 413);
      assert.equal(err.message, "file too large");
      return true;
    },
  );
  assert.equal(call, 1);
});
