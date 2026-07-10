# LLM theming calls the Anthropic SDK directly, not Zerve's native GEN_AI blocks

**Status:** accepted

The notebook themes negative reviews with an LLM in a **batched loop** (batching forced by context window; verification depends on comparing theme overlap across same-size batches). Zerve's native Bedrock models (Gemma-3, Nemotron, GPT-OSS-Safeguard) are only reachable through the single-shot `GEN_AI` block — one prompt per block execution, not callable from a Python loop. Zerve's own guidance is that looping/parallel LLM work must use a direct provider SDK with your own key. We therefore call the **Anthropic SDK** (`claude-haiku-4-5`) directly inside `get_themes`, in a Python loop, using a bring-your-own key.

## Considered options

- **Native GEN_AI block (free, no key)** — rejected: single-shot only; cannot batch, so it forfeits the batching + cross-batch stability verification that is the differentiating showpiece. Kept documented as the honest fallback if the key path ever fails.
- **Zerve Fleet (spread/slicer/aggregator)** — still requires a direct SDK call per slice, so it needs a key anyway; adds parallelism we don't need at this sample size.
- **Anthropic SDK in a Python loop** — chosen. Satisfies every §4 requirement, is Zerve's recommended path, and (crucially) uses the *same* SDK offline and in Zerve, so `get_themes` is written and tested offline, pickled, and ported verbatim.

## Consequences

- `get_themes` is the only provider-specific code; everything downstream stays provider-agnostic (handoff §4).
- The themes pickle (keyed to `review_id`) is produced **offline**, matching the sentiment pickle. Zerve "run all" loads cache — no API dependency on a plain run.
- Requires an `sk-ant-...` key in a local `.env` (gitignored) and outbound network from the executing block. Cost is well under $1 at the planned sample size.
