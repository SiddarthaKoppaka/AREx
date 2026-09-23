# Kaggle context-management boundary

Checked against the official ARC-AGI-3 competition pages on 2026-09-21.

## Competition requirements

- [Kaggle's code requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/code-requirements) require a Notebook submission, at most nine hours of CPU or GPU runtime, and disabled internet access. Freely and publicly available external data, including pretrained models, is allowed.
- The [ARC Prize competition page](https://arcprize.org/competitions/2026/arc-agi-3) also states that evaluation has no internet access and prize-eligible solutions must be open sourced. The host [confirmed](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/688481) that a local, offline open-weight Qwen model is acceptable.
- The [official action interface](https://docs.arcprize.org/actions) exposes the currently legal subset of `RESET` and `ACTION1`–`ACTION7` in `available_actions`. `ACTION6` requires coordinates; at `GAME_OVER`, only `RESET` is legal. The [game schema](https://docs.arcprize.org/game-schema) allows multiple frame arrays in one response and versioned game IDs.

## AREx implementation choices

These are project safeguards, not additional Kaggle rules.

- Run context management and artifact retrieval entirely inside the offline notebook process. Read model assets from attached `/kaggle/input` resources; write transient run artifacts under `/kaggle/working`. Never depend on a hosted model, network fetch, or another process outside the submission.
- Keep one isolated cognitive workspace, task graph, event trace, and artifact namespace per game session. Reuse loaded model weights, not game-derived memory. A game version change must not inherit unverified action meanings.
- Preserve every observation layer in the canonical event trace or a verified lossless artifact. RLE, deltas, previews, and purpose-specific retrieval are derived views; full evidence remains locally recoverable. No compacted prompt changes the official callback observation or action contract.
- Make phase changes and handoffs model-authored within the existing decision call. Deterministic compression and retrieval should not add a model generation per phase. Repairs still count as model calls, and the nine-hour runtime applies to the entire submission.
- Treat `active_context_target` as an efficiency target below the calculated emergency input ceiling. The ceiling must reserve generation and template margin from the actual model context window; refuse an oversized prompt rather than truncate it silently.

The [ARC Prize testing policy](https://arcprize.org/policy) describes model-selected notes and context compaction in its benchmark harnesses. Its separate Community Leaderboard policy does not override Kaggle's offline Notebook requirements.
