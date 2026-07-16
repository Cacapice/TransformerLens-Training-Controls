# Add TransformerBridge adapter for lightweight decoder-only pretraining models

<!--
When opening your PR, please make sure to only request a merge to `main` when you have found a bug in the currently released version of TransformerLens. All other PRs should go to `dev` in order to keep the docs in sync with the currently released version.
Please also make sure the branch you are attempting to merge from is not named `main`, or `dev`. Branches with these names from a different remote cause conflicting name issues when we periodically attempt to bring your PR up to date with the current stable TransformerLens source.
If your PR is primarily affecting docs, make sure has the string "docs" in its name. Building docs is disabled by default to avoid CI time, but the job has been configured to run whenever a branch with the word "docs" in it is being merged.
-->

# Description

Please include a summary of the change and which issue is fixed. Please also include relevant motivation and context. List any dependencies that are required for this change.

## Summary

Adds an interoperability adapter for live model modules loaded from
lightweight decoder-only pretraining checkpoints (RoPE, RMSNorm, SwiGLU,
optional MoE), enabling mechanistic interpretability experiments on
models trained from scratch.

Rather than converting checkpoints into a separate TransformerLens
implementation, the adapter wraps the existing module and delegates
execution to its native `forward`, preserving the source model's semantics
(including RoPE conventions) while exposing the standard TransformerBridge
interface.

## Motivation

Interpretability research on small from-scratch models (circuit discovery,
induction heads, toy superposition models, controlled probing) needs
precise control over architecture, data, and checkpoints. This PR adds
that interoperability path: train a small decoder-only transformer, wrap
it with `build_pretrain_bridge`, confirm the wrapping is transparent (the
bridge delegates to and matches the source model's own forward exactly,
rather than running a separately reimplemented TransformerLens forward),
then use TransformerLens's caching, hooks, and patching. The goal is
architecture interoperability, not a general-purpose training framework.

## What changed

- **`PretrainArchitectureAdapter`**
  (`transformer_lens/model_bridge/supported_architectures/pretrain.py`)
  maps token embeddings, decoder blocks (RoPE attention, RMSNorm, gated
  SwiGLU MLP), final norm, and tied/untied unembedding.
- **`DenseOrMoEFeedForwardBridge`** dispatches each block's MLP
  structurally (`router` + `experts` versus `gate` + `up` + `down`), so
  dense, MoE, and mixed models use the same component mapping.
- The selected delegate is excluded from normal module registration to
  prevent duplicate named hook points. Structural validation requires
  projection/router fields to be modules and `experts` to be a registered
  `ModuleList` or `ModuleDict`, producing clear construction-time errors
  for unsupported module shapes.
- **`build_pretrain_bridge(model, cfg)`**: the public entry point. Wraps
  the source model, normalizes its output contract, and ensures bridge
  mode changes propagate to the source model (see Scope boundaries).
  Direct `build_bridge_from_module` use still works for advanced cases.
- Registered `"SwiGLUMoEForCausalLM"` in `SUPPORTED_ARCHITECTURES` and
  `INTENTIONAL_EXCLUDES` (no HF Hub checkpoint backs it).
- Config mutation: like `nanogpt.py`, the adapter mutates the passed-in
  `cfg` in place rather than copying it.
- Unit tests split by concern: `test_pretrain_adapter.py` (mapping,
  dispatch, tied embeddings) and `test_pretrain_model_container.py`
  (container, kwarg filtering, output normalization, hooks, train/eval,
  dtype, persistence -- see Scope boundaries), sharing mocks from
  `_pretrain_mocks.py`. Integration tests use a
  real reference model (`tests/mocks/tiny_pretrain_model.py`) with genuine
  adjacent-pair RoPE/RMSNorm math for logit and hook-placement parity
  (see the integration test module docstring for the caveat on what
  logit parity does and doesn't prove).
- **`MoEBridge` defensive validation:** added validation for tuple
  outputs -- rejecting empty tuples and non-tensor first elements with a
  clear `TypeError` (instead of a lower-level `HookPoint` type-check
  error), and guarding `hook_router_scores` so it is only invoked for
  tensor-valued router scores. Covered directly by new `MoEBridge` tests
  (empty tuple, non-tensor first element, non-tensor metadata preserved
  without firing the router hook, tensor router scores still firing it),
  not just indirectly through the pretrain adapter's tests. This is a
  small compatibility and error-reporting improvement discovered while
  integrating the pretrain adapter and does not change the behavior of
  standard `(hidden_states, router_scores)` MoE outputs.
- **`NativeForwardAttentionBridge`**: a thin `AttentionBridge` subclass
  used for this adapter's opaque attention wrap. Plain `AttentionBridge`
  declares class-level `hook_aliases` (`hook_q`/`hook_k`/`hook_v`/
  `hook_z`) and `property_aliases` (`W_Q`/`W_K`/`W_V`/`W_O`) that assume
  mapped Q/K/V/O submodules; this adapter's wrap deliberately has none
  (see Hook coverage), so those aliases could never resolve and would
  either emit spurious "did not resolve" warnings or misrepresent the
  adapter's actual supported interface. The subclass clears both alias
  dicts and sets `supports_split_qkv_fork = False` so it doesn't
  advertise per-head hooks, weight aliases, or split-QKV-fork machinery
  it can't back.
- **Blocks use `DelegatedAttentionBlockBridge`, not plain `BlockBridge`**:
  reuses an existing abstraction rather than adding another special-case
  block, since it was written precisely for architectures where attention
  is delegated wholesale. It complements
  `NativeForwardAttentionBridge.supports_split_qkv_fork = False` (which
  prevents the split-QKV-fork machinery and its associated
  `hook_attn_in`/`hook_q_input`/`hook_k_input`/`hook_v_input` HookPoints
  from being exposed for this attention component) by also removing the
  block-level aliases that would otherwise dangle, pointing at hooks that
  aren't exposed. `hook_attn_out` is untouched by either change -- the
  attention component still fires its own `hook_out` normally, and the
  block-level alias to it still resolves.

## Supported behavior

`build_pretrain_bridge()` wraps the supplied model in place. Callers that
need a raw source-model checkpoint should capture its `state_dict()`
before bridge construction.

Requires a specific module protocol (documented in the adapter file's
docstring) rather than any decoder-only architecture generally. Within
that protocol:

- preserves the source model's rotary convention for native-forward
  execution, since the adapter delegates rather than reimplementing it —
  the tradeoff is that this adapter does not expose per-head attention
  hooks (`hook_q`/`hook_k`/`hook_v`/`hook_pattern`);
- RMS norm + gated SwiGLU MLP, dense/MoE/mixed, dispatched structurally;
- tied or untied embeddings; `forward` with or without a `**kwargs`
  catch-all (only `output_attentions` is stripped — anything else,
  including a caller typo, passes through and raises naturally);
- output as a `"logits"` dict, tensor, tensor-first tuple, or
  `.logits`-exposing object, all validated, with clear errors on
  malformed shapes.

## Container wrapping

`PretrainModelContainer` resolves attribute-name collisions with
`TransformerBridge` and normalizes output to the `.logits` contract
(raising immediately on missing/malformed output). Full mechanics are in
the class docstring. Kept local rather than generalizing
`TransformerBridge`'s output handling, to keep this PR architecture-scoped
— a candidate for later centralization, not addressed here.

The dense/MoE delegate is excluded from the public module tree to avoid
duplicate hook points; forward execution, gradients, dtype/device
conversion, `state_dict`-based reconstruction, hook lifecycle, and (via a
small generated subclass) train/eval mode all still reach the wrapped
source model — see Scope boundaries for why train/eval needed a
workaround here.

## Hook coverage

Block-level hooks only (`resid_pre`/`resid_mid`/`resid_post`, MLP
`hook_in`/`hook_out`) — no per-head `hook_q`/`hook_k`/`hook_v`/
`hook_pattern`. TransformerLens's HF-oriented attention bridges use
rotate-half RoPE, which does not match this adapter's adjacent-pair
convention and would break numerical equivalence, so it uses an opaque
`NativeForwardAttentionBridge` (a thin `AttentionBridge` subclass with no
Q/K/V/O submodules, and with `hook_aliases`/`property_aliases` cleared so
it doesn't advertise per-head hooks or weight aliases it can't back)
delegating to the source model's own attention instead. Per-head hooks
would need a new attention bridge built for that convention.

## Scope boundaries

No training loop, optimizers/schedules, dataset/tokenizer infrastructure,
experiment tracking, distributed training, or checkpoint-shard
reconstruction — those stay with the source framework. This PR only adds
the TransformerBridge mapping and its tests.

Worth flagging for discussion: `build_pretrain_bridge` reassigns
`bridge.__class__` to propagate train/eval mode (see code comments).
Reassigning `__class__` at runtime is unconventional; a
`TransformerBridge.train()` fix upstream would be cleaner if maintainers
are open to that scope.

Persistence is tested by saving the source model's pre-bridge
`state_dict`, reconstructing the source model and bridge from
configuration, and verifying output and mode-propagation equivalence.
Whole-object pickling is not asserted because it depends on unrelated
`TransformerBridge` internals, including locally defined attention
hook-conversion classes. A standardized `TransformerBridge` checkpoint
interface could provide a stable, versioned save/load contract,
improving usability and scalability without requiring whole-object
serialization.

## Testing

```bash
python -m pytest tests/unit/model_bridge/supported_architectures/test_pretrain_adapter.py -v
python -m pytest tests/unit/model_bridge/supported_architectures/test_pretrain_model_container.py -v
python -m pytest tests/integration/model_bridge/test_pretrain_adapter.py -v
python -m pytest tests/unit/model_bridge/generalized_components/test_moe_bridge_tuple_output.py -v
python -m pytest tests/unit/tools/test_model_registry.py -k TestRegistrySyncedWithFactory -v
```

Locally: 81 tests passed (30 adapter unit, 34 container unit, 9
integration, 4 generalized-component unit tests (MoEBridge tuple-output
validation), and 4 registry-sync tests). Covers: component
mapping and MLP dispatch (dense/MoE/mixed, checking actual
`run_with_cache` keys, not just `isinstance`, plus basic
type validation when an attribute name matches the MoE/gated-MLP protocol
but the value doesn't); logit and residual-stream parity; that
integration fixtures supply a `cfg` truthfully describing the model they
build (`build_pretrain_bridge` itself performs no cfg-vs-model
validation, so a mismatch silently reports wrong numbers on `bridge.cfg`
-- see `TestIntegrationFixturesSupplyATruthfulConfig`); activation
caching and hook interventions; kwarg filtering and output-normalization
edge cases; hook registration, gradients, dtype/device, `state_dict`-based
reconstruction; train/eval mode propagation (a cached generated subclass,
rather than per-instance method closures); malformed-architecture errors;
`MoEBridge`'s own tuple-output validation and router-score hook gating
(empty tuple, non-tensor first element, non-tensor metadata preserved
without firing the hook, tensor router scores still firing it);
`NativeForwardAttentionBridge`'s cleared `hook_aliases`/`property_aliases`/
`supports_split_qkv_fork`, and that `DelegatedAttentionBlockBridge`
correctly removes the now-unexposed split-QKV-fork block-level aliases
while leaving `hook_attn_out` (both as the block-level alias and the
attention component's own `hook_out`) resolvable.

Exact logit parity mainly demonstrates that wrapping does not alter the
delegated forward; it does not independently establish that the source
RoPE implementation is correct. The residual-stream decomposition and
hook-intervention tests provide the stronger evidence on hook placement.

Numerical parity is tested at float32 on CPU; float64 is tested for
preservation through construction only, not full parity at that dtype.

## Type of change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

<!--
Example:
| Before | After |
| ------ | ----- |
| _gif/png before_ | _gif/png after_ |
To upload images to a PR -- simply drag and drop an image while in edit mode and it should upload the image directly. You can then paste that source into the above before/after sections.
-->

# Checklist:

- [x] I have commented my code, particularly in hard-to-understand areas
- [x] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
- [x] I have not rewritten tests relating to key interfaces which would affect backward compatibility

<!--
As you go through the checklist above, you can mark something as done by putting an x character in it
For example,
- [x] I have done this task
- [ ] I have not done this task
-->
