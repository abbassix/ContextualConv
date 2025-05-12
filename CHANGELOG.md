# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.5.2] – 2025-05-12

### Fixed
- Added missing `Union` import in `contextual_conv.py`, which caused test collection to fail in v0.5.1.
- Fixed `pytest.raises` usage in tests by replacing unsupported `flags=` argument with `re.compile(...)` for case-insensitive matching.

---

## [0.5.1] – 2025-05-12

### Added
- `infer_context(x, return_raw_output=True)` now returns both the inferred context vector and the unmodulated convolutional output.
- README updated to document raw output support in `.infer_context()`.

### Changed
- Layer constructors no longer raise `ValueError` if both `use_scale` and `use_bias` are `False` — this is now allowed when no context is provided, enabling plain convolution mode.

### Fixed
- Constructor logic now correctly distinguishes between context-free and context-conditioned use cases.

---

## [0.5.0] – 2025-05-11

### Added
- `scale_mode` argument to control modulation behavior:
  - `"film"`: `out * (1 + γ) + β`
  - `"scale"`: `out * γ + β`
- Support for custom context summarization via optional `g` function.
- New test cases covering both modulation modes and custom context functions.
- One-hot encoded `context` vectors used in identity initialization tests.
- Full README overhaul with `scale_mode` usage examples and updated mode table.

### Changed
- Renamed internal `_apply_film` to `_apply_modulation` for generality.
- `_init_scale` now initializes weights and biases based on `scale_mode`:
  - In `"film"` mode: `γ ≈ 0` (via weight=0, bias=0)
  - In `"scale"` mode: `γ ≈ 1` (via weight=1, bias=0)
- Improved test stability by using controlled context inputs.

### Fixed
- Identity initialization in `"scale"` mode now correctly yields no change in output.
- Test failures due to variance in random context vectors are resolved using one-hot vectors.

---

## [0.4.0] – 2025-05-10

### Added
- New method: `infer_context(x)` to estimate the context vector from inputs.
- Optional `activation` argument to apply a nonlinearity after convolution but before FiLM modulation.
- `linear_bias` argument to allow disabling bias in `ContextProcessor` (required for reversibility and `infer_context`).
- Tests for `infer_context()` and `ContextProcessor`, including correctness and error modes.
- PyTorch profiler-compatible unit tests and coverage support in CI.
- `README.md` updated with coverage badge and improved usage instructions.

### Changed
- `_init_scale_to_one` now checks for `None` before zeroing `bias`, fixing a crash when `bias=False`.
- Identity initialization now works even with `linear_bias=False`.
- Refactored `_forward_impl` to optionally apply `activation`.
- Improved compatibility with torchscript and reproducibility-focused workflows.
- Tightened test coverage, including numerical checks and FiLM effectiveness.

### Fixed
- Fixed import error in tests due to missing `ContextProcessor` export in `__init__.py`.
- Avoided `AttributeError` when `bias` is disabled in the last Linear layer.
- Escaped invalid backslashes in docstrings to silence DeprecationWarnings in Python ≥3.12.

---

## [0.3.0] – 2025-04-22

### Added
- **FiLM-style conditioning**: support for per-channel **scale** (`gamma`) and **bias** (`beta`) from a global context vector.
- New flags `use_scale` and `use_bias` in `ContextualConv1d` and `ContextualConv2d`.
- Unified internal implementation for scale, bias, and combined FiLM-style application.
- Optional `h_dim` for MLP context processing now supports scale and bias jointly.
- Unit tests for 1D and 2D layers across all conditioning modes.
- Revised `README.md` with FiLM examples, usage table, and updated installation instructions.

### Changed
- Refactored `contextual_conv.py` with a shared base class and helpers.
- Improved docstrings and code readability.

### Fixed
- Escaped docstring backslashes to avoid DeprecationWarnings in Python 3.12+

---

## [0.2.0] – 2024-12-01

### Added
- Initial release of `ContextualConv1d` and `ContextualConv2d` with support for per-channel **bias** conditioning.
- `ContextProcessor` with optional MLP.
- Unit tests and minimal documentation.

---

