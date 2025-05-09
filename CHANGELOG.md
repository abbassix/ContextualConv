# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

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

