# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

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

