# Task: Config Setup

## Context
Project structure is in place. Need central configuration system.

## Goal
Create `config/config.yaml` and `src/utils.py` with config loading.

## Acceptance Criteria
- [ ] `config/config.yaml` exists with all training params
- [ ] `src/utils.py` has `load_config()`, `set_seed()`, `compute_metrics()`
- [ ] Config includes: data, model, training, paths sections
- [ ] Tests pass: `pytest tests/unit/test_utils.py -v`

## Files
- Create: `config/config.yaml`, `src/utils.py`
- Create: `tests/unit/test_utils.py`
