# Contributing

## Development setup

```bash
pip install -e .[dev]
pytest
```

## Guidelines

- Keep architecture changes explicit in `cida_plugin/config.py`
- Document changes for any new component
- Prefer small, isolated changes over sweeping rewrites
- If you change the deliberation protocol, update the README
