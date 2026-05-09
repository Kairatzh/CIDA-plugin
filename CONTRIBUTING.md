# Contributing

## Development setup

```bash
pip install -e .[dev,bpe,benchmarks]
pytest
```

## Guidelines

- Keep architecture changes explicit in `cida/config.py`
- Document ablations for any new CDP component
- Prefer small, isolated changes over sweeping rewrites
- If you change the deliberation protocol, update the README research notes

## Benchmarks

Benchmark scripts are research-oriented and may require local datasets or optional dependencies.
Please do not commit generated benchmark outputs by default.
