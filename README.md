# AI Cost Modeler

Model and compare AI inference costs before you build — and validate estimates against real run data.

## Installation

```bash
uv sync
```

## Usage

```bash
uv run cost --help
```

## Configuration

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
```

## Development

```bash
uv run cost estimate
uv run cost compare
uv run cost report
uv run cost cherny-check
```
