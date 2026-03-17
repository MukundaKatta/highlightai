# HighlightAI - Sports Highlight Generator

HighlightAI automatically generates sports highlight reels by detecting exciting
moments in game footage using audio analysis, motion detection, and event
recognition.

## Features

- **Excitement Detection**: Scores frames by crowd noise levels, on-screen
  motion intensity, and celebration patterns.
- **Event Detection**: Identifies key sport-specific moments such as goals,
  touchdowns, dunks, and wickets.
- **Audio Analysis**: Detects crowd roar peaks and commentary excitement spikes.
- **Highlight Clipping**: Extracts the top N moments with configurable context
  windows.
- **Highlight Compilation**: Assembles clips into a cohesive reel with
  transitions.
- **Moment Ranking**: Scores and orders highlights by excitement and importance.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Generate highlights from a game recording
highlightai generate --input game.mp4 --top 10

# Simulate a game and generate a highlight reel report
highlightai simulate --sport soccer --duration 90

# List supported sports and event types
highlightai sports
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

```
src/highlightai/
  cli.py              - Click CLI entry-point
  models.py           - Pydantic data models
  simulator.py        - Synthetic game data generator
  report.py           - Rich console reporter
  detector/
    excitement.py     - ExcitementDetector (crowd/motion/celebration scoring)
    events.py         - EventDetector (goals/touchdowns/dunks/wickets)
    audio.py          - AudioAnalyzer (crowd roar / commentary peaks)
  editor/
    clipper.py        - HighlightClipper (extract top N moments)
    compiler.py       - HighlightCompiler (assemble clips + transitions)
    ranker.py         - MomentRanker (score & order highlights)
```

## License

MIT
