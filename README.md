# Agentic-Ticker

A single-file Python application for stock analysis using a local LLM agent loop.

## Overview

Agentic-Ticker demonstrates a manual agent loop for stocks that uses a local LLM (via Ollama) as a planner to orchestrate a sequence of analysis steps:

1. **load_prices** - Fetches OHLC data from yfinance
2. **compute_indicators** - Calculates MA5, MA20, daily returns, and annualized volatility
3. **detect_events** - Flags significant price movements (|Δ| ≥ threshold%)
4. **fetch_news** - Retrieves relevant news from Google News RSS
5. **build_report** - Generates a markdown report linking events to headlines

The application features a Streamlit UI that displays analysis results in real-time, including price charts, event tables, news tables, and a comprehensive report.

## Features

- **Local Execution**: Runs entirely offline after initial setup
- **No API Keys Required**: Uses free data sources (yfinance, Google News RSS)
- **Agent Loop**: Local LLM (Ollama) orchestrates the analysis pipeline
- **Interactive UI**: Streamlit dashboard with real-time results
- **Comprehensive Analysis**: Technical indicators, event detection, and news correlation

## Prerequisites

- Python 3.11+
- Ollama with llama3.1:8b model (optional for full LLM functionality)
- Required Python packages (see requirements.txt)

## Installation

1. Install Ollama from [https://ollama.com/](https://ollama.com/)
2. Pull the llama3.1:8b model:
   ```bash
   ollama pull llama3.1:8b
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
streamlit run agentic_ticker.py
```

Then open your browser to the provided URL (typically http://localhost:8501).

## Project Structure

```
├── agentic_ticker.py          # Main single-file application
├── requirements.txt           # Python dependencies
├── setup.cfg                  # Configuration for linting tools
├── src/                       # Source code (modular structure)
│   ├── models/                # Data models
│   ├── services/              # Business logic
│   └── ui/                    # User interface components
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── contract/              # Contract tests
└── specs/001-build-a-single/  # Feature specification and documentation
    ├── spec.md                # Feature specification
    ├── plan.md                # Implementation plan
    ├── tasks.md               # Detailed task breakdown
    ├── research.md            # Technical research
    ├── data-model.md          # Data model documentation
    ├── quickstart.md          # Quickstart guide
    ├── contracts/             # API contracts
    └── ...
```

## Development

This project follows a specification-driven development approach with:
- Clear feature specifications
- Detailed implementation plans
- Comprehensive task breakdowns
- Test-driven development (TDD)
- Modular code organization

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test suites:
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Contract tests
pytest tests/contract/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.