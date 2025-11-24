# ArbiterLabs

> **Dropping Soon.**

ArbiterLabs is a quantitative research initiative dedicated to the exploration, documentation, and rigorous testing of algorithmic trading strategies.

## Core Philosophy

### 🔬 Isolated Environments
We treat every strategy as a distinct software entity. Each algorithm runs in its own **isolated environment** with its own specific set of dependencies. This ensures:
- **Reproducibility**: Results can be independently verified without dependency conflicts.
- **Purity**: Strategy logic is not contaminated by shared global state.
- **Modularity**: Strategies can be versioned and upgraded independently.

### 📊 Weekly Performance Logs
Transparency is our currency. **Every weekend**, we publish detailed outputs for our active strategies. These logs include:
- Performance metrics (Return, Drawdown, Sharpe Ratio).
- Analysis of behavior across different market regimes.
- Version-specific notes and adjustments.

---

### Project Setup

This project is built with **Vite + React**.

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```
