# Parabolic SAR Strategy

Trend-following indicator providing dynamic entry/exit points and trailing stop levels.

## How It Works
- Plots dots above/below price that accelerate with trend momentum
- Long signal when price crosses above SAR dots
- Short signal when price crosses below SAR dots
- Acceleration Factor (AF) starts at 0.02, increases by 0.02 per new extreme
- Maximum AF capped at 0.20

## Key Parameters
- `acceleration_factor`: 0.02 (initial AF)
- `max_acceleration`: 0.20 (AF cap)
- `af_increment`: 0.02 (AF increase per extreme point)

## Usage Notes
- Always-in-market approach (long or short)
- Works best in trending markets with clear direction
- Generates false signals in ranging/choppy conditions
- SAR provides automatic trailing stop mechanism
- Lower AF for smoother signals, higher for aggressive trend capture
