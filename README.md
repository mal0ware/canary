# Canary

**A risk model said losses this bad happen one day in a hundred. Count the days.**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

Every risk model produces a number. Which ones were right as often as they claimed, and which ones failed on consecutive days, when it matters most?

## What this will contain

- Value at risk and expected shortfall by historical simulation, a normal model, and filtered historical simulation with GARCH
- Kupiec's test of how often the limit was breached, and Christoffersen's test of whether breaches cluster
- GARCH(1,1) against HAR volatility forecasts, scored by QLIKE with a Diebold-Mariano test
- The variance risk premium from VIX, as a side study

## The result it is built to show

Which model's 1% limit was breached 1% of the time, and which one bunched its breaches into the same weeks.

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- Kupiec, "Techniques for Verifying the Accuracy of Risk Measurement Models", Journal of Derivatives, 1995
- Christoffersen, "Evaluating Interval Forecasts", International Economic Review 39, 1998
- Bollerslev, "Generalized autoregressive conditional heteroskedasticity", Journal of Econometrics 31(3), 1986
- Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility", Journal of Financial Econometrics 7(2), 2009
- Hansen and Lunde, "A forecast comparison of volatility models: does anything beat a GARCH(1,1)?", Journal of Applied Econometrics 20(7), 2005

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

Free daily prices and the Cboe VIX history, fetched by script. Realized volatility is computed from free intraday data where an archive's terms are unclear.

## How it is tested

The maths is tested before any result is trusted.

- A model that is correct by construction passes both coverage tests at the stated rate
- A model with clustered breaches fails Christoffersen's test and passes Kupiec's
- GARCH parameter estimates match a reference implementation

## Layout

```
src/canary/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [ ] Project skeleton, CI, and the reproduce command
- [ ] Value at risk and expected shortfall by historical simulation, a normal model, and filtered historical simulation with GARCH
- [ ] Kupiec's test of how often the limit was breached, and Christoffersen's test of whether breaches cluster
- [ ] GARCH(1,1) against HAR volatility forecasts, scored by QLIKE with a Diebold-Mariano test
- [ ] The variance risk premium from VIX, as a side study
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- A model can pass every backtest and fail in the next crisis. Coverage tests look backward.
- At the 1% level there are few breaches to count, so the tests have little power. The README shows the power, not only the verdict.
- This is research code. It is not investment advice and it does not trade.

## The name

The canary in the mine was a risk model. The only question that mattered was whether it warned in time.

## Licence

MIT. See `LICENSE`.
