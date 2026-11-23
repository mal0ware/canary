# Sources, formulas and comparison

Every formula the code uses is listed here with the place it was read. All sources were read on 2026-11-23. Where only a working paper or manuscript could be reached, that is stated, and its equation and page numbers are the working paper's.

## What was read

| Source | Copy read | Version |
|---|---|---|
| Kupiec (1995) | Federal Reserve FEDS working paper 95-24, from the St. Louis Fed FRASER archive | Working paper, May 1995. The Journal of Derivatives article is behind a paywall and was not read. |
| Christoffersen (1998) | Author manuscript, Internet Archive copy of the CiteSeerX file | Late draft, 28 pages, no equation numbers. The International Economic Review print was not read. |
| Bollerslev (1986) | Author's page at Duke | Published article, J. Econometrics 31, 307-327 |
| Corsi (2009) | Course mirror at WU Vienna | Published article, J. Financial Econometrics 7(2), 174-196 |
| Hansen and Lunde (2005) | Carolina Digital Repository, UNC | Text of the published article, without journal page numbers |
| Diebold and Mariano (1995) | Diebold's page at Penn | JBES 2002 reprint of the 1995 article, pages 134-144 |
| Patton (2011) | Author's page at Duke | Published article, J. Econometrics 160, 246-256 |
| Barone-Adesi, Giannopoulos and Vosper (1999) | filteredhistoricalsimulation.com | Working paper, "VaR Without Correlations for Nonlinear Portfolios", revised February 1998. The Journal of Futures Markets print was not read. |
| Bollerslev, Tauchen and Zhou (2009) | Bollerslev's page at Duke | Published article, RFS 22(11), 4463-4492 |
| Cboe VIX methodology | cdn.cboe.com | "Cboe Volatility Index Methodology" v6.0 and "Mathematics Methodology" |

Not reached: Hull and White (1998), an alternative statement of filtered historical simulation. Nothing here comes from it.

## Formulas

Notation: a breach series is 0/1 with 1 on a breach day. `p` is the stated tail probability (0.01 or 0.05), `n` the number of days and `x` the number of breaches.

### Kupiec's test (`canary.backtest.kupiec`)

Kupiec, expression (6), working paper p. 19:

    LR = -2 log[(1 - p)^(n - x) p^x] + 2 log[(1 - x/n)^(n - x) (x/n)^x]

Under the null it "has a Chi-Square distribution with one degree of freedom", with 5% critical value 3.841 (p. 11). The code takes 0 log 0 as 0, so zero breaches gives -2n log(1 - p).

### Christoffersen's tests (`canary.backtest.christoffersen`)

Christoffersen, Sections 3.1 to 3.3 of the manuscript (pp. 6-9), unnumbered. With `n_ij` the number of days in state `i` followed by a day in state `j`:

    L(Pi_1) = (1 - pi_01)^n_00 pi_01^n_01 (1 - pi_11)^n_10 pi_11^n_11
    pi_01 = n_01 / (n_00 + n_01),   pi_11 = n_11 / (n_10 + n_11)
    L(Pi_2) = (1 - pi_2)^(n_00 + n_10) pi_2^(n_01 + n_11),   pi_2 = (n_01 + n_11) / (n_00 + n_10 + n_01 + n_11)
    LR_ind = -2 log[L(Pi_2) / L(Pi_1)]  ~  chi-squared(1)
    LR_cc  = -2 log[L(p) / L(Pi_1)]     ~  chi-squared(2)
    LR_cc  = LR_uc + LR_ind

"As is standard, we condition on the first observation everywhere", so the likelihoods use the n - 1 transitions. The identity holds exactly when LR_uc is computed on those same transitions, which is what the `lr_uc` field of the result does. The headline Kupiec statistic uses all n days.

The paper codes the indicator as 1 when the outcome is inside the interval, with p the coverage. The code uses 1 for a breach and p for the tail probability. The statistics are the same under that relabelling.

### GARCH(1,1) (`canary.garch`)

Bollerslev, equations (1) and (2), p. 309, and (8), p. 311:

    eps_t | past ~ N(0, h_t)
    h_t = alpha_0 + alpha_1 eps_(t-1)^2 + beta_1 h_(t-1),   alpha_0 > 0, alpha_1 >= 0, beta_1 >= 0

Theorem 1 (p. 310): the process is wide-sense stationary with variance alpha_0 / (1 - alpha_1 - beta_1) if and only if alpha_1 + beta_1 < 1. Log likelihood, equation (18), p. 315, apart from a constant:

    l_t = -(1/2) log h_t - (1/2) eps_t^2 / h_t

Equation (8) is printed without the plus sign after alpha_0. The code follows equation (2). The code calls the parameters omega, alpha, beta, adds the constant -(1/2) log(2 pi), sets the first variance to the sample variance of the window, and caps alpha + beta at 0.9999.

### Filtered historical simulation (`canary.var.fhs_var_es`)

Barone-Adesi, Giannopoulos and Vosper, working paper equations (3) and (5), pp. 3-4:

    e_t = eps_t / sqrt(h_t)                 standardised residual
    z*_(t+1) = e* sqrt(h_(t+1))             a drawn residual scaled by the forecast volatility

with h_(t+1) = omega + alpha eps_t^2 + beta h_t (their footnote 3). For a one-day horizon no random draw is needed: the code takes the empirical quantile of the standardised residuals in the window and scales it by sqrt(h_(t+1)).

### The other three models (`canary.var`)

These are textbook definitions and are not taken from one of the sources above.

- Historical simulation: the VaR is minus the order statistic x_(k) of the window, k = ceil(p n). Expected shortfall is minus the mean of the k smallest returns.
- Normal: VaR = -(mu + sigma z_p), ES = -mu + sigma phi(z_p) / p, with the window's mean and standard deviation.
- GARCH, normal: the same with sigma replaced by the GARCH forecast.

### Realised variance and HAR (`canary.volforecast`)

Corsi, equation (3), p. 177, defines daily realised volatility as the square root of the sum of squared intraday returns. Equation (4) defines the weekly aggregate as the simple average of the last five daily values, including day t. The monthly aggregate is the 22-day analogue (stated in prose on p. 188). The HAR regression is equation (8), p. 181:

    RV_(t+1)^(d) = c + beta^(d) RV_t^(d) + beta^(w) RV_t^(w) + beta^(m) RV_t^(m) + omega_(t+1)

estimated by OLS.

### QLIKE (`canary.volforecast.qlike`)

Hansen and Lunde, Section 3, unnumbered: QLIKE = mean of log(h_t^2) + sigma_t^2 / h_t^2, with h_t^2 the variance forecast. Patton, equation (6), writes the same loss. Patton's Proposition 1 with Remark 2 (p. 251) shows it ranks forecasts the same way under any conditionally unbiased proxy. The code uses the b = -2 member of Patton's equation (24):

    L = proxy / forecast - log(proxy / forecast) - 1

which differs from the form above by a term that does not involve the forecast, so loss differences between two forecasts are identical. A test checks this.

### Diebold-Mariano (`canary.volforecast.diebold_mariano`)

Diebold and Mariano, Section 1.1, unnumbered (reprint p. 135). With d_t the loss differential:

    S_1 = dbar / sqrt(2 pi f_d(0) / T)  ~  N(0, 1)
    2 pi f_d(0) = sum over |tau| <= S(T) of gamma_d(tau)
    gamma_d(tau) = (1/T) sum (d_t - dbar)(d_(t - |tau|) - dbar)

with a rectangular window and S(T) = k - 1 for k-step forecasts. For one-step forecasts that is lag 0, which is the main statistic here.

### Variance risk premium (`canary.vrp`)

Bollerslev, Tauchen and Zhou, equation (22), p. 4475: VRP_t = IV_t - RV_t, where IV_t is the risk-neutral expected variance over [t, t+1] and RV_t the realised variance over [t-1, t]. They use the squared VIX for IV. Cboe: "VIX index = sigma x 100", with time to expiration in calendar years and a constant maturity of 30 days.

## Differences from the sources

1. Kupiec, Table 2 (maximum sample size at which x failures reject the null at 5%). Expression (6) with the 3.841 cut-off reproduces 14 of the 16 printed cells. For p* = 0.01, x = 4 the table prints 124 and the statistic gives 125 (LR = 3.867 at n = 125). For p* = 0.03, x = 4 the table prints 44 and the statistic gives 42 (LR = 3.68 at n = 44). I do not know the reason. Both are in his print, checked against the page image.
2. Kupiec, p. 22: a model with a true 2% rate shows three or fewer breaches in 250 days "26.23 percent of the time". The binomial gives 26.22%.
3. The non-rejection regions usually quoted from Kupiec for 255, 510 and 1,000 days are not in the working paper. They were read from Nieppola (2009), a master's thesis that reproduces them. The code reproduces all nine cells tested (p = 0.01, 0.025, 0.05). The first cell is printed as "N < 7", which hides that zero failures in 255 days also rejects (LR = 5.13).
4. Christoffersen's independence test in a long sample at 1%. With 25,338 days about 2.6 consecutive pairs are expected under independence, and a series with none has an LR above 3.84. In simulation the test rejected independent breaches 11% of the time at this sample size, against a nominal 5% (`results/power_independence.csv`, pi11 = 0.01). The paper does not discuss this case. `results/var_coverage.csv` therefore also gives a simulated p-value and a one-sided p-value on the number of pairs, each from 2,000 independent series at the observed rate. For the filtered model they are 0.042 and 0.016, against the chi-squared 0.024.
5. HAR is fitted to realised variance, not to Corsi's realised volatility, because the loss function scores a variance forecast. Corsi's own estimates use a two-scales estimator on tick data. This code uses the plain sum of squared five-minute returns.
6. HAR on a 24-hour market. Corsi's 5 and 22 are trading days in a week and a month. BTC trades every day, so 7 and 30 were also run. Both are reported.
7. Realised variance on days with missing bars is scaled up by 288 / (returns present) when at least 95% are present, and the day is dropped otherwise. This rule is mine. It rescales 6 days and drops 27. The HAR lags then run over the remaining days as if they were consecutive.
8. Hansen and Lunde scale realised variance by a constant so that its mean equals the variance of daily returns (their equation 1). No scaling is applied here because BTC has no overnight gap. Over the scored days the mean GARCH forecast was 1.03 times mean realised variance and 1.21 times the mean squared daily return, so realised variance from five-minute bars ran about 17% above squared daily returns. Results are given under both proxies.
9. Hansen and Lunde compare 330 models that use daily returns, with the SPA test. This repository compares two models with a Diebold-Mariano test, and one of them uses intraday data. It does not test their claim.
10. Variance risk premium. Bollerslev, Tauchen and Zhou report means of 33.23 (implied), 14.93 (realised) and 18.30 (premium) for 1990 to 2007, in what appear to be monthly percent-squared units. The same window here, divided by 12, gives 33.24, 20.24 and 13.00. The implied leg matches. The realised leg is higher because it is built from 21 daily close-to-close returns of the CRSP value-weighted market, where they sum 78 five-minute returns a day plus the overnight return of the S&P 500. The window is also 21 trading days where VIX covers 30 calendar days.
11. Filtered historical simulation in the source includes an ARMA mean. Here the mean is the window average.

## Claim by claim

| Claim | Source | Here |
|---|---|---|
| The proportion-of-failures test has little power in samples of a year | Kupiec, pp. 20-22 | Confirmed. Power against a true rate of 2% is 24% at 250 days and 78% at 1,000 (`power_kupiec.csv`). |
| The sample proportion needs 7,920 days to be within 0.005 of 0.01 with 95% confidence (Chebyshev) | Kupiec, p. 20 | Not recomputed. |
| A model can have the right rate and dependent breaches, and only a conditional test sees it | Christoffersen, Sections 2 and 3 | Confirmed on data. The normal model at the 5% limit has a 4.97% breach rate (Kupiec p = 0.84) and a 15.2% next-day rate (independence p = 1e-45). |
| Standardised GARCH residuals are suitable for historical simulation | Barone-Adesi et al., p. 3 | Supported for the rate (1.02%, 258 breaches against 253 expected), with some clustering left: 7 next-day pairs against 2.6 expected at 1%, 112 against 64 at 5%. |
| alpha + beta < 1 gives a finite variance | Bollerslev, Theorem 1 | Full-sample estimates: 0.989 for the US market, 0.970 for BTC (`garch_full_sample.csv`). |
| A three-term HAR forecasts realised volatility well out of sample | Corsi, Section 3 | HAR beat GARCH(1,1) on BTC realised variance (QLIKE 0.352 against 0.383, DM 4.00). Most of the gap is from 2020 to 2022. Corsi's own comparisons were not rerun. |
| Nothing beats GARCH(1,1) for an exchange rate, and it is beaten for IBM | Hansen and Lunde, Section 7 | Not tested. See difference 9. |
| Implied variance exceeds realised variance on average | Bollerslev, Tauchen and Zhou, Table 1 | Confirmed. Mean premium 110 (trailing) and 100 (forward) annualised points over 1990 to 2026, positive in 87% and 84% of months. |

## Choices and variants

The settings below were fixed before the full run and not changed after seeing results.

- Window 1,000 days, GARCH refit every 21 days, tail probabilities 1% and 5%.
- Four VaR models. The design named three. GARCH with normal shocks was added before the run, to separate the effect of the distribution from the effect of the volatility model.
- HAR lags (1, 5, 22), QLIKE loss, realised variance as the proxy, GARCH refitted daily.

Variants run, all reported in `results/`:

| Variant | Count | Where |
|---|---|---|
| VaR window: 500, 1,000, 2,000 days | 3 | `var_window_sensitivity.csv` |
| HAR lags: (1, 5, 22) and (1, 7, 30) | 2 | `vol_forecast.csv` |
| Proxy: realised variance and squared return | 2 | `vol_forecast.csv` |
| Diebold-Mariano variance: lag 0 rectangular, and Bartlett with lag 8 | 2 | `vol_forecast.csv` |
| Variance risk premium: trailing and forward realised leg | 2 | `vrp_summary.csv` |

The Bartlett lag is floor(4 (n/100)^(2/9)), a rule of thumb that was not read from a source. Nothing else was tried. Two things were added after the first run: the simulated p-values in difference 4, when the power table showed the size problem, and the yearly table of the volatility comparison, when the cumulative chart showed the gap was uneven. Neither changes a setting.

## Not verified

- The journal prints of Kupiec, Christoffersen and Barone-Adesi et al. Equation and page numbers above are from the working papers.
- That Bollerslev, Tauchen and Zhou's Table 1 is in monthly units. The paper says "annualized percentage form whenever appropriate". The monthly reading comes from their mean implied variance of 33.23, which is what VIX squared over 12 gives.
- Whether the realised leg here would match theirs with intraday S&P 500 data. No free source of it was found.
- The Bartlett-kernel variant and the order-statistic and normal expected-shortfall formulas come from general knowledge, with no source read. The normal formula is tested against numerical integration.
- CI on GitHub Actions has not been seen to run.

## VOLARE

The design allowed realised volatility from the VOLARE archive (volare.unime.it) "if its terms allow". The site has no terms-of-use or licence page. Downloads need an account, which is requested through a form and created by hand, and the site asks users to cite arXiv:2602.19732. Redistribution is not addressed. Because access needs an account, the build rules ruled it out, and realised variance is computed from the Binance public archive.
