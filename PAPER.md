# Sources, formulas and comparison

Every formula the code uses is listed here with the place it was read. The sources were read on 2026-11-23, except Acerbi and Szekely and Patton, Ziegel and Chen, which were read on 2026-11-24. Where only a working paper or manuscript could be reached, that is stated, and its equation and page numbers are the working paper's.

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
| Acerbi and Szekely (2014) | msci.com | MSCI research paper, "Backtesting Expected Shortfall", December 2014, 37 pages |
| Patton, Ziegel and Chen (2017) | arXiv:1707.05108v1 | Preprint dated 11 July 2017. The Journal of Econometrics print (2019) was not read. |
| Cboe VIX methodology | cdn.cboe.com | "Cboe Volatility Index Methodology" v6.0 and "Mathematics Methodology" |

Not reached: Hull and White (1998), an alternative statement of filtered historical simulation. Nothing here comes from it.

Not read: Glosten, Jagannathan and Runkle (1993) for the GJR model, Bollerslev (1987) for GARCH with Student t shocks, and Newey and West (1987). The code for those three follows the standard definitions, written from memory, and is checked against the `arch` package and by simulation.

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

### GJR-GARCH and Student t shocks (`canary.garch`)

No source was read for these. The variance equation with the GJR term is

    h_t = omega + (alpha + gamma 1[eps_(t-1) < 0]) eps_(t-1)^2 + beta h_(t-1)

with alpha >= 0 and alpha + gamma >= 0. The code treats alpha + beta + gamma / 2 as the persistence and caps it at 0.9999. The Student t shock has nu > 2 degrees of freedom and is scaled to unit variance, so the log likelihood of one day is

    lgamma((nu + 1) / 2) - lgamma(nu / 2) - (1/2) log(pi (nu - 2)) - (1/2) log h_t - ((nu + 1) / 2) log(1 + eps_t^2 / (h_t (nu - 2)))

nu is bounded between 2.05 and 200. The tests compare the estimates with `arch` (within 0.002 on simulated series of 4,000 days) and the t likelihood with scipy's density.

The GJR model is fitted by Gaussian quasi-maximum likelihood and used only as a filter: its standardised residuals feed filtered historical simulation.

### Filtered historical simulation (`canary.var.fhs_var_es`)

Barone-Adesi, Giannopoulos and Vosper, working paper equations (3) and (5), pp. 3-4:

    e_t = eps_t / sqrt(h_t)                 standardised residual
    z*_(t+1) = e* sqrt(h_(t+1))             a drawn residual scaled by the forecast volatility

with h_(t+1) = omega + alpha eps_t^2 + beta h_t (their footnote 3). For a one-day horizon no random draw is needed: the code takes the empirical quantile of the standardised residuals in the window and scales it by sqrt(h_(t+1)).

### The other four models (`canary.var`)

These are textbook definitions and are not taken from one of the sources above.

- Historical simulation: the VaR is minus the order statistic x_(k) of the window, k = ceil(p n). Expected shortfall is minus the mean of the k smallest returns.
- Normal: VaR = -(mu + sigma z_p), ES = -mu + sigma phi(z_p) / p, with the window's mean and standard deviation.
- GARCH, normal: the same with sigma replaced by the GARCH forecast.
- GARCH, Student t: with q the p-quantile of the ordinary t with nu degrees of freedom and f its density, and c = sqrt((nu - 2) / nu), VaR = -(mu + sigma c q) and ES = -mu + sigma c (f(q) / p) (nu + q^2) / (nu - 1). The tail-mean formula is tested against numerical integration.

### Expected-shortfall tests (`canary.backtest.acerbi_szekely`)

Acerbi and Szekely, equations (4) and (6), pp. 3-4. With X_t the return, VaR and ES positive, I_t = 1[X_t + VaR_t < 0] and N_T the number of breaches:

    Z1 = (sum of X_t I_t / ES_t) / N_T + 1
    Z2 = sum of X_t I_t / (T alpha ES_t) + 1

Both have expectation zero under the null that the forecast tail is the true tail, and are negative when the risk is understated. Their equation (7) links them: Z2 = 1 - (1 - Z1) N_T / (T alpha). Z1 "is in fact completely insensitive to an excessive number of exceptions". Z2 is not.

Significance is their procedure (12), p. 6: simulate M histories with X_t drawn independently from the forecast distribution P_t, holding the forecasts fixed, and compare. The code draws 1,000 histories per model inside the rolling loop: from the window for historical simulation, from the standardised residuals scaled by the forecast volatility for the filtered models, and from the normal or t for the others. The p-value is (1 + number of simulated Z at or below the observed Z) / (M + 1).

### FZ0 loss (`canary.backtest.fz0_loss`)

Patton, Ziegel and Chen, equation (6), p. 6 of the preprint, with v and e the VaR and ES as negative returns:

    L = -(1 / (alpha e)) 1[Y <= v] (v - Y) + v / e + log(-e) - 1

The code holds VaR and ES as positive losses, so it computes 1[y < -VaR](-VaR - y) / (alpha ES) + VaR / ES + log(ES) - 1. At the true VaR and ES the expected loss is log(ES), which a test checks. Mean losses are compared with a Diebold-Mariano test against the GARCH-filtered model, with a Bartlett kernel and lag floor(4 (n/100)^(2/9)).

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

The realised leg here is the sum of squared daily log returns dated within 30 calendar days, times 365 / 30, in percent squared. The t-statistic of the mean premium divides by a Newey-West standard error: the long-run variance gamma_0 + 2 sum of (1 - k / (L + 1)) gamma_k for k = 1..L, with L = floor(4 (n/100)^(2/9)), which is 5 for 439 months.

## Differences from the sources

1. Kupiec, Table 2 (maximum sample size at which x failures reject the null at 5%). Expression (6) with the 3.841 cut-off reproduces 14 of the 16 printed cells. For p* = 0.01, x = 4 the table prints 124 and the statistic gives 125 (LR = 3.867 at n = 125). For p* = 0.03, x = 4 the table prints 44 and the statistic gives 42 (LR = 3.68 at n = 44). I do not know the reason. Both are in his print, checked against the page image. The test does not take the code's word for the two cells. It computes expression (6) a second way, from scipy's binomial log-probabilities, finds the table from that, and requires the code to agree with it in all 16 cells.
2. Kupiec, p. 22: a model with a true 2% rate shows three or fewer breaches in 250 days "26.23 percent of the time". The binomial gives 26.22%.
3. The non-rejection regions usually quoted from Kupiec for 255, 510 and 1,000 days are not in the working paper. They were read from Nieppola (2009), a master's thesis that reproduces them. The code reproduces all nine cells tested (p = 0.01, 0.025, 0.05). The first cell is printed as "N < 7", which hides that zero failures in 255 days also rejects (LR = 5.13).
4. Christoffersen's independence test in a long sample at 1%. With 25,338 days about 2.6 consecutive pairs are expected under independence, and a series with none has an LR above 3.84. In simulation the test rejected independent breaches 11% of the time at this sample size, against a nominal 5% (`results/power_independence.csv`, pi11 = 0.01). The paper does not discuss this case. `results/var_coverage.csv` therefore also gives a simulated p-value and a one-sided p-value on the number of pairs, each from 10,000 independent series at the observed rate. For the GARCH-filtered model they are 0.045 and 0.019, against the chi-squared 0.024.
5. HAR is fitted to realised variance, not to Corsi's realised volatility, because the loss function scores a variance forecast. Corsi's own estimates use a two-scales estimator on tick data. This code uses the plain sum of squared five-minute returns.
6. HAR on a 24-hour market. Corsi's 5 and 22 are trading days in a week and a month. BTC trades every day, so 7 and 30 were also run. Both are reported.
7. Realised variance on days with missing bars is scaled up by 288 / (returns present) when at least 95% are present, and the day is dropped otherwise. This rule is mine. It rescales 6 days and leaves 27 without a realised variance. Those 27 stay in the series as missing values, one row per calendar day. Each HAR term is the mean of the days present in its lag window, a day with no value of its own has no regressor row, and a regression pair with a missing side is left out. So there is no forecast for the day after a gap. GARCH uses the close-to-close return, which exists for every day. The first version dropped the 27 days and ran the lags over the rest as if they were consecutive, which misdated every lag that spanned a gap. On 2018-02-08 the exchange has bars only until 00:25, so that day's return covers 25 minutes and the next day's the rest. Both days are in the estimation windows only.
8. Hansen and Lunde scale realised variance by a constant so that its mean equals the variance of daily returns (their equation 1). No scaling is applied here because BTC has no overnight gap. Over the scored days the mean GARCH forecast was 1.04 times mean realised variance and 1.22 times the mean squared daily return, so realised variance from five-minute bars ran about 18% above squared daily returns. Results are given under both proxies.
9. Hansen and Lunde compare 330 models that use daily returns, with the SPA test. This repository compares two models with a Diebold-Mariano test, and one of them uses intraday data. It does not test their claim.
10. Variance risk premium. Bollerslev, Tauchen and Zhou report means of 33.23 (implied), 14.93 (realised) and 18.30 (premium) for 1990 to 2007, in what appear to be monthly percent-squared units. February 1990 to December 2007 here, divided by 12, gives 33.14, 20.68 and 12.47. The implied leg is close. The realised leg is higher. It is built from daily close-to-close returns of the CRSP value-weighted market, where they sum 78 five-minute returns a day plus the overnight return of the S&P 500. Their realised variance is for the calendar month. Here it is for the 30 calendar days ending at the month end, so the two legs cover the same span as VIX. The first version used 21 trading days against VIX's 30 calendar days and an iid standard error. The premium has a first-order autocorrelation of 0.12 over the whole sample and 0.30 up to 2007, so the t-statistic is now Newey-West.
11. Filtered historical simulation in the source includes an ARMA mean. Here the mean is the window average.
12. Acerbi and Szekely's indicator is strict, X_t + VaR_t < 0, for continuous distributions. For others they give I' = 1[X + VaR < 0] + ((alpha - P[X + VaR < 0]) / P[X + VaR = 0]) 1[X + VaR = 0] (p. 4). A window of 1,000 returns puts probability 1/1,000 on its own 1% quantile, and the weight in that formula is 1 whenever alpha times the window length is a whole number, which holds for every window used. So a simulated draw at or below the limit counts as a breach. With the strict form the simulated breach rate would be 0.9% and the null mean of Z2 about +0.1. The first run of this test had that error, and it was corrected before any ES number was written up. A test now checks the simulated rate. Observed breaches keep the strict form, as everywhere else in the code.
13. Their p-value is the share of simulated statistics strictly below the observed one. The code counts ties and the observed history itself, (1 + count) / (M + 1), so the smallest possible value with 1,000 histories is 0.001.
14. Size of the ES tests for an estimated model. Their null is that returns come from the forecast distribution. A window model is never exactly that, even on independent data, because its shortfall is the mean of ten points. `results/es_test_size.csv` runs historical simulation on independent normal and Student t (5 degrees of freedom) draws and applies the same test. With 25,338 days Z1 rejects at the 5% level in 73 and 67 of 100 replications, with mean Z1 of -0.014 and -0.034 and a mean ratio of loss to forecast shortfall of 1.012 and 1.025. With 2,362 days it rejects in 9 and 5 of 100. Z2 rejects in none of the 400. The paper does not treat estimation error in the forecasts. No correction is made here. The table is the caveat.
15. Patton, Ziegel and Chen use the FZ0 loss to estimate models and to compare forecasts. Here it only compares. Their consistency condition needs ES beyond VaR and both strictly negative as returns, which holds for every forecast in this study.

## Claim by claim

| Claim | Source | Here |
|---|---|---|
| The proportion-of-failures test has little power in samples of a year | Kupiec, pp. 20-22 | Confirmed. Power against a true rate of 2% is 24% at 250 days and 78% at 1,000 (`power_kupiec.csv`). |
| The sample proportion needs 7,920 days to be within 0.005 of 0.01 with 95% confidence (Chebyshev) | Kupiec, p. 20 | Not recomputed. |
| A model can have the right rate and dependent breaches, and only a conditional test sees it | Christoffersen, Sections 2 and 3 | Confirmed on data. The normal model at the 5% limit has a 4.97% breach rate (Kupiec p = 0.84) and a 15.2% next-day rate (independence p = 1e-45). |
| Standardised GARCH residuals are suitable for historical simulation | Barone-Adesi et al., p. 3 | Supported for the rate with a 1,000-day window (1.02%, 258 breaches against 253 expected) and a 2,000-day one (1.00%), not with 500 days (1.15%, Kupiec p = 0.021). Some clustering is left: 7 next-day pairs against 2.6 expected at 1%, 112 against 64 at 5%. Losses on breach days were 5% above the forecast shortfall. On BTC the 1% rate was 0.72%, not rejected. |
| Z2 tests frequency and magnitude together, Z1 magnitude only | Acerbi and Szekely, pp. 3-4 | Seen on data. GARCH with Student t shocks passes Z1 (p = 0.13) with 369 breaches against 253 expected, and fails Z2 (p = 0.001). The GARCH-filtered model has the right count, passes Z2 (p = 0.12) and fails Z1 (p = 0.001). |
| The FZ0 loss ranks VaR and ES forecasts | Patton, Ziegel and Chen, Proposition 1 | Used as given. The GJR-filtered model has the lowest mean loss on the US market at 1% and 5% and on BTC at 1%. On BTC at 5% no model differs from the GARCH-filtered one at the 5% level except historical simulation and the normal model, which are worse. |
| alpha + beta < 1 gives a finite variance | Bollerslev, Theorem 1 | Full-sample estimates: 0.989 for the US market, 0.974 for BTC (`garch_full_sample.csv`). |
| A three-term HAR forecasts realised volatility well out of sample | Corsi, Section 3 | HAR beat GARCH(1,1) on BTC realised variance (QLIKE 0.352 against 0.383, DM 3.99). Most of the gap is from 2020 to 2022. Corsi's own comparisons were not rerun. |
| Nothing beats GARCH(1,1) for an exchange rate, and it is beaten for IBM | Hansen and Lunde, Section 7 | Not tested. See difference 9. |
| Implied variance exceeds realised variance on average | Bollerslev, Tauchen and Zhou, Table 1 | Confirmed. Mean premium 101 (trailing) and 114 (forward) annualised points over 1990 to 2026, positive in 86% and 85% of months, with Newey-West t-statistics of 4.5 and 4.3. |

## What was chosen, and when

The first version of the study was committed on 2026-11-23. Its settings are in the pipeline commit (9fca519), which comes before the first results commit (a3b8fd3):

- Window 1,000 days, GARCH refit every 21 days, tail probabilities 1% and 5%.
- Four VaR models: historical simulation, normal, GARCH with normal shocks, and filtered historical simulation. GARCH with normal shocks is there to separate the effect of the distribution from the effect of the volatility model.
- HAR lags (1, 5, 22), QLIKE loss, realised variance as the proxy, GARCH refitted daily.

Variants run in the first version, all reported in `results/`:

| Variant | Count | Where |
|---|---|---|
| VaR window: 500, 1,000, 2,000 days | 3 | `var_window_sensitivity.csv` |
| HAR lags: (1, 5, 22) and (1, 7, 30) | 2 | `vol_forecast.csv` |
| Proxy: realised variance and squared return | 2 | `vol_forecast.csv` |
| Diebold-Mariano variance: lag 0 rectangular, and Bartlett with lag 8 | 2 | `vol_forecast.csv` |
| Variance risk premium: trailing and forward realised leg | 2 | `vrp_summary.csv` |

Two things were added to the first version after its first run: the simulated p-values in difference 4, when the power table showed the size problem, and the yearly table of the volatility comparison, when the cumulative chart showed the gap was uneven.

### Post hoc: the second pass, 2026-11-24

Everything in this list was chosen after the first results had been read, in answer to a review of them. None of it was tuned on its own outcome: each item was run once as described, apart from the correction noted in the fourth item.

- Two more VaR models, GARCH with Student t shocks and filtered historical simulation with a GJR-GARCH filter. They were picked because the first version listed fat-tailed shocks and asymmetry as not tried. One specification of each was run.
- The Acerbi-Szekely tests and the FZ0 comparison, with the GARCH-filtered model as the reference because it was the first version's best model. 1,000 simulated histories.
- The size table for the ES tests (`es_test_size.csv`), added when Z1 rejected every empirical model: 100 replications, 200 histories each, normal and Student t (5) draws.
- The non-strict indicator in the ES null simulation (difference 12). The first run used the strict form and gave the GARCH-filtered model p-values of 0.021 (Z1) and 0.009 (Z2). The corrected values are 0.001 and 0.12. The correction made Z1 worse and Z2 better for that model, and it was made because the simulated breach rate was 0.9%, which a test now checks.
- Gap handling in the BTC series (difference 7). The BTC VaR backtest now has 2,362 days where it had 2,335.
- The 30-calendar-day realised leg and the Newey-West t-statistic in the variance risk premium (difference 10).
- 10,000 simulated series for the independence p-values, up from 2,000. With 2,000 the one-sided p-value for the GARCH-filtered model came out at 0.016 on one random stream and 0.029 on another.
- The independent reference for Kupiec's table in the tests.

The headline changed with the second pass. The first version said filtered historical simulation was the only one of four models breached about 1% of the time. The window table and the BTC table were in `results/` then and were not in that sentence. They are now.

The Bartlett lag is floor(4 (n/100)^(2/9)), a rule of thumb that was not read from a source. No p-value in this repository is corrected for the number of tests.

## Not verified

- The journal prints of Kupiec, Christoffersen, Barone-Adesi et al. and Patton, Ziegel and Chen. Equation and page numbers above are from the working papers and the preprint.
- The GJR, Student t GARCH and Newey-West formulas against their original papers. They agree with the `arch` package and with simulation.
- That Bollerslev, Tauchen and Zhou's Table 1 is in monthly units. The paper says "annualized percentage form whenever appropriate". The monthly reading comes from their mean implied variance of 33.23, which is what VIX squared over 12 gives.
- Whether the realised leg here would match theirs with intraday S&P 500 data. No free source of it was found.
- The Bartlett-kernel variant and the order-statistic and normal expected-shortfall formulas come from general knowledge, with no source read. The normal formula is tested against numerical integration.
- How much of the Z1 shortfall of the filtered models is estimation error. Difference 14 measures it for historical simulation on independent data only.
- Acerbi and Szekely's third test, Z3, was not implemented.

## VOLARE

The VOLARE archive (volare.unime.it) was considered as a source of realised volatility. The site has no terms-of-use or licence page. Downloads need an account, which is requested through a form and created by hand, and the site asks users to cite arXiv:2602.19732. Redistribution is not addressed. This repository uses only data that can be fetched without an account, so realised variance is computed from the Binance public archive.
