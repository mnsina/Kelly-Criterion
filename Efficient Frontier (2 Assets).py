# -*- coding: utf-8 -*-
"""
MPT demo corregido:
 - Detecta si ingresaste RF en % (p. ej. 2) y lo convierte a decimal (0.02).
 - Usa SLSQP explícito en minimize para mayor robustez.
 - Calcula tangency portfolio por la fórmula analítica inv_cov*(mu-rf) (más estable).
 - Kelly con RF ahora incluye peso en RF para que suma total = 1.
 - Evita divisiones por cero en Sharpe y chequea frontier vacío.
 - Muestra Kelly unconstrained (risky-only) y versión con RF sum=1.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =========================
# Utility Functions
# =========================

def generate_random_data(n_assets, seed=None):
    if seed is not None:
        np.random.seed(seed)
    returns = np.random.uniform(0.05, 0.2, n_assets)
    # Random correlations between -0.3 and 0.3
    rand_pos = np.random.uniform(0.05, 0.3, (n_assets, n_assets))
    rand_neg = np.random.uniform(-0.3, -0.05, (n_assets, n_assets))
    mask = np.random.rand(n_assets, n_assets) < 0.5
    rand_pos[mask] = rand_neg[mask]
    corr = (rand_pos + rand_pos.T)/2
    np.fill_diagonal(corr, 1.0)
    # fix small/negative eigenvalues
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals[eigvals < 1e-6] = 1e-6
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    stds = np.random.uniform(0.05, 0.2, n_assets)
    cov = np.outer(stds, stds) * corr
    return returns, cov

def portfolio_perf(weights, returns, cov):
    var = float(max(np.dot(weights.T, np.dot(cov, weights)), 0.0))
    port_return = float(np.dot(weights, returns))
    port_vol = float(np.sqrt(var))
    return port_return, port_vol

# =========================
# Main Program
# =========================

def main():
    # -----------------
    # Step 0: Generate risky assets
    # -----------------
    n = int(input("How many risky assets? (ej. 4) "))
    returns, cov = generate_random_data(n)
    assets = [f"Asset{i+1}" for i in range(n)]
    print("\nExpected Returns:", np.round(returns,4))
    print("\nCovariance Matrix:\n", np.round(cov,6))

    # -----------------
    # Step 1: Risky assets only (weights sum to 1, shorting allowed)
    # -----------------
    n_points = 400
    frontier_risky = []

    target_grid = np.linspace(max(returns.min(), -0.5), returns.max()*2.5, n_points)
    for target_return in target_grid:
        constraints = (
            {'type':'eq','fun': lambda w: np.sum(w)-1},
            {'type':'eq','fun': lambda w, tr=target_return: np.dot(w, returns)-tr}
        )
        bounds = tuple((None,None) for _ in range(n))
        x0 = np.ones(n)/n
        res = minimize(lambda w: float(w @ cov @ w), x0,
                       method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter':2000, 'ftol':1e-12})
        if res.success:
            sigma = np.sqrt(max(res.fun, 0.0))
            frontier_risky.append((sigma, target_return))
    if len(frontier_risky) == 0:
        raise RuntimeError("No efficient frontier points found. Try different random seed or asset count.")
    frontier_risky = np.array(frontier_risky)

    # MVP risky-only
    res_mv = minimize(lambda w: float(w @ cov @ w),
                      np.ones(n)/n,
                      method='SLSQP',
                      bounds=tuple((None,None) for _ in range(n)),
                      constraints=({'type':'eq','fun': lambda w: np.sum(w)-1},),
                      options={'maxiter':2000})
    if not res_mv.success:
        print("Warning: MVP risky-only optimization failed; using equal weights fallback.")
        w_mv = np.ones(n)/n
    else:
        w_mv = res_mv.x
    r_mv, sd_mv = portfolio_perf(w_mv, returns, cov)
    print("\nMVP (risky only) weights:")
    for i,w in enumerate(w_mv): print(f"{assets[i]}: {w:.4f}")
    print(f"MVP return {r_mv:.4f}, sd {sd_mv:.4f}")

    # Kelly risky-only (normalized sum=1)
    inv_cov = np.linalg.inv(cov)
    w_kelly_uncon = inv_cov @ returns
    sum_k = np.sum(w_kelly_uncon)
    w_kelly = w_kelly_uncon / sum_k if abs(sum_k) > 1e-12 else np.ones(n)/n
    r_k, sd_k = portfolio_perf(w_kelly, returns, cov)
    print("\nKelly (risky only) normalized (sum=1) weights:")
    for i,w in enumerate(w_kelly): print(f"{assets[i]}: {w:.4f}")

    # -----------------
    # Step 2: Include Risk-Free Asset
    # -----------------
    rf_input = float(input("\nEnter risk-free rate (can be negative). If you type 2=2%, I will convert: "))
    rf = rf_input/100.0 if rf_input > 1.0 else rf_input
    print(f"Using rf = {rf:.4f}")

    # Tangency Portfolio (analytic)
    excess = returns - rf
    w_tan_unnorm = inv_cov @ excess
    sum_unnorm = np.sum(w_tan_unnorm)
    w_tan = w_tan_unnorm / sum_unnorm if abs(sum_unnorm) > 1e-12 else np.ones(n)/n
    r_tan, sd_tan = portfolio_perf(w_tan, returns, cov)
    print(f"\nTangency Portfolio (analytic) σ={sd_tan:.6f}, μ={r_tan:.6f}")
    for i,w in enumerate(w_tan):
        print(f"{assets[i]}: {w:.4f}")

    # Kelly with RF (normalized sum=1 incluyendo RF)
    w_kelly_risky = inv_cov @ (returns - rf)
    w_rf = 1 - np.sum(w_kelly_risky)  # peso en activo libre de riesgo
    r_kelly_rf = w_rf*rf + np.dot(w_kelly_risky, returns)
    sd_kelly_rf = np.sqrt(np.dot(w_kelly_risky, cov @ w_kelly_risky))
    print("\nKelly with RF (weights sum=1 including RF):")
    for i,w in enumerate(w_kelly_risky):
        print(f"{assets[i]}: {w:.4f}")
    print(f"Weight RF: {w_rf:.4f}")
    print(f"Return μ={r_kelly_rf:.6f}, σ={sd_kelly_rf:.6f}")

    # -----------------
    # Step 3: Plot
    # -----------------
    plt.figure(figsize=(10,6))
    for i in range(n):
        plt.scatter(np.sqrt(cov[i,i]), returns[i], color='black', marker='x')
        plt.text(np.sqrt(cov[i,i])*1.01, returns[i]*1.005, assets[i], fontsize=9)

    plt.plot(frontier_risky[:,0], frontier_risky[:,1], color='blue', label='Efficient Frontier (risky only)')
    plt.scatter(sd_mv, r_mv, color='green', label='MVP (risky only)')
    plt.scatter(sd_k, r_k, color='red', marker='^', label='Kelly (risky only, normalized)')

    # CML line
    cml_x = np.linspace(0, max(frontier_risky[:,0])*2.5, 400)
    if sd_tan > 1e-12:
        cml_y = rf + (r_tan - rf)/sd_tan * cml_x
        plt.plot(cml_x, cml_y, color='cyan', linestyle='--', label='CML (with RF)')

    # Tangency & Kelly RF
    plt.scatter(sd_tan, r_tan, color='magenta', marker='x', label='Tangency Portfolio')
    plt.scatter(sd_kelly_rf, r_kelly_rf, color='brown', marker='o', label='Kelly (RF sum=1)')

    plt.xlabel('Portfolio Std Dev (σ)')
    plt.ylabel('Portfolio Return (μ)')
    plt.title('MPT: Risky Assets + Risk-Free (weights sum=1 including RF)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

