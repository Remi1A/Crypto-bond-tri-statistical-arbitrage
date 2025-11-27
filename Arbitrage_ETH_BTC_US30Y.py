# ===============================================================
# Tri-Arbitrage rotatif — Tests des 3 stratégies & hedges
# Cible ∈ {ETH, BTC, US30Y}, hedge = les 2 autres actifs
# - Niveaux: target ~ 1 + feat1 + feat2  (résidu = spread S_t)
# - Signal: z-score statique de S_t 
# - Rendements: r_target ~ 1 + r_feat1 + r_feat2  (hedge gammas)
# - Backtest: sweep k (seuil), coûts inclus
# - Graphs: z-scores (3), Sharpe vs seuil (3-en-1), Equities (3-en-1), plan 3D du meilleur
# - NEW: impression des coefficients (β & γ) pour la stratégie BTC + explication simple
# ===============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import spearmanr, kendalltau, linregress

class TriArbRotator:
    def __init__(
        self,
        csv_file="btc_us30y_daily.csv",     # colonnes: Date,BTC,ETH,US30Y
        split_date="2024-07-23",            # mardi 23/07/2024 (ETF ETH)
        annualization=252,
        duration_30y=19.0,                  # r_30Y ≈ -D * Δy (D ≈ 18-20)
        cost_btc=0.00001, cost_eth=0.00002, cost_bond=0.0005,
        thresholds=np.round(np.arange(0.50, 3.01, 0.01), 2),
        verbose=True
    ):
        self.csv_file = csv_file
        self.split_date = pd.Timestamp(split_date)
        self.A = annualization
        self.D = duration_30y
        self.costs = {"BTC": cost_btc, "ETH": cost_eth, "US30Y": cost_bond}
        self.THRESHOLDS = thresholds
        self.verbose = verbose
        self.targets = ["ETH","BTC","US30Y"]

    # --------------- IO / Préparation ---------------
    def load(self):
        df = pd.read_csv(self.csv_file)
        if not set(["Date","BTC","ETH","US30Y"]).issubset(df.columns):
            raise ValueError("Le CSV doit contenir: Date, BTC, ETH, US30Y")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        df = df.asfreq("D").ffill()
        self.levels = df[["BTC","ETH","US30Y"]].copy()
        self.after = self.levels.loc[self.levels.index >= self.split_date].copy()

    # --------------- Aides ---------------
    def _bond_return(self, y):
        y_dec = y / 100.0 if y.abs().median() > 1 else y
        dy = y_dec.diff().fillna(0.0)
        return (-self.D) * dy

    def _returns(self, levels):
        r_btc = levels["BTC"].pct_change().fillna(0.0)
        r_eth = levels["ETH"].pct_change().fillna(0.0)
        r_bond = self._bond_return(levels["US30Y"])
        return pd.DataFrame({"BTC": r_btc, "ETH": r_eth, "US30Y": r_bond})

    @staticmethod
    def _z_static(s):
        mu, sd = s.mean(), s.std(ddof=0)
        return (s - mu) / (sd if sd > 0 else np.nan)

    # --------------- Régressions ---------------
    def _fit_plane_levels(self, target, feats, levels):
        y = levels[target].values
        X = np.column_stack([np.ones(len(levels))] + [levels[c].values for c in feats])
        beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha = float(beta_hat[0])
        betas = {f: float(b) for f, b in zip(feats, beta_hat[1:])}
        fitted = X @ beta_hat
        resid = pd.Series(y - fitted, index=levels.index, name=f"S_{target}")
        return alpha, betas, resid

    def _fit_hedge_returns(self, target, feats, rets):
        y = rets[target].values
        X = np.column_stack([np.ones(len(rets))] + [rets[c].values for c in feats])
        gamma_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha = float(gamma_hat[0])
        gammas = {f: float(g) for f, g in zip(feats, gamma_hat[1:])}
        return alpha, gammas

    # --------------- Backtest ---------------
    def _backtest_once(self, z, rets, target, feats, gammas, k):
        idx = z.index
        r = rets.reindex(idx).fillna(0.0)

        pos = {a: np.zeros(len(idx)) for a in [target] + feats}
        state = 0
        n_trades = 0
        z_vals = z.values

        for i in range(1, len(idx)):
            if state == 0:
                if z_vals[i-1] >= k:    state = -1; n_trades += 1
                elif z_vals[i-1] <= -k: state = +1; n_trades += 1
            else:
                if state == +1 and z_vals[i-1] >= 0:  state = 0
                elif state == -1 and z_vals[i-1] <= 0: state = 0

            if state == +1:
                pos[target][i] = +1
                for f in feats: pos[f][i] = -gammas[f]
            elif state == -1:
                pos[target][i] = -1
                for f in feats: pos[f][i] = +gammas[f]

        strat_r, tc = np.zeros(len(idx)), np.zeros(len(idx))
        for asset in pos.keys():
            strat_r += pos[asset] * r[asset].values
            tc += self.costs[asset] * np.abs(np.diff(pos[asset], prepend=0))

        ser = pd.Series(strat_r - tc, index=idx, name=f"r_{target}")
        eq = (1 + ser).cumprod()
        n = len(eq)
        ann_ret = eq.iloc[-1]**(self.A/n) - 1 if n > 0 else np.nan
        ann_vol = ser.std() * np.sqrt(self.A)
        sharpe  = ann_ret/ann_vol if ann_vol > 0 else np.nan
        mdd = (eq/eq.cummax() - 1).min()
        return {"equity": eq, "series": ser, "sharpe": sharpe, "ann_ret": ann_ret,
                "ann_vol": ann_vol, "mdd": mdd, "total_ret": eq.iloc[-1]-1, "n_trades": n_trades}

    def _sweep_target(self, target, levels, rets):
        feats = [c for c in ["BTC","ETH","US30Y"] if c != target]

        alpha_p, betas_p, resid = self._fit_plane_levels(target, feats, levels)
        z = self._z_static(resid).dropna()

        alpha_r, gammas = self._fit_hedge_returns(target, feats, rets)

        rows, eqs = [], {}
        for k in self.THRESHOLDS:
            r = self._backtest_once(z, rets, target, feats, gammas, k)
            rows.append({"target": target, "threshold": k, "sharpe": r["sharpe"],
                         "ann_ret": r["ann_ret"], "ann_vol": r["ann_vol"],
                         "mdd": r["mdd"], "total_ret": r["total_ret"],
                         "n_trades": r["n_trades"]})
            eqs[k] = r["equity"]

        df = pd.DataFrame(rows).sort_values(["target","sharpe"], ascending=[True, False]).reset_index(drop=True)
        best = df.groupby("target").head(1).reset_index(drop=True).iloc[0]
        return {
            "alpha_price": alpha_p, "betas_price": betas_p,
            "alpha_ret": alpha_r, "gammas": gammas,
            "z": z, "df": df, "eqs": eqs, "best": best
        }

    # --------------- Graphiques ---------------
    def plot_zscores_after(self):
        z = (self.after - self.after.mean()) / self.after.std(ddof=0)
        plt.figure(figsize=(11,5))
        for c in ["BTC","ETH","US30Y"]:
            plt.plot(z.index, z[c], label=f"{c} (z)")
        for h in [0,1,2,-1,-2]: plt.axhline(h, ls="--", alpha=0.25)
        plt.title(f"Z-scores (post {self.split_date.date()}) — BTC, ETH, US30Y")
        plt.xlabel("Date"); plt.ylabel("z-score")
        plt.legend(frameon=False); plt.grid(True, alpha=0.25); plt.tight_layout(); plt.show()

    def plot_sharpe_vs_threshold_all(self, results):
        plt.figure(figsize=(12,6))
        for tgt in self.targets:
            df_t = results[tgt]["df"].sort_values("threshold")
            x, y = df_t["threshold"].values, df_t["sharpe"].values
            plt.plot(x, y, label=tgt)
            j = int(np.nanargmax(y))
            plt.scatter([x[j]],[y[j]], marker="x", s=140, linewidths=3,
                        label=f"{tgt}* k={x[j]:.2f}, Sh={y[j]:.2f}")
        plt.title("Sharpe vs seuil (σ) — 3 stratégies (cibles)")
        plt.xlabel("Seuil k (σ)"); plt.ylabel("Sharpe")
        ax=plt.gca(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6)); ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        plt.grid(True, alpha=0.25); plt.legend(frameon=False, loc="best"); plt.tight_layout(); plt.show()

    def plot_equities_best_all(self, results):
        plt.figure(figsize=(12,6))
        for tgt in self.targets:
            k_star = float(results[tgt]["best"]["threshold"])
            sh = float(results[tgt]["best"]["sharpe"])
            eq = results[tgt]["eqs"][k_star]
            plt.plot(eq.index, eq.values, label=f"{tgt} (k={k_star:.2f}, Sh={sh:.2f})")
        plt.title("Equity curves — meilleur seuil par stratégie (coûts inclus)")
        plt.xlabel("Date"); plt.ylabel("Equity (base=1)")
        plt.legend(frameon=False); plt.grid(True, alpha=0.25); plt.tight_layout(); plt.show()

    def plot_best_plane_3d(
        self, results,
        highlight_top=25, grid_n=50,
        elev=22, azim=130,
        zoom_q=(0.10, 0.90),
        size_min=24, size_max=180,
        plane_alpha=0.20
    ):
        """
        3D scatter + plan OLS (meilleure stratégie) en échelles standardisées (z-scores)
        pour éviter l'écrasement. Couleur/taille ~ |résidu| (en unités *réelles*),
        segments verticaux sur les plus gros écarts, zoom par quantiles.
        """
        # -- meilleure cible au Sharpe
        scores = {t: float(results[t]["best"]["sharpe"]) for t in self.targets}
        best_target = max(scores, key=scores.get)
        feats = [c for c in self.targets if c != best_target]
        a = results[best_target]["alpha_price"]
        b1 = results[best_target]["betas_price"][feats[0]]
        b2 = results[best_target]["betas_price"][feats[1]]
    
        # -- données brutes
        df = self.after.copy()
        X = df[feats[0]].values
        Y = df[feats[1]].values
        Z = df[best_target].values
    
        # -- standardisation pour l'AFFICHAGE UNIQUEMENT
        def zsc(v):
            mu, sd = float(np.mean(v)), float(np.std(v, ddof=0)) or 1.0
            return (v - mu) / sd, mu, sd
        Xs, mx, sx = zsc(X)
        Ys, my, sy = zsc(Y)
        Zs, mz, sz = zsc(Z)
    
        # -- plan sur une grille (converti correctement entre espaces)
        xg_s = np.linspace(Xs.min(), Xs.max(), grid_n)
        yg_s = np.linspace(Ys.min(), Ys.max(), grid_n)
        Xg_s, Yg_s = np.meshgrid(xg_s, yg_s)
        # grille en "brut"
        Xg = Xg_s * sx + mx
        Yg = Yg_s * sy + my
        Zg = a + b1 * Xg + b2 * Yg               # plan en brut
        Zg_s = (Zg - mz) / (sz if sz != 0 else 1)  # converti en z-score pour l'affichage
    
        # -- résidus (en BRUT, pour la couleur/taille & segments)
        Zhat = a + b1 * X + b2 * Y
        resid = Z - Zhat
        abs_resid = np.abs(resid)
        rmax = np.nanmax(abs_resid) if np.isfinite(abs_resid).any() else 1.0
        sizes = size_min + (size_max - size_min) * (abs_resid / (rmax if rmax > 0 else 1.0))
    
        # -- figure
        fig = plt.figure(figsize=(9.6, 7.4), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
    
        # plan (transparent)
        ax.plot_surface(Xg_s, Yg_s, Zg_s, alpha=plane_alpha, rstride=1, cstride=1,
                        linewidth=0, edgecolor="none", antialiased=True)
    
        # nuage: couleurs = |résidu| (brut), coordonnées = z-scores
        sc = ax.scatter(Xs, Ys, Zs, c=abs_resid, s=sizes, cmap="viridis", depthshade=True)
    
        # segments verticaux pour outliers
        top = min(highlight_top, len(Zs))
        if top > 0:
            Zhat_s = (Zhat - mz) / (sz if sz != 0 else 1)
            idx_top = np.argsort(abs_resid)[-top:]
            for i in idx_top:
                ax.plot([Xs[i], Xs[i]], [Ys[i], Ys[i]], [Zhat_s[i], Zs[i]], linewidth=1.2)
    
        # zoom par quantiles dans l'espace standardisé
        ql, qh = zoom_q
        def lims(a):
            lo, hi = np.quantile(a, [ql, qh])
            pad = 0.06 * (hi - lo)
            return lo - pad, hi + pad
        xlo, xhi = lims(Xs); ylo, yhi = lims(Ys); zlo, zhi = lims(Zs)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
    
        # proportions d'axes équilibrées
        ax.set_box_aspect((xhi-xlo, yhi-ylo, zhi-zlo))
        ax.view_init(elev=elev, azim=azim)
        try: ax.dist = 7.0
        except Exception: pass
    
        ax.set_xlabel(f"{feats[0]} (z-score)")
        ax.set_ylabel(f"{feats[1]} (z-score)")
        ax.set_zlabel(f"{best_target} (z-score)")
        ax.set_title(f"{best_target} vs ({feats[0]}, {feats[1]}) — Plan OLS (échelles normalisées)")
    
        cb = fig.colorbar(sc, shrink=0.78, pad=0.06)
        cb.set_label("|résidu| (unités réelles)")
    
        ax.tick_params(pad=2, labelsize=9)
        plt.tight_layout()
        plt.show()




    # --------------- NEW: Coeffs & explications pour la stratégie BTC ---------------
    def print_btc_coefficients(self, results):
        if "BTC" not in results:
            print("Résultats BTC introuvables.")
            return
        bp = results["BTC"]["betas_price"]
        ap = results["BTC"]["alpha_price"]
        gr = results["BTC"]["gammas"]
        ar = results["BTC"]["alpha_ret"]

        print("\n=== Coefficients OLS (niveaux) — Plan prix ===")
        print("Forme : BTC = α + β_ETH · ETH + β_30Y · US30Y + ε")
        print(f"α (alpha_price)   = {ap:.6f}")
        print(f"β_ETH (betas_price['ETH'])   = {bp['ETH']:.6f}")
        print(f"β_30Y (betas_price['US30Y']) = {bp['US30Y']:.6f}")

        print("\n=== Coefficients OLS (rendements) — Hedge ===")
        print("Forme : r_BTC = a + γ_ETH · r_ETH + γ_30Y · r_30Y + η")
        print(f"a (alpha_ret)     = {ar:.6f}")
        print(f"γ_ETH (gammas['ETH'])   = {gr['ETH']:.6f}")
        print(f"γ_30Y (gammas['US30Y']) = {gr['US30Y']:.6f}")

    def print_quick_correlations(self):
        z = (self.after - self.after.mean()) / self.after.std(ddof=0)
        corr = z.corr()
        print("\n=== Corrélations (z-scores, post ETF) ===")
        print(corr.round(2).to_string())


    def plot_btc_spread_p95_trend(self, win=60, q=0.95):
        """
        BTC ~ 1 + ETH + US30Y sur les NIVEAUX (période post split_date).
        Résidu S_BTC -> z-score (statique), puis 95e pct roulant de |z|.
        On trace la série et une tendance linéaire. La pente est reportée
        par jour et *annualisée* (365j et 252j).
        """
        # --- Plan en niveaux pour BTC (signal = résidu) ---
        feats = ["ETH", "US30Y"]
        _, _, resid = self._fit_plane_levels("BTC", feats, self.after)
        z = self._z_static(resid).dropna()
    
        abs_z = z.abs()
        roll = abs_z.rolling(win, min_periods=max(30, win // 2)).quantile(q).dropna()
        if roll.empty:
            print("Série vide pour le rolling quantile — ajuste win/min_periods.")
            return
    
        # Axe temps : jours écoulés puis années (≈365.25 j)
        t_days = (roll.index - roll.index[0]).days.values.astype(float)
        t_years = t_days / 365.25  # calendrier
    
        # Tests non-paramétriques (inchangés)
        rho_s, p_s = spearmanr(t_days, roll.values)
        tau_k, p_k = kendalltau(t_days, roll.values)
    
        # Régression linéaire OLS sur *années* (pente = variation par an)
        lr = linregress(t_years, roll.values)   # slope_year, intercept, r, p, se
        slope_per_year_cal = lr.slope
        intercept = lr.intercept
        p_ols = lr.pvalue
        r2 = lr.rvalue**2
    
        # Dérivés utiles
        slope_per_day = slope_per_year_cal / 365.25            # calendrier
        slope_per_year_trading = slope_per_day * float(self.A) # ≈252 j ouvrés
    
        # Courbe de tendance (même abscisse "dates", mais x = années)
        trend = slope_per_year_cal * t_years + intercept
    
        # --- Plot ---
        plt.figure(figsize=(12,6))
        plt.plot(roll.index, roll.values,
                 label=f"Rolling {win}D {int(q*100)}e pct |z_S(BTC|ETH,US30Y)|")
        plt.plot(roll.index, trend, linewidth=2.0,
                 label=(f"Tendance linéaire  (pente/jour={slope_per_day:.4g}, "
                        f"pente/an(365j)={slope_per_year_cal:.4g}, "
                        f"pente/an(252j)={slope_per_year_trading:.4g}, "
                        f"p={p_ols:.3g}, R²={r2:.2f})"))
        plt.title(f"Extrêmes du spread BTC expliqué par (ETH, US30Y)\n"
                  f"{win}D {int(q*100)}e percentile de |z-résidu| + tendance (post {self.split_date.date()})")
        plt.xlabel("Date"); plt.ylabel(f"{int(q*100)}e pct de |z| (fenêtre {win}j)")
        plt.grid(True, alpha=0.3); plt.legend(frameon=False); plt.tight_layout(); plt.show()
    
        # --- Résumé console ---
        print("\n=== Pentes de la tendance (|z|-pct roulant) ===")
        print(f"Pente/jour (cal.)        : {slope_per_day:.6g}")
        print(f"Pente/an (365.25 jours)  : {slope_per_year_cal:.6g}")
        print(f"Pente/an (252 jours ouvrés): {slope_per_year_trading:.6g}")
        print("\n=== Tests de tendance ===")
        print(f"OLS (pente en/an) p-value={p_ols:.3g}, R²={r2:.2f}")
        print(f"Spearman ρ={rho_s:.3f} (p={p_s:.3g})")
        print(f"Kendall  τ={tau_k:.3f} (p={p_k:.3g})")


    # --------------- Orchestration ---------------
    def run_all(self):
        self.load()

        # 1) z-scores bruts pour visualiser la corrélation depuis split_date
        self.plot_zscores_after()
        self.print_quick_correlations()

        # 2) Retours post-ETF
        rets = self._returns(self.after)

        # 3) Sweep pour chaque cible (stratégie)
        results = {}
        for tgt in self.targets:
            results[tgt] = self._sweep_target(tgt, self.after, rets)
            if self.verbose:
                b = results[tgt]["best"]
                print(f"[{tgt}] best k={float(b['threshold']):.2f}, Sharpe={float(b['sharpe']):.2f}, "
                      f"AnnRet={float(b['ann_ret']):.2%}, AnnVol={float(b['ann_vol']):.2%}, "
                      f"MDD={float(b['mdd']):.2%}, Trades={int(b['n_trades'])}")

        # 4) Plots combinés
        self.plot_sharpe_vs_threshold_all(results)
        self.plot_equities_best_all(results)
        self.plot_best_plane_3d(results)
        self.plot_btc_spread_p95_trend(win=60, q=0.95)

        # 5) Résumé & Coefficients + Explication pour BTC
        summary = []
        for tgt in self.targets:
            b = results[tgt]["best"]
            summary.append({
                "Target": tgt,
                "k* (σ)": float(b["threshold"]),
                "Sharpe*": float(b["sharpe"]),
                "AnnRet*": float(b["ann_ret"]),
                "AnnVol*": float(b["ann_vol"]),
                "MDD*": float(b["mdd"]),
                "TotalRet*": float(b["total_ret"]),
                "Trades": int(b["n_trades"]),
            })
        self.summary_df = pd.DataFrame(summary).sort_values("Sharpe*", ascending=False).reset_index(drop=True)
        if self.verbose:
            print("\n=== Résumé (meilleur par stratégie) ===")
            print(self.summary_df.to_string(index=False))

        # NEW: coefficients & explication pour la stratégie BTC
        self.print_btc_coefficients(results)
     

        return results, self.summary_df
        

# =========================
# Utilisation
# =========================
if __name__ == "__main__":
    ta = TriArbRotator(
        csv_file="btc_us30y_daily.csv",     # colonnes: Date,BTC,ETH,US30Y
        split_date="2024-07-23",            # ETF ETH (mardi 23/07/2024)
        duration_30y=19.0,
        cost_btc=0.0005, cost_eth=0.0005, cost_bond=0.0001,
        thresholds=np.round(np.arange(0.50, 3.01, 0.01), 2),
        verbose=True
    )
    results, summary = ta.run_all()
