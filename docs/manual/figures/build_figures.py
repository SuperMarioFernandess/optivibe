"""Generate the physics figures embedded in OptiVibe_Руководство.docx (doc 15 §6).

Ported verbatim (numerics unchanged) from the S-DOC-1 monolith generator
``manual_mkfigs.py``; the only change is the output directory, which now resolves
to ``docs/manual/figures/`` next to this file instead of an absolute scratch
path, so ``node docs/manual/build.mjs`` finds the PNGs in the repo tree. Curves
and reference numbers come from the physics base (docs 02/03/07/08); Cyrillic is
rendered with the bundled DejaVu Sans (doc 15 §6).

Run directly (``python docs/manual/figures/build_figures.py``) or via the
build (``node docs/manual/build.mjs`` shells out to it when PNGs are missing).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 128,
        "savefig.dpi": 128,
        "font.size": 10.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
        "figure.autolayout": True,
    }
)
C = dict(
    blue="#1f5fb4",
    red="#c0392b",
    green="#1e8449",
    orange="#d68910",
    purple="#7d3c98",
    dark="#1b2631",
)
# Output next to this file: docs/manual/figures/.
F = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
b1 = 1.8751
sig1 = (np.cosh(b1) + np.cos(b1)) / (np.sinh(b1) + np.sin(b1))


def f1k(L):
    return 100.0 / L**2


# 1 mode shape
fig, ax = plt.subplots(figsize=(5.6, 3.2))
xi = np.linspace(0, 1, 300)
p1 = (np.cosh(b1 * xi) - np.cos(b1 * xi)) - sig1 * (np.sinh(b1 * xi) - np.sin(b1 * xi))
p1 /= p1[-1]
ax.plot(xi, p1, color=C["blue"], lw=2.4)
ax.axvline(0, color=C["dark"], lw=3)
ax.axhline(0, color=C["dark"], lw=0.6)
ax.text(0.02, -0.13, "заделка", fontsize=8, color=C["dark"])
ax.text(0.78, 0.55, "торец (z=L)", fontsize=8, color=C["dark"])
ax.set_xlabel("z/L")
ax.set_ylabel(r"$\varphi_1(z/L)$")
ax.set_title("Форма 1-й моды консоли (наклон у торца 1.377)")
ax.set_xlim(0, 1)
fig.savefig(F + "mode.png", bbox_inches="tight")
plt.close(fig)

# 2 f1 vs L
fig, ax = plt.subplots(figsize=(5.6, 3.2))
L = np.linspace(1, 6, 200)
ax.loglog(L, f1k(L), color=C["blue"], lw=2.4)
for Lp, n, c in [(1.41, "C", C["red"]), (2.0, "B", C["orange"]), (5.0, "A", C["green"])]:
    ax.plot(Lp, f1k(Lp), "o", color=c, ms=7, mec="k", mew=0.5)
    ax.annotate(
        f"{n}",
        (Lp, f1k(Lp)),
        textcoords="offset points",
        xytext=(6, 5),
        fontsize=9,
        color=c,
        fontweight="bold",
    )
ax.set_xlabel("L, мм")
ax.set_ylabel(r"$f_1$, кГц")
ax.set_title(r"$f_1\approx100/L^2$ кГц")
ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_yticks([3, 5, 10, 20, 50, 100])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig.savefig(F + "f1L.png", bbox_inches="tight")
plt.close(fig)

# 3 Hlat bode
fig, ax = plt.subplots(figsize=(5.8, 3.3))
f = np.logspace(-1, 5.2, 1500)


def H(f, Lmm, Q, Hqs):
    r = f / (f1k(Lmm) * 1e3)
    return Hqs * np.abs(1 / (1 - r**2 + 1j * r / Q))


ax.loglog(f, H(f, 1.41, 1950, 0.0384 * 1.41**4), color=C["red"], lw=2, label="C: L=1.41 мм")
ax.loglog(f, H(f, 2.0, 2610, 0.0384 * 2**4), color=C["orange"], lw=2, label="B: L=2.0 мм")
ax.loglog(
    f,
    H(f, 3.0, 2183, 0.0384 * 3**4),
    color=C["purple"],
    lw=1.8,
    ls="--",
    label="L=3 мм (резонанс в полосе)",
)
ax.axvspan(0.1, 20e3, color=C["green"], alpha=0.07)
ax.set_xlabel("f, Гц")
ax.set_ylabel(r"$|H_{lat}|$, нм/g")
ax.set_title(r"АЧХ $H_{lat}(f)=H^{QS}_{lat}\,|D(f)|$")
ax.legend(fontsize=7.5, loc="upper left")
ax.set_xlim(0.1, 2e5)
ax.set_ylim(1e-2, 1e4)
fig.savefig(F + "hlat.png", bbox_inches="tight")
plt.close(fig)

# 4 eta coupling
fig, ax = plt.subplots(figsize=(5.6, 3.2))
sx = 1.826
dx = np.linspace(-6, 6, 400)
eta = 0.42 * np.exp(-(dx**2) / (2 * sx**2))
ax.plot(dx, eta, color=C["blue"], lw=2.4)
dx0 = 2
e0 = 0.42 * np.exp(-(dx0**2) / (2 * sx**2))
sl = -e0 * dx0 / sx**2
xt = np.array([dx0 - 1.4, dx0 + 1.4])
ax.plot(xt, e0 + sl * (xt - dx0), color=C["red"], lw=1.6, ls="--")
ax.plot(dx0, e0, "o", color=C["red"], ms=8, mec="k", mew=0.5)
ax.annotate(
    f"bias $\\Delta x_0$=2 мкм\n$\\eta_0$≈0.25, наклон {sl:.2f}/мкм",
    (dx0, e0),
    textcoords="offset points",
    xytext=(8, -42),
    fontsize=8,
    color=C["red"],
)
ax.set_xlabel(r"$\Delta x$, мкм")
ax.set_ylabel(r"$\eta$")
ax.set_title("Оптическая связь η(Δx) и рабочая точка")
ax.set_xlim(-6, 6)
ax.set_ylim(0, 0.46)
fig.savefig(F + "eta.png", bbox_inches="tight")
plt.close(fig)

# 5 nea spectrum
fig, ax = plt.subplots(figsize=(5.8, 3.3))
f = np.logspace(-1, 4.6, 1500)
Lb = 2
Qb = 2610
f1b = f1k(Lb) * 1e3
r = f / f1b
Dm = np.abs(1 / (1 - r**2 + 1j * r / Qb))
shelf = 10.6
nopt = shelf / Dm * np.sqrt(1 + 4 / f)
nth = 0.57 * np.ones_like(f)
ax.loglog(f, nopt, color=C["blue"], lw=1.5, ls="--", label="дробовой (+1/f)")
ax.loglog(f, nth, color=C["green"], lw=1.5, ls=":", label="тепловой")
ax.loglog(f, np.sqrt(nopt**2 + nth**2), color=C["red"], lw=2.3, label="суммарный NEA")
ax.axvspan(1, 10e3, color=C["orange"], alpha=0.07)
ax.set_xlabel("f, Гц")
ax.set_ylabel(r"NEA, мкg/$\sqrt{Гц}$")
ax.set_title("Спектр NEA(f), вариант B")
ax.legend(fontsize=7.5)
ax.set_xlim(0.1, 6e4)
ax.set_ylim(0.3, 500)
fig.savefig(F + "nea.png", bbox_inches="tight")
plt.close(fig)

# 6 family
fig, ax = plt.subplots(figsize=(5.8, 3.4))
for lo, hi, col, lab in [
    (0.001, 1, "#d5f5e3", "прецизионный"),
    (1, 10, "#d6eaf8", "навигационный"),
    (10, 50, "#fdebd0", "индустриальный"),
    (50, 300, "#fadbd8", "широкополосный"),
]:
    ax.axhspan(lo, hi, color=col, alpha=0.7)
    ax.text(0.06, np.sqrt(lo * hi), lab, fontsize=7.5, va="center", color=C["dark"])
for fmx, nea, lab, col in [
    (0.1, 0.3, "A", C["green"]),
    (10, 10.6, "B", C["orange"]),
    (20, 42.6, "C r2", C["red"]),
    (20, 11.3, "C r1", C["purple"]),
    (5, 0.22, "D", C["blue"]),
]:
    ax.plot(fmx, nea, "o", color=col, ms=10, mec="k", mew=0.6)
    ax.annotate(
        lab,
        (fmx, nea),
        textcoords="offset points",
        xytext=(7, 5),
        fontsize=8.5,
        color=col,
        fontweight="bold",
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$f_{max}$, кГц")
ax.set_ylabel(r"NEA, мкg/$\sqrt{Гц}$")
ax.set_title("Семейство A/B/C/D и классы применений")
ax.set_xlim(0.05, 30)
ax.set_ylim(0.05, 400)
ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig.savefig(F + "family.png", bbox_inches="tight")
plt.close(fig)

print("figs:", sorted(p for p in os.listdir(F) if p.endswith(".png")))
