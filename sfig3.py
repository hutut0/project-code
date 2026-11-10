"""Supplementary Fig. 3 (v4, new) — robustness of the outcome analyses.
a  Cox HR vs Fine–Gray subdistribution HR (death as competing event); moved from v3 Fig. 3d.
b  Cardiac axis → brain outcomes: main model vs censoring at the first incident AF, HF or MI; moved from v3 Fig. 4d.
All numbers are read from frozen result files (B1_fine_gray, B1_full_matrix_cox, C3_censor_at_cardiac_event).
"""
import numpy as np
import pandas as pd
from nclib import (panel, COL, ci_point, refline, clean_y, legend_handles, log_x, log_y)
from nc_data import t, one, OUT_SHORT, EXPO_LAB

F = 'SFig3'


# ------------------------------------------------------------------ a  Cox vs Fine–Gray
@panel(F, 'a', 80, 70, 'Robust to the competing risk of death')
def a(c):
    fg = t('B', 'B1_fine_gray'); m = t('B', 'B1_full_matrix_cox')
    D = fg.merge(m[['outcome', 'exposure', 'HR', 'lo', 'hi', 'events']], on=['outcome', 'exposure'], suffixes=('_fg', '_cox'))
    ax = c.ax(14, 12, 60, 50)
    for x, col in [('CardiacAxis', COL['heart']), ('BrainAxis', COL['joint'])]:
        s = D[D.exposure == x]
        ax.scatter(s.HR, s.sHR, s=9, c=col, lw=0.3, edgecolors='white', zorder=3, label=EXPO_LAB[x])
    lo, hi = 0.75, 4.2
    ax.plot([lo, hi], [lo, hi], color=COL['mid'], lw=0.5, ls=(0, (2.5, 1.5)), zorder=1)
    log_x(ax, (lo, hi), [1, 2, 4]); log_y(ax, (lo, hi), [1, 2, 4])
    ax.set_xlabel('Cox HR per SD'); ax.set_ylabel('Fine–Gray subdistribution HR')
    r = np.corrcoef(np.log(D.HR), np.log(D.sHR))[0, 1]
    dmax = np.max(np.abs(np.log(D.HR) - np.log(D.sHR)))
    ax.text(0.04, 0.96, f'{len(D)} outcome–exposure pairs\nr (log scale) = {r:.4f}\nmax |Δ log HR| = {dmax:.3f}',
            transform=ax.transAxes, fontsize=5.4, va='top', linespacing=1.2)
    ax.legend(loc='lower right', fontsize=5.4, handletextpad=0.1, borderaxespad=0.3)
    D = D.assign(outcome_en=D.outcome.map(OUT_SHORT), exposure_en=D.exposure.map(EXPO_LAB))
    c.sd(D, f'Cox HR vs Fine–Gray subdistribution HR (death as competing event), outcomes with ≥ 50 events; '
            f'r (log) = {r:.4f}; max |Δ log HR| = {dmax:.3f}')


# ------------------------------------------------------------------ b  censoring at first cardiac event
@panel(F, 'b', 103, 70, 'Unchanged when incident cardiac events are censored')
def b(c):
    cs = t('C', 'C3_censor_at_cardiac_event'); m = t('B', 'B1_full_matrix_cox')
    outs = ['Stroke', 'Ischaemic_stroke', 'P1_major_neurological', 'Dementia']
    ax = c.ax(30, 16, 46, 44)
    rec = []
    for i, o in enumerate(outs):
        r0 = one(m, outcome=o, exposure='CardiacAxis'); r1 = one(cs, outcome=o)
        ci_point(ax, r0.HR, i - 0.17, r0.lo, r0.hi, COL['heart'], filled=r0.p < 0.05, ms=3.0, lw=0.7)
        ci_point(ax, r1.HR, i + 0.17, r1.lo, r1.hi, COL['heart'], filled=r1.p < 0.05, ms=3.0, lw=0.7, marker='D')
        for y_, rr in [(i - 0.17, r0), (i + 0.17, r1)]:
            ax.text(1.03, y_, f'{rr.HR:.2f} ({rr.lo:.2f}–{rr.hi:.2f})', transform=ax.get_yaxis_transform(), fontsize=5.0,
                    va='center', ha='left', color=COL['ink'])
            ax.text(1.40, y_, f'{int(rr.events)}', transform=ax.get_yaxis_transform(), fontsize=5.0, va='center',
                    ha='center', color=COL['mid'])
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
        rec.append(dict(outcome=o, outcome_en=OUT_SHORT[o], HR_main=r0.HR, lo_main=r0.lo, hi_main=r0.hi,
                        events_main=r0.events, HR_censored=r1.HR, lo_censored=r1.lo, hi_censored=r1.hi,
                        events_censored=r1.events, n_censored_at_cardiac_event=r1.censored_at_cardiac_event))
    ax.text(1.03, -0.75, 'HR (95% CI)', transform=ax.get_yaxis_transform(), fontsize=5.1, ha='left', va='center',
            color=COL['mid'])
    ax.text(1.40, -0.75, 'Events', transform=ax.get_yaxis_transform(), fontsize=5.1, ha='center', va='center',
            color=COL['mid'])
    ax.set_ylim(len(outs) - 0.5, -0.9)
    labs = [OUT_SHORT[o] + ('\n(descriptive)' if o == 'Dementia' else '') for o in outs]
    clean_y(ax, labs, fs=5.6)
    refline(ax, 1)
    log_x(ax, (0.7, 3.2), [1, 2, 3])
    ax.set_xlabel('Cardiac axis HR per SD (95% CI)', labelpad=1)
    c.legend(legend_handles([('mk', 'Main model', COL['heart'], 'o', True),
                             ('mk', 'Prior cardiac disease excluded; censored at incident AF/HF/MI', COL['heart'], 'D', True),
                             ('pt', 'Open: 95% CI includes 1', COL['base'], False)]),
             1, 0.2, ncol=2, fs=5.1, labelspacing=0.15, columnspacing=0.8)
    c.sd(pd.DataFrame(rec), 'Cardiac axis → brain outcomes, main model vs exclusion of prior cardiac disease and censoring at '
                            'the first incident AF, HF or MI (C3)')


LAYOUT = [(F, 'a', 0, 0), (F, 'b', 80, 0)]
HEIGHT = 70
