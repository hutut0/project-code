"""Submission Fig. 6 (in-house Fig7) — v4: incremental prediction across outcomes and beyond PREVENT (C1, B5, H5).
v4: counts in legend (a); correspondence criterion annotated (b); pre-specified added-value column (c, from H5 detail);
incremental net benefit over PREVENT (d); log-scale decile calibration (e).

Wording discipline: 'incremental prediction' only; PREVENT is computed from baseline risk factors measured a median
of ~9 years before imaging (value read from derived/D1_prevent_timing_summary.csv); models need recalibration.
"""
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from nclib import (panel, COL, ORGAN_COL, ci_point, refline, clean_y, legend_handles)
from nc_data import t, sel, one, OUT_SHORT, OUT_ORDER, H as HB_ROOT
from pathlib import Path
H5X = HB_ROOT / 'outcomes/deliverables/03_阶段B多病种心脑结局_全部结果.xlsx'
CEREBRO = {'P1_major_neurological', 'Stroke', 'Ischaemic_stroke', 'ICH', 'TIA'}  # composite + cerebrovascular family


def _h5():
    d = pd.read_excel(H5X, sheet_name='09b_H5明细')
    return d[d.baseline.str.startswith('标准临床评分')].copy()

F = 'Fig7'
MCOL = {'+心脏MRI': COL['heart'], '+脑MRI': COL['brain'], '+心脏MRI+脑MRI': COL['joint']}
MLAB = {'+心脏MRI': '+ Cardiac MRI', '+脑MRI': '+ Brain MRI', '+心脏MRI+脑MRI': '+ Cardiac + brain MRI'}
ORG_EN = {'心': 'Cardiac', '心脑': 'Heart–brain', '脑': 'Brain', '对照': 'Control'}
PREV = '标准临床评分'
B5_ORDER = ['P2_major_cardiac', 'MACE', 'Heart_failure', 'CHD', 'MI', 'Stroke', 'Ischaemic_stroke']


def _timing():
    d = one(t('D', 'D1_prevent_timing_summary'), sample='PREVENT-scored participants')
    return d


# ------------------------------------------------------------------ a  prediction atlas
@panel(F, 'a', 183, 66, 'Incremental discrimination across 25 pre-specified, modelled outcomes')
def a(c):
    A = t('C', 'C1_prediction_atlas')
    base = A[A.model == '基线'].set_index('outcome')
    org_order = {'心': 0, '心脑': 1, '脑': 2, '对照': 3}
    outs = sorted(base.index, key=lambda k: (org_order[base.loc[k, 'organ']], OUT_ORDER.index(k)))
    xs, x, prev = {}, 0.0, None
    for k in outs:
        o = base.loc[k, 'organ']
        if prev is not None and o != prev:
            x += 0.8
        xs[k] = x; x += 1; prev = o
    ax = c.ax(13, 19, 166, 33)
    off = {'+心脏MRI': -0.24, '+脑MRI': 0.0, '+心脏MRI+脑MRI': 0.24}
    for m, dx in off.items():
        for k in outs:
            r = one(A, outcome=k, model=m)
            ci_point(ax, xs[k] + dx, r.dC, r.dC_lo, r.dC_hi, MCOL[m], filled=r.dC_FDR < 0.05, ms=2.5, lw=0.6, orient='v')
    refline(ax, 0, orient='h')
    # organ bands
    for o in ['心', '心脑', '脑', '对照']:
        ks = [k for k in outs if base.loc[k, 'organ'] == o]
        x0, x1 = xs[ks[0]] - 0.5, xs[ks[-1]] + 0.5
        ax.axvspan(x0, x1, color=COL['light'] if o in ('心', '脑') else 'white', lw=0, zorder=0)
        ax.text((x0 + x1) / 2, 1.02, ORG_EN[o] + ' outcomes' if o != '对照' else 'Negative controls',
                transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=5.7, color=ORGAN_COL[o],
                fontweight='bold')
    ax.set_xlim(-0.7, x - 0.3)
    ax.set_xticks([xs[k] for k in outs])
    labs = [OUT_SHORT[k] + (' ‡' if bool(base.loc[k, 'descriptive']) else '') for k in outs]
    ax.set_xticklabels(labs, rotation=40, ha='right', rotation_mode='anchor', fontsize=5.4)
    for tl, k in zip(ax.get_xticklabels(), outs):
        tl.set_color(ORGAN_COL[base.loc[k, 'organ']] if not bool(base.loc[k, 'descriptive']) else COL['mid'])
    ax.tick_params(axis='x', length=0, pad=1)
    ax.set_ylabel('ΔC-index vs risk-factor model\n(out-of-fold, 95% CI)', linespacing=1.1)
    ax.spines['bottom'].set_visible(False)
    dis = A[(A.model != '基线') & (A.organ != '对照')]
    nd = dis.outcome.nunique()
    cnt = {m: int((dis[dis.model == m].dC_FDR < 0.05).sum()) for m in off}
    c.legend(legend_handles([('pt', f'{MLAB[m]} ({cnt[m]}/{nd} outcomes, FDR < 0.05)', MCOL[m], True) for m in off] +
                            [('pt', 'Open: FDR ≥ 0.05', COL['base'], False)]),
             182, 60.2, loc='upper right', ncol=4, fs=5.3, columnspacing=0.8)
    c.text(182, 12.0, '‡ descriptive\n(20–39 events)', fontsize=5.0, ha='right', va='top', color=COL['mid'], linespacing=1.1)
    sd = A.copy(); sd['outcome_en'] = sd.outcome.map(OUT_SHORT); sd['model_en'] = sd.model.map(MLAB).fillna('Risk-factor model')
    c.sd(sd, 'Out-of-fold C-index and ΔC versus the risk-factor model; FDR across outcomes within model; counts: ' +
         '; '.join(f'{MLAB[m]} {cnt[m]}/{nd}' for m in off) + ' disease outcomes (negative controls excluded)')


# ------------------------------------------------------------------ b  organ specificity
@panel(F, 'b', 66, 80, 'Disease–imaging correspondence')
def b(c):
    O = t('C', 'C1_organ_specificity').sort_values('diff').reset_index(drop=True)
    ax = c.ax(30, 22, 33, 51)
    for i, r in O.iterrows():
        col = COL['heart'] if r['diff'] > 0 else COL['brain']
        sig = (r.diff_lo > 0) or (r.diff_hi < 0)
        ax.plot([0, r['diff']], [i, i], color=col, lw=0.7, zorder=2)
        ax.plot([r.diff_lo, r.diff_hi], [i, i], color=col, lw=0.35, alpha=0.6, zorder=1)
        ax.plot([r['diff']], [i], 'o', ms=2.8, mfc=col if sig else 'white', mec=col, mew=0.7, zorder=3)
    refline(ax, 0)
    ax.set_ylim(len(O) - 0.4, -0.6)
    clean_y(ax, [OUT_SHORT[k] for k in O.outcome], fs=5.2, color=[ORGAN_COL[o] for o in O.organ])
    ax.set_xlabel('ΔC (cardiac MRI) − ΔC (brain MRI)', labelpad=1)
    ax.set_xlim(-0.16, 0.14)
    ax.text(0.02, 1.0, '← brain MRI', transform=ax.transAxes, fontsize=5.2, color=COL['brain'], ha='left', va='bottom')
    ax.text(0.98, 1.0, 'cardiac MRI →', transform=ax.transAxes, fontsize=5.2, color=COL['heart'], ha='right', va='bottom')
    car = O[O.organ == '心']; nsb = O[(O.organ == '脑') & (~O.outcome.isin(CEREBRO))]
    k_c, k_b = int((car['diff'] > 0).sum()), int((nsb['diff'] < 0).sum())
    assert (k_c, len(car), k_b, len(nsb)) == (7, 8, 5, 8), (k_c, len(car), k_b, len(nsb))
    c.text(1, 5.0, f'Larger ΔC with cardiac MRI: {k_c}/{len(car)} cardiac outcomes;\nwith brain MRI: {k_b}/{len(nsb)} non-stroke brain '
                   f'outcomes (< 2/3 criterion)', fontsize=5.0, color=COL['ink'], va='bottom', linespacing=1.12)
    c.legend(legend_handles([('pt', '95% CI excludes 0', COL['base'], True), ('pt', 'includes 0', COL['base'], False)]),
             1, 0.2, ncol=2, fs=5.2)
    sd = O.copy(); sd['outcome_en'] = sd.outcome.map(OUT_SHORT)
    sd['group_for_criterion'] = np.where(sd.organ == '心', 'cardiac', np.where((sd.organ == '脑') & ~sd.outcome.isin(CEREBRO),
                                         'non-stroke brain', 'not counted'))
    c.sd(sd, 'Difference in ΔC between cardiac and brain MRI added to the risk-factor model (bootstrap CI)')


# ------------------------------------------------------------------ c  beyond PREVENT
@panel(F, 'c', 117, 80, 'Added value over PREVENT')
def cpanel(c):
    D = sel(t('B', 'B5_discrimination_calibration'), baseline=PREV)
    N = sel(t('B', 'B5_NRI'), baseline=PREV)
    outs = [k for k in B5_ORDER if k in set(D.outcome)]
    tm = _timing()
    axC = c.ax(31, 18, 31, 53); axN = c.ax(67, 18, 31, 53)
    H5 = _h5()
    off = {'+心脏MRI': -0.24, '+脑MRI': 0.0, '+心脏MRI+脑MRI': 0.24}
    rec = []
    for i, k in enumerate(outs):
        b0 = one(D, outcome=k, model='基线')
        axC.text(-0.02, i + 0.33, f'PREVENT C {b0.C:.2f}', transform=axC.get_yaxis_transform(), fontsize=5.0,
                 color=COL['mid'], ha='right', va='center')
        for m, dx in off.items():
            r = one(D, outcome=k, model=m); n_ = one(N, outcome=k, model=m)
            ci_point(axC, r.dC, i + dx, r.dC_lo, r.dC_hi, MCOL[m], filled=(r.dC_lo > 0) or (r.dC_hi < 0), ms=2.5, lw=0.6)
            ci_point(axN, n_.NRI, i + dx, n_.NRI_lo, n_.NRI_hi, MCOL[m], filled=(n_.NRI_lo > 0) or (n_.NRI_hi < 0),
                     ms=2.5, lw=0.6)
            h = one(H5, outcome=k, model=m)
            star = bool(h.all_met)
            axN.plot([1.10 + list(off).index(m) * 0.085], [i], transform=axN.get_yaxis_transform(), clip_on=False,
                     marker='o', ms=3.2 if star else 1.4, mfc=MCOL[m] if star else COL['ctrl'], mec='none', ls='none')
            rec.append(dict(outcome=k, outcome_en=OUT_SHORT[k], model=MLAB[m], n=r.n, events=r.events, C_PREVENT=b0.C,
                            H5_dC=bool(h.crit_dC), H5_NRI=bool(h.crit_NRI), H5_NB=bool(h.crit_NB), H5_all_met=star,
                            H5_positive_thresholds=int(h.incremental_NB_positive_thresholds),
                            C=r.C, dC=r.dC, dC_lo=r.dC_lo, dC_hi=r.dC_hi, NRI=n_.NRI, NRI_events=n_.NRI_events,
                            NRI_nonevents=n_.NRI_nonevents, NRI_lo=n_.NRI_lo, NRI_hi=n_.NRI_hi, calib_slope=r.calib_slope,
                            O_to_E=r.O_to_E))
        if i % 2 == 0:
            for ax_ in (axC, axN):
                ax_.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
    for ax_ in (axC, axN):
        refline(ax_, 0); ax_.set_ylim(len(outs) - 0.5, -0.5)
    clean_y(axC, [OUT_SHORT[k] for k in outs], fs=5.6,
            color=[COL['brain'] if 'troke' in k else (COL['joint'] if k == 'MACE' else COL['heart']) for k in outs])
    axC.set_yticks(range(len(outs)))
    axN.set_yticks([]); axN.spines['left'].set_visible(False)
    axC.set_xlabel('ΔC-index over PREVENT', labelpad=1); axN.set_xlabel('Continuous NRI over PREVENT', labelpad=1)
    for lab in axC.get_yticklabels():
        lab.set_va('bottom')
    axN.text(1.10 + 0.085, -0.75, 'Pre-specified\nadded value', transform=axN.get_yaxis_transform(), fontsize=5.0,
             ha='center', va='bottom', linespacing=1.0)
    si = outs.index('Stroke') if 'Stroke' in outs else None
    if si is not None:
        axN.text(1.10 + 2 * 0.085 + 0.05, si, '*', transform=axN.get_yaxis_transform(), fontsize=6, ha='left', va='center',
                 color=COL['joint'])
    c.legend(legend_handles([('pt', MLAB[m], MCOL[m], True) for m in off]), 30, 77.0, loc='upper left', ncol=3, fs=5.2,
             columnspacing=0.7)
    c.text(1, 6.0, f'PREVENT computed from baseline risk factors measured a median of {tm.median_years:.1f} years '
                   f'(IQR {tm.q1_years:.1f}–{tm.q3_years:.1f}) before imaging; open: 95% CI includes 0',
           fontsize=5.0, color=COL['mid'])
    c.text(1, 0.3, 'Pre-specified added value (filled dot): ΔC and NRI lower bounds > 0 and positive incremental net benefit\n'
                   'at ≥ 2 of 4 thresholds; * criterion met numerically, but increment ≤ 0.22 per 1,000', fontsize=5.0,
           color=COL['mid'], linespacing=1.1)
    c.sd(pd.DataFrame(rec), f'Increment over PREVENT (n = {int(D.n.min()):,}–{int(D.n.max()):,}); PREVENT timing: '
                            f'median {tm.median_years:.2f} y (IQR {tm.q1_years:.2f}–{tm.q3_years:.2f}) before imaging')


# ------------------------------------------------------------------ d  incremental net benefit over PREVENT
DC_OUT = [('P2_major_cardiac', 0.10), ('Stroke', 0.05), ('Ischaemic_stroke', 0.05)]


@panel(F, 'd', 112, 58, 'Incremental net benefit over PREVENT')
def d(c):
    DC = sel(t('B', 'B5_decision_curve'), baseline=PREV)
    H5 = _h5()
    mods = ['+心脏MRI', '+脑MRI', '+心脏MRI+脑MRI']
    rec = []
    for j, (k, xmax) in enumerate(DC_OUT):
        ax = c.ax(12 + j * 34, 17, 28, 30)
        s_ = sel(DC, outcome=k)
        b0 = s_[s_.model == '基线'].set_index('threshold').net_benefit
        pts = sorted(s_[s_.prespecified.astype(bool)].threshold.unique())
        for p_ in pts:
            ax.axvline(p_ * 100, color='#DADDE3', lw=0.6, zorder=0)
        for m in mods:
            q = s_[(s_.model == m) & (s_.threshold <= xmax + 1e-9)].set_index('threshold').net_benefit.sort_index()
            dnb = (q - b0.reindex(q.index)) * 1000
            vis = dnb[dnb.index >= 0.01 - 1e-9]
            ax.plot(vis.index * 100, vis.values, color=MCOL[m], lw=0.9, zorder=3)
            pv = dnb.reindex(pts)
            for p_, v_ in pv.items():
                ax.plot([p_ * 100], [v_], marker='o', ms=2.6, mfc=MCOL[m] if v_ > 0 else 'white', mec=MCOL[m], mew=0.6, zorder=4)
            rec.append(pd.DataFrame({'outcome': k, 'outcome_en': OUT_SHORT[k], 'model': MLAB[m], 'threshold': dnb.index,
                                     'incremental_NB_per_1000': dnb.values,
                                     'prespecified': [x_ in pts for x_ in dnb.index]}))
        refline(ax, 0, orient='h')
        if k in ('Stroke', 'Ischaemic_stroke'):
            ax.set_ylim(-0.03, 0.25)  # shared scale for the two stroke outcomes
        ax.set_xlim(0.8, xmax * 100 + 0.2)
        ax.set_xticks([1, 3, 5] + ([10] if xmax >= 0.1 else []))
        ax.set_xlabel('Threshold risk (%)', labelpad=1)
        if j == 0:
            ax.set_ylabel('Δ net benefit per 1,000\n(model − PREVENT)', linespacing=1.1)
        ax.set_title(OUT_SHORT[k], fontsize=6, color=COL['heart'] if 'P2' in k else COL['brain'], pad=9)
        for mi, m in enumerate(mods):
            npos = int(one(H5, outcome=k, model=m).incremental_NB_positive_thresholds)
            ax.text(0.5 + (mi - 1) * 0.26, 1.02, f'{npos}/4', transform=ax.transAxes, fontsize=5.1, color=MCOL[m],
                    ha='center', va='bottom', fontweight='bold')
    c.legend(legend_handles([('line', MLAB[m], MCOL[m], '-') for m in mods] +
                            [('pt', 'Pre-specified threshold, Δ ≤ 0', COL['base'], False)]), 1, 3.6, ncol=4, fs=5.1,
             columnspacing=0.7, handlelength=1.4)
    c.text(111, 0.6, 'n/4: pre-specified thresholds (1, 2, 3, 5%) with positive increment; curves from 1%', fontsize=5.0,
           color=COL['mid'], ha='right', va='bottom')
    sd = pd.concat(rec)
    c.sd(sd, 'Incremental net benefit of imaging-augmented models over PREVENT (per 1,000); pre-specified thresholds 1, 2, 3 '
             'and 5%; counts of positive thresholds from the pre-specified H5 evaluation')


# ------------------------------------------------------------------ e  calibration
CAL_OUT = [('P2_major_cardiac', 0.10), ('Stroke', 0.05)]


@panel(F, 'e', 71, 58, 'Calibration: recalibration required')
def e(c):
    CD = sel(t('B', 'B5_calibration_deciles'), baseline=PREV)
    DCm = sel(t('B', 'B5_discrimination_calibration'), baseline=PREV)
    rec = []; anyzero = False
    for j, (k, _) in enumerate(CAL_OUT):
        ax = c.ax(11 + j * 31, 17, 25, 31)
        s_ = sel(CD, outcome=k)
        lo = min(s_.mean_pred.min(), s_.observed[s_.observed > 0].min()) * 100 * 0.7
        hi = max(s_.mean_pred.max(), s_.observed.max()) * 100 * 1.4
        ax.plot([lo, hi], [lo, hi], color=COL['mid'], lw=0.4, ls=(0, (2.5, 1.5)), zorder=0)
        txt = []
        for m, col, mk in [('基线', COL['base'], 'o'), ('+心脏MRI+脑MRI', COL['joint'], 's')]:
            q = s_[s_.model == m].sort_values('decile')
            pos = q.observed > 0
            ax.plot(q.mean_pred[pos] * 100, q.observed[pos] * 100, ls='none', marker=mk, ms=2.6, mfc=col,
                    mec='white', mew=0.3, zorder=3, alpha=0.95)
            if (~pos).any():
                anyzero = True
                ax.plot(q.mean_pred[~pos] * 100, [lo * 1.08] * int((~pos).sum()), ls='none', marker=mk, ms=2.4,
                        mfc='white', mec=col, mew=0.5, zorder=3, clip_on=False)
            r = one(DCm, outcome=k, model=m)
            txt.append((col, f"slope {r.calib_slope:.2f}\nO/E {r.O_to_E:.2f}"))
            q = q.copy(); q['outcome_en'] = OUT_SHORT[k]; q['calib_slope'] = r.calib_slope; q['O_to_E'] = r.O_to_E
            rec.append(q)
        for i_, (col, s2) in enumerate(txt):
            ax.text(0.04, 0.97 - i_ * 0.2, s2, transform=ax.transAxes, fontsize=5.0, color=col, va='top', linespacing=1.05)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
        tk = [v for v in [0.1, 0.3, 1, 3, 10] if lo <= v <= hi]
        for a_ in (ax.xaxis, ax.yaxis):
            a_.set_major_locator(FixedLocator(tk)); a_.set_minor_formatter(NullFormatter())
            a_.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlabel('Predicted (%)', labelpad=1)
        if j == 0:
            ax.set_ylabel('Observed (%)')
        ax.set_title(OUT_SHORT[k], fontsize=6, color=COL['heart'] if 'P2' in k else COL['brain'], pad=2)
    c.legend(legend_handles([('mk', 'PREVENT', COL['base'], 'o', True), ('mk', '+ Cardiac + brain MRI', COL['joint'], 's', True)]),
             1, 3.6, ncol=2, fs=5.1)
    if anyzero:
        c.text(1, 0.6, 'Open markers on the axis floor: deciles with no observed event', fontsize=5.0, color=COL['mid'])
    sd = pd.concat(rec); sd['model_en'] = sd.model.map({'基线': 'PREVENT', '+心脏MRI+脑MRI': '+ Cardiac + brain MRI'})
    c.sd(sd, 'Observed vs mean predicted risk by decile (log axes; no decile had zero observed risk); calibration '
             'slope > 1 indicates predictions that are too conservative (recalibration required)')


LAYOUT = [(F, 'a', 0, 0), (F, 'b', 0, 66), (F, 'c', 66, 66), (F, 'd', 0, 146), (F, 'e', 112, 146)]
HEIGHT = 204
