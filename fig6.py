"""Submission Fig. 5 (in-house Fig6) — v4: genetic triangulation (S3, S5, S9, M5, BP-MVMR, G1–G6).
v4 layout: forests carry an "OR (95% CI)" column; middle row c | d | e compacted; triangulation (f) is full width with
verdict badges and question grouping (Q1 mechanism, Q3 stroke).

Evidence discipline (locked): WMH → ischaemic stroke is grade D (MR-Egger imprecise, not pleiotropy);
the genetic cardiac axis failed the G1 positive control, so its stroke result is *inconclusive* (framework v4, E18).
"""
import json
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from nclib import (panel, COL, CMAP_DIV, ci_point, refline, clean_y, legend_handles, heat_colorbar, log_x, fmt_p, rounded_box)
from nc_data import t, sel, one, P, H as HB_ROOT

F = 'Fig6'
GDIR = P['G']
OUT_EN = {'HF_HERMES': 'Heart failure', 'AF_Nielsen': 'Atrial fibrillation', 'CAD_Nikpay': 'Coronary artery disease',
          'Stroke': 'Ischaemic stroke', 'AD_Kunkle': "Alzheimer's disease", 'PD_Nalls': "Parkinson's disease"}


def _orci(b, se):
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)


def _forest(c, ax, rows, xlim, ticks, label_w, fcond=True, fs=5.4, num_x=None, f_x=1.06):
    """rows: list of dicts (group header rows have key 'header'). num_x: axes-fraction x of the OR (95% CI) column."""
    y = 0; ys, labs, cols = [], [], []
    for r in rows:
        if 'header' in r:
            ax.text(r.get('hx', -0.02), y, r['header'], transform=ax.get_yaxis_transform(), fontsize=5.6,
                    fontweight='bold', color=r.get('hc', COL['base']), va='center', ha='left')
            y += 1; continue
        ci_point(ax, r['OR'], y, r['lo'], r['hi'], r['col'], filled=r.get('filled', True), ms=2.6, lw=0.7,
                 marker=r.get('marker', 'o'))
        if r.get('shade'):
            ax.axhspan(y - 0.5, y + 0.5, color=COL['light'], lw=0, zorder=0)
        if fcond and np.isfinite(r.get('F', np.nan)):
            ax.text(f_x, y, f"{r['F']:.1f}", transform=ax.get_yaxis_transform(), fontsize=5.2, ha='center', va='center',
                    color=COL['bp'] if r['F'] < 10 else COL['ink'])
        if num_x is not None:
            sig = r['lo'] > 1 or r['hi'] < 1
            ax.text(num_x, y, f"{r['OR']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})", transform=ax.get_yaxis_transform(), fontsize=5.0,
                    ha='left', va='center', color=COL['ink'], fontweight='bold' if sig else 'normal')
        if r.get('note'):
            ax.text(r['hi'] * 1.04, y, r['note'], fontsize=5.0, va='center', color=r.get('note_col', COL['mid']))
        ys.append(y); labs.append(r['label']); cols.append(r.get('lc', COL['ink'])); y += 1
    ax.set_ylim(y - 0.5, -0.7)
    clean_y(ax, labs, ys, fs=fs, color=cols)
    refline(ax, 1)
    log_x(ax, xlim, ticks)
    if fcond:
        ax.text(f_x, -1.2, '$F_{cond}$', transform=ax.get_yaxis_transform(), fontsize=5.4, ha='center', va='center')
    if num_x is not None:
        ax.text(num_x, -1.2, 'OR (95% CI)', transform=ax.get_yaxis_transform(), fontsize=5.2, ha='left', va='center',
                color=COL['mid'])
    return y


# ------------------------------------------------------------------ a  WMH -> ischaemic stroke, segmented forest
@panel(F, 'a', 108, 86, 'WMH → ischaemic stroke: consistent across methods and samples')
def a(c):
    s3 = one(t('S', 'S3_univariable_summary'), exposure='WMH', outcome='Stroke')
    m5 = t('M5', 'M5_forest_source_table'); w = m5[m5.panel == 'WMH'].assign(label=lambda d_: d_.label.str.strip()).set_index('label')
    s9m = one(t('S', 'S9_axis_mvmr'), exposure='WMH')
    s5 = t('S', 'S5_mvmr')
    b2 = sel(t('BP', 'B02_all_model_estimates'), method='IVW', exposure='WMH')
    Fsbp = one(b2, model='WMH+SBP__Stroke').F_conditional_standard
    Fs4sbp = one(b2, model='CardiacAxisS4+WMH+SBP__Stroke').F_conditional_standard
    B = COL['brain']
    rows = [dict(header='Observational')]
    o = w.loc['Observational HR per SD (3.4-y incidence)']
    rows.append(dict(label='HR per SD measured WMH', OR=o.OR, lo=o.lo, hi=o.hi, col=COL['base'], src='M5 (B1)'))
    rows.append(dict(header='Genetic, trans-ancestry IS GWAS (20 SNPs)'))
    for lab, b, se, note in [('IVW', s3.b_IVW, s3.se_IVW, ''), ('Weighted median', s3.b_WM, s3.se_WM, ''),
                             ('Weighted mode', s3.b_Mode, s3.se_Mode, ''),
                             ('MR-PRESSO (2 outliers removed)', s3.presso_b_corr, s3.presso_se_corr, ''),
                             ('MR-Egger (imprecise)', s3.b_Egger, s3.se_Egger, '')]:
        OR, lo, hi = _orci(b, se)
        rows.append(dict(label=lab, OR=OR, lo=lo, hi=hi, col=B, note=note, note_col=COL['ink'], src='S3',
                         filled=(lo > 1 or hi < 1)))
    rows.append(dict(header='Conditioned on the heart (MVMR)'))
    rows.append(dict(label='| genetic cardiac axis', OR=s9m.OR_mv, lo=s9m.OR_lo, hi=s9m.OR_hi, col=B, F=s9m.condF, src='S9'))
    for ht in ['LVEDV', 'LVESV', 'LVEF']:
        r = one(s5, model=f'{ht}+WMH', exposure='WMH')
        rows.append(dict(label=f'| {ht}', OR=r.OR_mv, lo=r.OR_mv_lo, hi=r.OR_mv_hi, col=B, F=r.condF_ldsc, src='S5'))
    rows.append(dict(header='Conditioned on SBP (MVMR)'))
    for lab, key, Fv in [('| SBP, IVW', 'MVMR | SBP (IVW)', Fsbp), ('| SBP, Q-het robust', 'MVMR | SBP (Q-het robust)', Fsbp),
                         ('| SBP + cardiac axis, Q-het robust', 'MVMR | S4 + SBP (Q-het robust)', Fs4sbp)]:
        r = w.loc[key]
        rows.append(dict(label=lab, OR=r.OR, lo=r.lo, hi=r.hi, col=COL['bp'], F=Fv, src='M5/B02'))
    rows.append(dict(header='Independent replication (MEGASTROKE)'))
    for lab, key in [('Any ischaemic stroke', 'MEGASTROKE ischaemic (independent)'),
                     ('Large-artery', 'MEGASTROKE large-artery'), ('Small-vessel', 'MEGASTROKE small-vessel'),
                     ('Cardioembolic', 'MEGASTROKE cardioembolic')]:
        r = w.loc[key]
        rows.append(dict(label=lab, OR=r.OR, lo=r.lo, hi=r.hi, col=B, marker='D', filled=r.p < 0.05, src='M5'))
    for r in rows:
        if 'header' in r:
            r['hx'] = -(41 - 1.0) / 37
            r['hc'] = COL['ink']
    ax = c.ax(41, 10, 37, 68)
    _forest(c, ax, rows, (0.5, 3.0), [0.5, 1, 2], 41, num_x=(79.0 - 41) / 37, f_x=(103.5 - 41) / 37)
    ax.set_xlabel('OR (or HR) per SD (95% CI)', labelpad=1)
    c.legend(legend_handles([('pt', '95% CI excludes 1', COL['base'], True), ('pt', 'includes 1', COL['base'], False),
                             ('pt', 'SBP-conditioned', COL['bp'], True)]), 1, 0.3, ncol=3, fs=5.2, columnspacing=0.6)
    c.text(107, 1.2, '$F_{cond}$ < 10: weak conditional instruments', fontsize=5.0, ha='right', color=COL['bp'])
    c.text(107, 4.4, f'MR-Egger intercept P = {s3.egger_intercept_p:.2f}', fontsize=5.0, ha='right', color=COL['mid'])
    sd = pd.DataFrame([{k: v for k, v in r.items() if k not in ('col', 'lc', 'hx', 'hc', 'note_col')} for r in rows])
    c.sd(sd, f'WMH → ischaemic stroke; primary grade {s3.grade} retained (MR-Egger SE {s3.se_Egger:.2f}, '
             f'intercept P = {s3.egger_intercept_p:.2f}; I2GX = {s3.I2GX:.2f}); Q P = {s3.Q_p:.1e}')


# ------------------------------------------------------------------ b  SNP-level scatter
@panel(F, 'b', 75, 86, 'Instrument-level agreement')
def b(c):
    h = pd.read_csv(P['S'] / 'S3_harmonised' / 'WMH__Stroke.csv')
    h = h[h.mr_keep.astype(bool)].copy()
    s3 = one(t('S', 'S3_univariable_summary'), exposure='WMH', outcome='Stroke')
    sgn = np.sign(h['beta.exposure'])
    h['bx'] = h['beta.exposure'] * sgn; h['by'] = h['beta.outcome'] * sgn
    ax = c.ax(12, 11, 60, 66)
    ax.errorbar(h.bx, h.by, xerr=1.96 * h['se.exposure'], yerr=1.96 * h['se.outcome'], fmt='none',
                ecolor=COL['ctrl'], elinewidth=0.4, zorder=2)
    ax.scatter(h.bx, h.by, s=9, c=COL['brain'], edgecolors='white', lw=0.3, zorder=3)
    xx = np.array([0, h.bx.max() * 1.08])
    ax.plot(xx, s3.b_IVW * xx, color=COL['brain'], lw=1.0, label=f'IVW, OR {np.exp(s3.b_IVW):.2f}')
    ax.plot(xx, s3.b_WM * xx, color=COL['ink'], lw=0.7, ls=(0, (3, 1.5)), label=f'Weighted median, {np.exp(s3.b_WM):.2f}')
    ax.plot(xx, s3.egger_intercept + s3.b_Egger * xx, color=COL['mid'], lw=0.8, ls=(0, (1, 1)),
            label=f'MR-Egger, {np.exp(s3.b_Egger):.2f} (imprecise)')
    refline(ax, 0, orient='h')
    ax.set_xlim(0, xx[1]); ax.set_xlabel('SNP effect on WMH (SD)'); ax.set_ylabel('SNP effect on ischaemic stroke (log OR)')
    ax.legend(loc='upper left', fontsize=5.3, handlelength=2.2, borderaxespad=0.3)
    ax.text(0.98, 0.04, f'{len(h)} SNPs; Cochran Q {fmt_p(s3.Q_p)}, I² = {s3.I2 * 100:.0f}%\n'
                        f'Egger intercept {fmt_p(s3.egger_intercept_p)}; I²$_{{GX}}$ = {s3.I2GX:.2f}\n'
                        f'Steiger direction correct; leave-one-out OR {np.exp(s3.loo_min_b):.2f}–{np.exp(s3.loo_max_b):.2f}',
            transform=ax.transAxes, fontsize=5.1, ha='right', va='bottom', linespacing=1.25)
    c.sd(h[['rsid', 'chr', 'pos', 'ea', 'oa', 'beta.exposure', 'se.exposure', 'beta.outcome', 'se.outcome', 'bx', 'by']],
         'Harmonised SNP effects, oriented to the WMH-increasing allele (bx, by); lines = IVW, weighted median, MR-Egger')


# ------------------------------------------------------------------ c  specificity after conditioning on SBP
@panel(F, 'c', 58, 64, 'Outcome specificity after SBP conditioning')
def cpanel(c):
    g6 = t('G', 'mvmr_SBP/G6_SBP_conditioned_MVMR') if (P['G'] / 'mvmr_SBP/G6_SBP_conditioned_MVMR.csv').exists() else None
    b5 = one(t('BP', 'B05_same_instrument_comparison_without_BP'), model='WMH+SBP__Stroke', exposure='WMH')
    b2 = one(sel(t('BP', 'B02_all_model_estimates'), method='IVW'), model='WMH+SBP__Stroke', exposure='WMH')
    rows = []
    o0 = _orci(b5.beta, b5.se)
    rows.append(dict(outcome='Ischaemic stroke', OR0=o0[0], lo0=o0[1], hi0=o0[2], OR1=np.exp(b2.beta), lo1=np.exp(b2.lo),
                     hi1=np.exp(b2.hi), p1=b2.p, F=b2.F_conditional_standard, nsnp=b2.nsnp, src='B05/B02'))
    for k in ['HF_HERMES', 'AF_Nielsen', 'CAD_Nikpay']:
        s = sel(g6, outcome=k, exposure='WMH')
        r0 = one(s, method='IVW_same_instruments_omit_SBP'); r1 = one(s, method='IVW_MVMR')
        rows.append(dict(outcome=OUT_EN[k], OR0=r0.OR, lo0=r0.OR_lo, hi0=r0.OR_hi, OR1=r1.OR, lo1=r1.OR_lo, hi1=r1.OR_hi,
                         p1=r1.p, F=r1.F_conditional, nsnp=r1.nsnp, src='G6'))
    D = pd.DataFrame(rows)
    ax = c.ax(22.5, 15, 27, 39)
    for i, r in D.iterrows():
        ax.annotate('', (r.OR1, i + 0.17), (r.OR0, i - 0.17), arrowprops=dict(arrowstyle='-|>,head_length=0.25,head_width=0.12',
                    lw=0.45, color=COL['mid'], shrinkA=1.6, shrinkB=1.6), zorder=1)
        ci_point(ax, r.OR0, i - 0.17, r.lo0, r.hi0, COL['brain'], filled=(r.lo0 > 1 or r.hi0 < 1), ms=2.8, lw=0.7)
        ci_point(ax, r.OR1, i + 0.17, r.lo1, r.hi1, COL['bp'], filled=(r.lo1 > 1 or r.hi1 < 1), ms=2.8, lw=0.7)
        ax.text(1.14, i, f'{r.F:.1f}', transform=ax.get_yaxis_transform(), fontsize=5.2, ha='center', va='center',
                color=COL['bp'] if r.F < 10 else COL['ink'])
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
    ax.text(1.14, -0.85, '$F_{cond}$', transform=ax.get_yaxis_transform(), fontsize=5.3, ha='center', va='center')
    ax.set_ylim(len(D) - 0.5, -0.6)
    clean_y(ax, [{'Coronary artery disease': 'Coronary artery dis.'}.get(o_, o_) for o_ in D.outcome], fs=5.3,
            color=[COL['brain'], COL['heart'], COL['heart'], COL['heart']])
    refline(ax, 1); log_x(ax, (0.9, 1.5), [0.9, 1, 1.2, 1.4])
    ax.set_xlabel('OR per SD genetic WMH', labelpad=1)
    c.legend(legend_handles([('pt', 'Same SNPs, SBP not modelled', COL['brain'], True),
                             ('pt', '+ SBP (IVW-MVMR)', COL['bp'], True), ('pt', 'Open: 95% CI includes 1', COL['base'], False)]),
             1, 0.2, ncol=1, fs=5.0, labelspacing=0.15)
    c.sd(D, 'WMH effects with the same final instruments, without vs with SBP; conditional F < 10 flags weak conditional '
            'instruments; filled = 95% CI excludes 1')


# ------------------------------------------------------------------ d  genetic cardiac axis
@panel(F, 'd', 74, 64, 'Genetic cardiac axis: positive control failed')
def d(c):
    g = t('G', 'G1_G4_univariable_summary'); m5 = t('M5', 'M5_forest_source_table')
    s4 = m5[m5.panel == 'CardiacAxisS4'].assign(label=lambda d_: d_.label.str.strip()).set_index('label')
    H = COL['heart']
    rows = [dict(header='Positive control (G1): not passed')]
    for k in ['HF_HERMES', 'AF_Nielsen', 'CAD_Nikpay']:
        r = one(g, exposure='CardiacAxisS4', outcome=k)
        rows.append(dict(label=OUT_EN[k], OR=r.OR_IVW, lo=r.OR_lo, hi=r.OR_hi, col=H, filled=r.FDR_family < 0.05,
                         src='G1', nsnp=r.nsnp, Q_p=r.Q_p, pc=True))
    rows.append(dict(header='Ischaemic stroke (inconclusive)'))
    for lab, key, mk in [('Trans-ancestry IS, IVW (14 SNPs)', 'Primary: UKB + FinnGen (IVW, 14 SNPs)', 'o'),
                         ('| WMH (MVMR)', 'MVMR | WMH', 'o'), ('| WMH + SBP (Q-het robust)', 'MVMR | WMH + SBP (Q-het robust)', 'o'),
                         ('MEGASTROKE, any ischaemic', 'MEGASTROKE ischaemic (independent)', 'D'),
                         ('MEGASTROKE, cardioembolic', 'MEGASTROKE cardioembolic', 'D')]:
        r = s4.loc[key]
        rows.append(dict(label=lab, OR=r.OR, lo=r.lo, hi=r.hi, col=H, marker=mk, filled=r.p < 0.05, src='M5'))
    rows.append(dict(header='Joint 446-SNP set incl. BP loci → stroke'))
    for lab, key, col in [('SBP not modelled', 'Joint 446 IVs incl. BP loci, BP not modelled', H),
                          ('+ SBP (IVW-MVMR)', 'same 446 IVs + SBP (IVW)', COL['bp'])]:
        r = s4.loc[key]
        rows.append(dict(label=lab, OR=r.OR, lo=r.lo, hi=r.hi, col=col, filled=r.p < 0.05, src='M5/B05/B02'))
    for r in rows:
        if 'header' in r:
            r['hx'] = -(30.5 - 1.0) / 22; r['hc'] = COL['heart'] if 'not passed' in r['header'] else COL['ink']
    ax = c.ax(30.5, 12, 22, 45)
    v = json.load(open(GDIR / 'G5_verdicts.json'))
    if not v['G1_pass']:
        ax.axhspan(0.5, 3.5, color=COL['heart_l'], alpha=0.35, lw=0, zorder=0)
    _forest(c, ax, rows, (0.4, 2.5), [0.5, 1, 2], 30.5, fcond=False, fs=5.2, num_x=(54.0 - 30.5) / 22)
    ax.set_xlabel('OR per SD genetic cardiac axis', labelpad=1)
    c.legend(legend_handles([('mk', 'Primary IS GWAS / G1', COL['heart'], 'o', True),
                             ('mk', 'MEGASTROKE', COL['heart'], 'D', True), ('pt', 'Open: 95% CI includes 1', COL['base'], False)]),
             1, 0.2, ncol=3, fs=5.0, columnspacing=0.5, handletextpad=0.2)
    sd = pd.DataFrame([{k: v_ for k, v_ in r.items() if k not in ('col', 'lc', 'hx', 'hc')} for r in rows])
    sd['G1_pass'] = v['G1_pass']; sd['G1_text'] = v['G1_text']
    c.sd(sd, 'Genetic cardiac axis (S4) MR; observational HR not shown on this panel (see Fig. 4)')


# ------------------------------------------------------------------ e  disease-level bidirectional map
EXP = [('CardiacAxisS4', 'Genetic cardiac axis', COL['heart']), ('WMH', 'WMH', COL['brain']),
       ('FA_genu', 'FA (genu)', COL['brain']), ('ISOVF_genu', 'ISOVF (genu)', COL['brain'])]
OUTS = ['HF_HERMES', 'AF_Nielsen', 'CAD_Nikpay', 'Stroke', 'AD_Kunkle', 'PD_Nalls']


@panel(F, 'e', 51, 64, 'Disease-level genetic map')
def e(c):
    g = t('G', 'G1_G4_univariable_summary'); s3 = t('S', 'S3_univariable_summary'); s9 = t('S', 'S9_axis_univariable_summary')
    rec = []
    for x, _, _ in EXP:
        for o in OUTS:
            if o == 'Stroke':
                src = s9 if x == 'CardiacAxisS4' else s3
                r = one(src, exposure=x, outcome='Stroke'); b, se, p = r.b_IVW, r.se_IVW, r.p_IVW
                fdr = r['FDR_family'] if 'FDR_family' in r.index else np.nan
                fam = 'S9' if x == 'CardiacAxisS4' else 'S3'
            else:
                r = one(g, exposure=x, outcome=o); b, se, p, fdr, fam = r.b_IVW, r.se_IVW, r.p_IVW, r.FDR_family, r.family
            OR, lo, hi = _orci(b, se)
            rec.append(dict(exposure=x, outcome=o, family=fam, nsnp=r.nsnp, b_IVW=b, se_IVW=se, OR=OR, lo=lo, hi=hi, p=p, FDR=fdr))
    D = pd.DataFrame(rec)
    Bm = D.pivot(index='exposure', columns='outcome', values='b_IVW').loc[[e_[0] for e_ in EXP], OUTS]
    Om = D.pivot(index='exposure', columns='outcome', values='OR').loc[Bm.index, OUTS]
    Qm = D.pivot(index='exposure', columns='outcome', values='FDR').loc[Bm.index, OUTS]
    Pm = D.pivot(index='exposure', columns='outcome', values='p').loc[Bm.index, OUTS]
    ax = c.ax(19.5, 25, 30.5, 22)
    vmax = np.log(1.5); norm = Normalize(-vmax, vmax)
    ax.imshow(np.clip(Bm.values, -vmax, vmax), cmap=CMAP_DIV, norm=norm, aspect='auto', interpolation='none')
    for i in range(Bm.shape[0]):
        for j in range(Bm.shape[1]):
            sig = Qm.values[i, j] < 0.05 if np.isfinite(Qm.values[i, j]) else False
            ax.text(j, i, f'{Om.values[i, j]:.2f}', ha='center', va='center', fontsize=5.0,
                    fontweight='bold' if sig else 'normal', color='white' if abs(Bm.values[i, j]) > 0.6 * vmax else COL['ink'])
            if not np.isfinite(Qm.values[i, j]) and Pm.values[i, j] < 0.05:
                ax.text(j + 0.38, i - 0.3, '†', fontsize=5, ha='center', va='center')
    OSH = {'HF_HERMES': 'HF', 'AF_Nielsen': 'AF', 'CAD_Nikpay': 'CAD', 'Stroke': 'IS', 'AD_Kunkle': 'AD', 'PD_Nalls': 'PD'}
    ax.set_xticks(range(len(OUTS))); ax.set_xticklabels([OSH[o] for o in OUTS], fontsize=5.3)
    for tl, o in zip(ax.get_xticklabels(), OUTS):
        tl.set_color(COL['heart'] if o in ('HF_HERMES', 'AF_Nielsen', 'CAD_Nikpay') else COL['brain'])
    ax.set_yticks(range(len(EXP))); ax.set_yticklabels(['Genetic cardiac\naxis' if e_[0] == 'CardiacAxisS4' else e_[1]
                                                        for e_ in EXP], fontsize=5.2, linespacing=0.95)
    for tl, e_ in zip(ax.get_yticklabels(), EXP):
        tl.set_color(e_[2])
    ax.tick_params(length=0, pad=1)
    ax.set_xticks(np.arange(-.5, len(OUTS)), minor=True); ax.set_yticks(np.arange(-.5, len(EXP)), minor=True)
    ax.grid(which='minor', color='white', lw=0.8); ax.tick_params(which='minor', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(Rectangle((-0.5, -0.5), 3, 1, fill=False, ec=COL['ink'], lw=0.8, zorder=4))
    gb = c.ax(19.5, 48.2, 30.5, 1.4); gb.set_xlim(-0.5, 5.5); gb.set_ylim(0, 1); gb.axis('off')
    gb.add_patch(Rectangle((-0.45, 0), 2.9, 1, fc=COL['heart'], ec='none'))
    gb.add_patch(Rectangle((2.55, 0), 2.9, 1, fc=COL['brain'], ec='none'))
    gb.text(1.0, 1.6, 'Cardiac', ha='center', va='bottom', fontsize=5.3, color=COL['heart'])
    gb.text(4.0, 1.6, 'Brain', ha='center', va='bottom', fontsize=5.3, color=COL['brain'])
    sm = __import__('matplotlib.cm', fromlist=['ScalarMappable']).ScalarMappable(norm=norm, cmap=CMAP_DIV)
    heat_colorbar(c, sm, 25, 16.5, 20, 1.3, '', ticks=np.log([0.67, 1, 1.5]), ticklabels=['0.67', '1', '1.5'])
    c.text(24, 17.2, 'OR', fontsize=5.4, ha='right', va='center')
    c.text(1, 0.4, 'Bold: FDR < 0.05 within family\nBox: positive control (G1)\nHF, heart failure; AF, atrial fibrillation;\n'
                   'CAD, coronary artery disease; IS, ischaemic\nstroke; AD, Alzheimer; PD, Parkinson',
           fontsize=5.0, ha='left', va='bottom', color=COL['mid'], linespacing=1.1)
    c.sd(D, 'Disease-level IVW MR; FDR within pre-specified family (G1–G4, S3); S9 has no family FDR. FA per SD higher')


# ------------------------------------------------------------------ f  triangulation matrix
CODE = {'S': ('Support', 'o', True, COL['ink']), 'I': ('Inconclusive / imprecise', 'o', 'half', COL['ink']),
        'N': ('Null', 'o', False, COL['ink']), 'X': ('Against', 'X', True, COL['ink']),
        'C': ('Discordant', 'D', True, COL['ink']), 'NA': ('Not available', '.', True, COL['ctrl']),
        'PF': ('Positive control failed', 's', False, COL['ink'])}
LINK_EN = {0: 'Cardiac remodelling → white matter', 1: 'White matter → heart (reverse)', 2: 'Shared genetic architecture',
           3: 'Inflammation (CRP) → heart / WM', 4: 'WMH → ischaemic stroke', 5: 'Cardiac remodelling → ischaemic stroke'}


@panel(F, 'f', 183, 58, 'Triangulation across lines of evidence')
def f(c):
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    tm = t('M5', 'Triangulation_matrix_updated')
    v = json.load(open(GDIR / 'G5_verdicts.json'))
    ev_cols = [k for k in tm.columns if k not in ('link', 'Verdict')]
    ins = ev_cols.index('Genetic cardiac axis MR (S4)') + 1
    cols = ev_cols[:ins] + ['Positive control (G1)'] + ev_cols[ins:]
    rec = []
    AX0, AXW, AXB, AXH = 50, 92, 9, 37
    ax = c.ax(AX0, AXB, AXW, AXH)
    n = len(tm)
    for i, r in tm.iterrows():
        uses_s4 = 'Cardiac remodelling' in r.link
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
        for j, k in enumerate(cols):
            if k == 'Positive control (G1)':
                cell = ('PF: ' + v['G1_text']) if (uses_s4 and not v['G1_pass']) else 'NA: —'
            else:
                cell = str(r[k])
            code = cell.split(':')[0].strip()
            lab, mk, fill, col = CODE[code]
            if fill == 'half':
                ax.plot([j], [i], marker='o', ms=5.4, mfc=COL['ctrl_l'], mec=col, mew=0.7, ls='none')
                ax.plot([j], [i], marker=MarkerStyle('o', fillstyle='left'), ms=5.4, mfc=col, mec=col, mew=0.7, ls='none')
            elif mk == '.':
                ax.plot([j], [i], marker='o', ms=1.6, mfc=col, mec=col, ls='none')
            else:
                ax.plot([j], [i], marker=mk, ms=5.4 if mk != 'X' else 5.6, mfc=col if fill else 'white', mec=col,
                        mew=0.7, ls='none')
            nt = ''
            if k == 'LDSC rg' and 'stroke' in r.link:
                _e4 = pd.read_csv(HB_ROOT / 'manuscript/supp/ancestry_check/results/E4_LDSC_rg_IS_EUR.csv')
                _t = 'WMH' if r.link.startswith('WMH') else 'CardiacAxisS4'
                _q = _e4[_e4.trait1 == _t].iloc[0]
                nt = (f'Cell reflects the frozen trans-ancestry stroke GWAS (G07; rg {_q.rg_transancestry_B01:.2f}, '
                      f's.e. {_q.rg_se_transancestry_B01:.2f}). Post hoc ancestry-matched European-only outcome (G08; '
                      f'Supplementary Fig. 2f, Supplementary Data 2): rg {_q.rg:.2f} (s.e. {_q.rg_se:.2f}), P = {_q.p_rg:.2g}; '
                      f'stroke h2 Z = {_q.h2_2_z:.2f}.' + (' The main text reports the ancestry-matched value (rule R5).' if _t == 'WMH' else ' Not reported in the main text.'))
            rec.append(dict(link=r.link, link_en=LINK_EN[i], evidence=k, cell=cell, code=code, note=nt))
        verdict = str(r.Verdict).split(':', 1)
        vtxt = verdict[1].strip() if len(verdict) > 1 else verdict[0]
        vcode = verdict[0].strip()
        note = ''
        if uses_s4 and not v['G1_pass'] and vcode == 'C':
            vtxt, vcode, note = 'Inconclusive (positive control failed)', 'I', 'updated per framework v4 E18 after G1'
        rec.append(dict(link=r.link, link_en=LINK_EN[i], evidence='Verdict', cell=str(r.Verdict), code=vcode,
                        verdict_displayed=vtxt, note=note))
        # verdict badge (board coordinates)
        yc = AXB + AXH * (1 - (i + 0.5) / n)
        main, _, sub = vtxt.partition(' (')
        sub = sub.rstrip(')')
        if vcode == 'S':
            main, sub = 'Supports a causal role', sub.replace('grade D retained', 'MR grade D retained')
        fc, ec, tc = {'S': (COL['brain'], COL['brain'], 'white'), 'I': (COL['ctrl_l'], COL['ctrl'], COL['ink']),
                      'X': ('white', COL['ink'], COL['ink']), 'N': ('white', COL['ink'], COL['ink'])}.get(
            vcode, ('white', COL['ink'], COL['ink']))
        bd = c.board()
        rounded_box(bd, 145, yc - 2.3, 18.5, 4.6, fc=fc, ec=ec, lw=0.6, r=1.2, z=3)
        bd.text(145 + 9.25, yc, main if vcode != 'S' else 'Supports', fontsize=5.4, ha='center', va='center', color=tc,
                fontweight='bold', zorder=4)
        SUB = {'> 0.04 SD excluded': '> 0.04 SD\nexcluded', 'MR grade D retained': 'causal role;\nMR grade D retained',
               'positive control failed': 'positive control\nfailed'}
        if sub:
            bd.text(165.0, yc, SUB.get(sub, sub), fontsize=5.0, ha='left', va='center', color=COL['mid'],
                    linespacing=1.05, zorder=4)
    ax.set_xlim(-0.55, len(cols) - 0.45); ax.set_ylim(n - 0.5, -0.5)
    short = {'Observational (two cohorts)': 'Observational\n(two cohorts)', 'Single-trait MR': 'Single-trait\nMR',
             'Genetic cardiac axis MR (S4)': 'Genetic\ncardiac axis', 'Positive control (G1)': 'Positive\ncontrol (G1)',
             'MVMR (heart / WMH)': 'MVMR\nheart / WMH', 'BP-conditioned MVMR': 'SBP-\nconditioned', 'LDSC rg': 'LDSC\n$r_g$',
             'Independent replication': 'Independent\nreplication'}
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([short[k] for k in cols], fontsize=5.3, linespacing=1.0)
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1.5)
    ax.set_yticks(range(n)); ax.set_yticklabels([LINK_EN[i] for i in range(n)], fontsize=5.6)
    for tl, i in zip(ax.get_yticklabels(), range(n)):
        tl.set_color(COL['brain'] if i == 4 else COL['ink']); tl.set_fontweight('bold' if i == 4 else 'normal')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(3.5, color=COL['ink'], lw=0.5)
    for j in range(1, len(cols)):
        ax.axvline(j - 0.5, color='white', lw=0.8, zorder=0.5)
    # question grouping bars
    bd = c.board()
    for (r0, r1, lab, col) in [(0, 3, 'Heart ↔ brain', COL['joint']), (4, 5, 'Stroke', COL['brain'])]:
        y_top = AXB + AXH * (1 - r0 / n) - 0.6; y_bot = AXB + AXH * (1 - (r1 + 1) / n) + 0.6
        bd.plot([2.2, 2.2], [y_bot, y_top], color=col, lw=1.6, solid_capstyle='butt')
        bd.text(1.0, (y_top + y_bot) / 2, lab, rotation=90, fontsize=5.3, color=col, ha='center', va='center',
                fontweight='bold')
    c.text(145, AXB + AXH + 1.2, 'Verdict', fontsize=6.0, fontweight='bold', va='bottom')
    hs = []
    for k in ['S', 'I', 'N', 'X', 'PF', 'NA']:
        lab, mk, fill, col = CODE[k]
        if fill == 'half':
            hs.append(Line2D([], [], marker=MarkerStyle('o', fillstyle='left'), ms=4.4, mfc=col, mec=col,
                             markerfacecoloralt=COL['ctrl_l'], ls='none', label=lab))
        elif mk == '.':
            hs.append(Line2D([], [], marker='o', ms=1.6, mfc=col, mec=col, ls='none', label=lab))
        else:
            hs.append(Line2D([], [], marker=mk, ms=4.4, mfc=col if fill else 'white', mec=col, mew=0.7, ls='none', label=lab))
    c.legend(hs, 50, 0.3, ncol=6, fs=5.2, columnspacing=1.0, handletextpad=0.3)
    c.sd(pd.DataFrame(rec), 'Triangulation matrix (M5 updated) with the G1 positive-control column added; the cardiac '
                            'remodelling → stroke verdict is shown as inconclusive per framework v4 (E18)')


LAYOUT = [(F, 'a', 0, 0), (F, 'b', 108, 0), (F, 'c', 0, 86), (F, 'd', 58, 86), (F, 'e', 132, 86), (F, 'f', 0, 150)]
HEIGHT = 208
