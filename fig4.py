"""Submission Fig. 3 (in-house Fig4) — v4: disease consequences of cardiac and brain structure are asymmetric.

Panels: a  30 outcomes x 5 exposures (heat map, events, power tier)
        b  exposure organ x outcome organ quadrants (per-outcome HRs, negative-control range, NC-referenced ratios)
        c  vascular gradient within brain disease
        d  cross-organ associations on a common HR scale (main vs excluding prior disease + other axis)
Moved out in v4: Cox vs Fine-Gray -> Supplementary Fig. 3a; mutual adjustment -> Fig. 4e (in-house Fig5).
Every number is read from frozen result files (nc_data.t); NC-referenced ratios are deterministic arithmetic on B1
(log HR difference, variances added, independence assumed — the post hoc method stated in the text).
"""
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Rectangle
from nclib import (panel, COL, CMAP_DIV, ci_point, refline, clean_y, legend_handles, heat_colorbar, log_x, rounded_box)
from nc_data import t, sel, one, OUT_ORDER, OUT_SHORT, EXPO, EXPO_LAB, FAMILY_EN, outcome_dict

F = 'Fig4'
ORG_COL = {'心': COL['heart'], '心脑': COL['joint'], '脑': COL['brain'], '对照': COL['ctrl']}
TIER = {'充分功效（MDE ≤ 1.30）': ('Adequate (MDE ≤ 1.30)', '#4A4E57'),
        '有限功效（1.30–1.50）': ('Limited (MDE 1.30–1.50)', '#9AA0A8'),
        '仅描述（MDE > 1.50）': ('Descriptive (MDE > 1.50)', '#D5D9DE')}
EXP_COL = {'CardiacAxis': COL['heart'], 'BrainAxis': COL['joint'], 'logWMH': COL['brain'], 'FA_r': COL['brain'],
           'GMVn_r': COL['brain']}
NC = ['NC_inguinal_hernia', 'NC_senile_cataract']


def _nc_ratio(m, x, o, nc='NC_senile_cataract'):
    a, b = one(m, outcome=o, exposure=x), one(m, outcome=nc, exposure=x)
    d = a.logHR - b.logHR; se = float(np.hypot(a.se, b.se))
    return dict(exposure=x, outcome=o, reference=nc, HR_outcome=a.HR, HR_reference=b.HR, ratio=np.exp(d),
                lo=np.exp(d - 1.96 * se), hi=np.exp(d + 1.96 * se), log_ratio=d, se=se,
                method='post hoc; log HR difference, variances added (independence assumed)')


# ------------------------------------------------------------------ a  full matrix
@panel(F, 'a', 100, 132, '30 pre-specified outcomes × 5 imaging exposures')
def a(c):
    m = t('B', 'B1_full_matrix_cox')
    od = outcome_dict().set_index('key')
    tiers = t('A', 'A5_power_feasibility_tiers').set_index('key')['feasibility_tier']
    ys, y, prev = {}, 0.0, None
    for k in OUT_ORDER:
        fam = od.loc[k, '疾病家族']
        if prev is not None and fam != prev:
            y += 0.45
        ys[k] = y; y += 1; prev = fam
    ymax = y
    L, B_, W, H = 31, 10, 40, 104
    ax = c.ax(L, B_, W, H)
    vmax = np.log(2.5); norm = Normalize(-vmax, vmax)
    for j, x in enumerate(EXPO):
        for k in OUT_ORDER:
            r = sel(m, outcome=k, exposure=x); yk = ys[k]
            if len(r) == 0:
                ax.add_patch(Rectangle((j - 0.5, yk - 0.5), 1, 1, fc='white', ec=COL['grid'], lw=0.3, hatch='/////', zorder=1))
                continue
            r = r.iloc[0]
            ax.add_patch(Rectangle((j - 0.5, yk - 0.5), 1, 1, fc=CMAP_DIV(norm(np.clip(r.logHR, -vmax, vmax))), ec='white',
                                   lw=0.6, zorder=1))
            sig = np.isfinite(r.p_adj) and r.p_adj < 0.05
            tc = 'white' if abs(r.logHR) > 0.55 * vmax else COL['ink']
            ax.text(j, yk, f'{r.HR:.2f}', ha='center', va='center', fontsize=5.0, color=tc,
                    fontweight='bold' if sig else 'normal', zorder=3)
    ax.set_xlim(-0.5, len(EXPO) - 0.5); ax.set_ylim(ymax - 0.5, -0.5)
    ax.set_xticks(range(len(EXPO)))
    ax.set_xticklabels([EXPO_LAB[x] for x in EXPO], rotation=40, ha='left', rotation_mode='anchor', fontsize=5.8)
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1)
    for tl, x in zip(ax.get_xticklabels(), EXPO):
        tl.set_color(EXP_COL[x])
    ax.set_yticks([ys[k] for k in OUT_ORDER]); ax.set_yticklabels([OUT_SHORT[k] for k in OUT_ORDER], fontsize=5.5)
    for tl, k in zip(ax.get_yticklabels(), OUT_ORDER):
        tl.set_color(ORG_COL[od.loc[k, '器官']])
    for s in ax.spines.values():
        s.set_visible(False)
    st = c.ax(0.6, B_, 1.2, H); st.set_ylim(ymax - 0.5, -0.5); st.set_xlim(0, 1); st.axis('off')
    for k in OUT_ORDER:
        st.add_patch(Rectangle((0, ys[k] - 0.5), 1, 1, fc=ORG_COL[od.loc[k, '器官']], ec='none'))
    ev = c.ax(L + W + 1.5, B_, 11.5, H); ev.set_ylim(ymax - 0.5, -0.5)
    rows = []
    for k in OUT_ORDER:
        r = od.loc[k]; tier = tiers.get(k)
        ev.barh(ys[k], r.events, height=0.72, color=TIER[tier][1], lw=0)
        rows.append(dict(outcome=k, label=OUT_SHORT[k], organ=r['器官'], family=FAMILY_EN[r['疾病家族']], events=r.events,
                         n_at_risk=r.n_at_risk, person_years=r.person_years, power_tier=tier,
                         modelled=len(sel(m, outcome=k)) > 0))
    ev.set_xscale('log'); ev.set_xlim(5, 1500)
    ev.set_xticks([10, 100, 1000]); ev.set_xticklabels(['10', '100', '1,000'], fontsize=5.3)
    ev.xaxis.set_minor_locator(__import__('matplotlib.ticker', fromlist=['NullLocator']).NullLocator())
    ev.set_yticks([]); ev.spines['left'].set_visible(False)
    c.text(L + W + 7.2, B_ + H + 1.2, 'Events', fontsize=5.8, ha='center', va='bottom')
    fb = c.ax(L + W + 13.6, B_, 15.4, H); fb.set_ylim(ymax - 0.5, -0.5); fb.set_xlim(0, 15.4); fb.axis('off')
    seen = set()
    for k in OUT_ORDER:
        fam = FAMILY_EN[od.loc[k, '疾病家族']]; organ = od.loc[k, '器官']
        key = (fam, organ if fam == 'Composite' else '')
        if key in seen:
            continue
        seen.add(key)
        yy = [ys[q] for q in OUT_ORDER if FAMILY_EN[od.loc[q, '疾病家族']] == fam and
              (fam != 'Composite' or od.loc[q, '器官'] == organ)]
        fb.plot([0.6, 0.6], [min(yy) - 0.35, max(yy) + 0.35], color=COL['mid'], lw=0.5)
        fb.text(1.3, np.mean(yy), {'Neurodegenerative': 'Neuro-\ndegenerative', 'Other neurological': 'Other\nneurological',
                                   'Cerebrovascular': 'Cerebro-\nvascular', 'Negative control': 'Negative\ncontrol'}.get(fam, fam),
                fontsize=5.0, color=COL['mid'], va='center', linespacing=0.95)
    sd = m.copy(); sd['label'] = sd.outcome.map(OUT_SHORT)
    c.sd(sd[['outcome', 'label', 'exposure', 'exposure_tier', 'n', 'events', 'HR', 'lo', 'hi', 'p', 'p_adj', 'adj_method']],
         'Cox HR per SD (higher cardiac axis / brain ageing axis / WMH; lower FA / GMV), fully adjusted')
    c.sd(pd.DataFrame(rows), 'Events during follow-up and pre-specified power tier (A5, fixed before association analysis)')
    sm = __import__('matplotlib.cm', fromlist=['ScalarMappable']).ScalarMappable(norm=norm, cmap=CMAP_DIV)
    heat_colorbar(c, sm, 33, 3.6, 20, 1.4, '', ticks=np.log([0.5, 1, 2]), ticklabels=['0.5', '1', '2'])
    c.text(32, 4.3, 'HR', fontsize=5.8, ha='right', va='center')
    hs = [Rectangle((0, 0), 1, 1, fc=v[1], ec='none', label=v[0].split(' (')[0]) for v in TIER.values()]
    hs.append(Rectangle((0, 0), 1, 1, fc='white', ec=COL['mid'], lw=0.3, hatch='/////', label='< 20 events'))
    c.legend(hs, 57, 0.6, ncol=2, fs=5.0, handlelength=1.0, columnspacing=0.6, handletextpad=0.3, labelspacing=0.2,
             title='Power tier (events bar)', title_fontsize=5.0)
    c.text(1, 1.0, 'Bold: FDR < 0.05\n(Holm for co-primary)', fontsize=5.0, color=COL['mid'], va='bottom', linespacing=1.1)


# ------------------------------------------------------------------ b  organ-to-organ quadrants
def _swarm(xmm, d=1.25, step=1.15, levels=(0, 1, -1, 2, -2, 3, -3)):
    order = np.argsort(xmm); placed = []; out = np.zeros(len(xmm))
    for i in order:
        for lv in levels:
            yy = lv * step
            if all((xmm[i] - px) ** 2 + (yy - py) ** 2 >= d ** 2 for px, py in placed):
                break
        placed.append((xmm[i], yy)); out[i] = yy
    return out


@panel(F, 'b', 83, 74, 'Cross-organ associations are asymmetric')
def b(c):
    m = t('B', 'B1_full_matrix_cox'); od = outcome_dict().set_index('key')
    rows_ = [('CardiacAxis', 'Cardiac\naxis', COL['heart']), ('BrainAxis', 'Brain\nageing axis', COL['joint'])]
    cols_ = [('心', 'Cardiac diseases', COL['heart']), ('脑', 'Brain diseases', COL['brain'])]
    XL = (0.6, 4.5); cw, ch = 31.0, 15.5; x0s = [16.5, 50.5]; y0s = [45.5, 27.0]
    lx = lambda v: (np.log(v) - np.log(XL[0])) / (np.log(XL[1]) - np.log(XL[0])) * cw
    rec = []
    LABEL = {('CardiacAxis', '脑'): (['Stroke', 'Ischaemic_stroke'], 'Stroke, IS'),
             ('CardiacAxis', '心'): (['Cardiomyopathy'], 'Cardiomyopathy'),
             ('BrainAxis', '脑'): (['Dementia'], 'Dementia')}
    for i, (x, xl, xc) in enumerate(rows_):
        c.text(0.8, y0s[i] + ch / 2, xl, fontsize=5.8, color=xc, fontweight='bold', va='center', ha='left', linespacing=1.0)
        ncs = sel(m, exposure=x, outcome=NC)
        for j, (o, ol, oc) in enumerate(cols_):
            ax = c.ax(x0s[j], y0s[i], cw, ch)
            within = (x == 'CardiacAxis' and o == '心') or (x == 'BrainAxis' and o == '脑')
            ax.set_facecolor(to_rgba(oc, 0.07) if within else 'white')
            s = m[(m.exposure == x) & m.outcome.map(lambda k: od.loc[k, '器官'] == o and od.loc[k, '疾病家族'] != '复合')].copy()
            ax.axvspan(lx(ncs.HR.min()), lx(ncs.HR.max()), color=COL['ctrl'], alpha=0.22, lw=0, zorder=0)
            ax.axvline(lx(1), color=COL['mid'], lw=0.4, ls=(0, (2.5, 1.5)), zorder=0)
            xm = lx(s.HR.values); yy = _swarm(xm)
            sig = (s.p_adj < 0.05).values
            for xi, yi, sg in zip(xm, yy, sig):
                ax.plot(xi, yi, 'o', ms=3.0, mfc=xc if sg else 'white', mec=xc, mew=0.7, zorder=3)
            med = float(np.exp(s.logHR.median()))
            ax.plot([lx(med)] * 2, [-ch / 2 + 2.0, ch / 2 - 3.6], color=COL['ink'], lw=1.1, zorder=2, solid_capstyle='butt')
            ax.text(0.6, ch / 2 - 0.5, f'{int(sig.sum())}/{len(s)} FDR < 0.05 · median HR {med:.2f}', fontsize=5.0,
                    va='top', ha='left', color=COL['ink'])
            if (x, o) in LABEL:
                ks, txt = LABEL[(x, o)]
                ii = [list(s.outcome).index(k) for k in ks]
                xa = float(np.mean(xm[ii])); ya = float(min(yy[ii]))
                ax.annotate(txt, (xa, ya - 0.7), xytext=(xa, -ch / 2 + 0.4), fontsize=4.9, color=COL['base'], ha='center',
                            va='bottom', zorder=4, arrowprops=dict(arrowstyle='-', lw=0.35, color=COL['mid'],
                                                                   shrinkA=0, shrinkB=0))
            ax.set_xlim(0, cw); ax.set_ylim(-ch / 2, ch / 2)
            ax.set_yticks([])
            tk = [0.75, 1, 1.5, 2, 3, 4]
            ax.set_xticks([lx(v) for v in tk]); ax.set_xticklabels([f'{v:g}' for v in tk] if i == 1 else [], fontsize=5.3)
            for sp in ['top', 'right', 'left']:
                ax.spines[sp].set_visible(False)
            if i == 0:
                c.text(x0s[j] + cw / 2, y0s[0] + ch + 1.3, ol, fontsize=5.9, color=oc, fontweight='bold', ha='center', va='bottom')
            for _, r in s.iterrows():
                rec.append(dict(exposure=x, outcome_organ={'心': 'heart', '脑': 'brain'}[o], within_organ=within,
                                outcome=r.outcome, label=OUT_SHORT[r.outcome], HR=r.HR, lo=r.lo, hi=r.hi, p_adj=r.p_adj,
                                FDR_sig=bool(r.p_adj < 0.05), quadrant_median_HR=med,
                                NC_min_HR=ncs.HR.min(), NC_max_HR=ncs.HR.max()))
    c.text(x0s[0] + cw + 1.7, y0s[1] - 4.6, 'HR per SD (log scale)', fontsize=5.8, ha='center', va='top')
    # negative-control-referenced ratios (post hoc, as reported in the text)
    R = [_nc_ratio(m, 'BrainAxis', 'P2_major_cardiac'), _nc_ratio(m, 'CardiacAxis', 'Stroke')]
    axr = c.ax(40, 2.6, 26, 9.0)
    for k, r in enumerate(R):
        col = COL['joint'] if r['exposure'] == 'BrainAxis' else COL['heart']
        ci_point(axr, r['ratio'], k, r['lo'], r['hi'], col, filled=r['lo'] > 1, ms=2.8, lw=0.8)
        axr.text(r['hi'] * 1.05, k, f"{r['ratio']:.2f} ({r['lo']:.2f}–{r['hi']:.2f})", fontsize=5.0, va='center')
    axr.set_ylim(1.6, -0.6); refline(axr, 1); log_x(axr, (0.8, 2.6), [1, 1.5, 2])
    axr.set_yticks([0, 1]); axr.set_yticklabels(['Brain ageing axis → P2', 'Cardiac axis → stroke'], fontsize=5.3)
    for tl, col in zip(axr.get_yticklabels(), [COL['joint'], COL['heart']]):
        tl.set_color(col)
    axr.tick_params(axis='y', length=0); axr.spines['left'].set_visible(False)
    axr.tick_params(axis='x', labelsize=5.2, pad=0.8)
    c.text(0.8, 14.0, 'HR ratio vs negative control (cataract), post hoc', fontsize=5.4, fontweight='bold', va='bottom')
    hs = legend_handles([('pt', 'FDR < 0.05', COL['base'], True), ('pt', 'FDR ≥ 0.05', COL['base'], False),
                         ('patch', 'Negative-control HR range', to_rgba(COL['ctrl'], 0.35))])
    from matplotlib.lines import Line2D
    hs.append(Line2D([], [], color=COL['ink'], lw=1.1, label='Median'))
    c.legend(hs, 16.5, 65.4, ncol=4, fs=4.9, columnspacing=0.6, handletextpad=0.25, handlelength=1.0)
    c.sd(pd.DataFrame(rec), 'Per-outcome Cox HRs by exposure organ × outcome organ (composites excluded); median of log HR; '
                            'grey band = range of the two negative-control HRs for the same exposure')
    c.sd(pd.DataFrame(R), 'Negative-control-referenced HR ratios (post hoc; reference = senile cataract)')


# ------------------------------------------------------------------ c  cross-organ forests on a common scale
@panel(F, 'd', 183, 62, 'Cross-organ associations persist after excluding prior disease and adjusting for the other organ')
def cpanel(c):
    b2 = t('B', 'B2_cross_organ_sensitivity'); m = t('B', 'B1_full_matrix_cox')
    blocks = [('Cardiac axis → brain outcomes', '心脏轴（主模型）', '心脏轴，剔除影像前 P2 并调整脑轴', '脑', COL['heart'], 'CardiacAxis',
               'excl. prior cardiac disease (P2) + brain ageing axis'),
              ('Brain ageing axis → cardiac outcomes', '脑结构老化轴（主模型）', '脑结构老化轴，剔除影像前 P1 并调整心脏轴', '心',
               COL['joint'], 'BrainAxis', 'excl. prior neurological disease (P1) + cardiac axis')]
    rec = []
    XL = (0.55, 4.2)
    for j, (title, mm, ms, organ, col, ex, sens) in enumerate(blocks):
        sub = sel(b2, model=mm, organ=organ)
        keys = [k for k in OUT_ORDER if k in set(sub.outcome)] + NC
        n = len(keys)
        x0 = 27 + j * 92
        rowh = 44.0 / 16; top = 9 + rowh * 16
        ax = c.ax(x0, top - rowh * n, 44, rowh * n)
        ncs = sel(m, exposure=ex, outcome=NC)
        ax.axvspan(ncs.HR.min(), ncs.HR.max(), color=COL['ctrl'], alpha=0.22, lw=0, zorder=0)
        for i, k in enumerate(keys):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
            if k.startswith('NC_'):
                r = one(m, outcome=k, exposure=ex)
                ci_point(ax, r.HR, i, r.lo, r.hi, COL['ctrl'], filled=False, ms=2.5, lw=0.6)
                ax.text(1.02, i, f'{r.HR:.2f} ({r.lo:.2f}–{r.hi:.2f})', transform=ax.get_yaxis_transform(), fontsize=5.0,
                        va='center', color=COL['mid'])
                rec.append(dict(block=title, outcome=k, model='negative control (main)', HR=r.HR, lo=r.lo, hi=r.hi, p=r.p))
                continue
            r1 = one(sub, outcome=k); r2 = one(sel(b2, model=ms, organ=organ), outcome=k)
            ci_point(ax, r1.HR, i - 0.2, r1.lo, r1.hi, col, filled=r1.p < 0.05, ms=2.5, lw=0.6)
            ci_point(ax, r2.HR, i + 0.2, r2.lo, r2.hi, col, filled=r2.p < 0.05, ms=2.4, lw=0.6, marker='D')
            ax.text(1.02, i, f'{r1.HR:.2f} ({r1.lo:.2f}–{r1.hi:.2f})', transform=ax.get_yaxis_transform(), fontsize=5.0,
                    va='center', color=COL['ink'], fontweight='bold' if r1.p < 0.05 else 'normal')
            for mod, r in [('main', r1), (sens, r2)]:
                rec.append(dict(block=title, outcome=k, model=mod, HR=r.HR, lo=r.lo, hi=r.hi, p=r.p, events=r.events))
        ax.set_ylim(n - 0.5, -0.7)
        clean_y(ax, [OUT_SHORT[k] for k in keys], fs=5.3,
                color=[COL['ctrl'] if k.startswith('NC_') else COL['ink'] for k in keys])
        refline(ax, 1); log_x(ax, XL, [0.6, 1, 2, 4])
        ax.set_xlabel('HR per SD (95% CI)', labelpad=1)
        c.text(x0 - 26, top + 1.6, title, fontsize=6.0, color=col, fontweight='bold', va='bottom')
        c.text(x0 + 45.5, top + 1.6, 'Main HR (95% CI)', fontsize=5.2, color=COL['mid'], va='bottom')
    c.legend(legend_handles([('mk', 'Main model', COL['base'], 'o', True),
                             ('mk', 'Excluding prior disease of the exposure organ + adjusted for the other axis', COL['base'], 'D', True),
                             ('pt', 'P ≥ 0.05', COL['base'], False),
                             ('patch', 'Negative-control HR range', to_rgba(COL['ctrl'], 0.35))]),
             1, 0.3, ncol=4, fs=5.1, columnspacing=0.8, handletextpad=0.25)
    c.sd(pd.DataFrame(rec), 'Cross-organ HRs; filled = P < 0.05; sensitivity excludes prior disease of the exposure organ and '
                            'adjusts for the other organ axis')


# ------------------------------------------------------------------ d  vascular gradient
@panel(F, 'c', 83, 58, 'Cardiac-axis risk is graded towards vascular brain disease')
def dpanel(c):
    g = t('B', 'B3_vascular_gradient'); m = t('B', 'B1_full_matrix_cox')
    order = [('CardiacAxis', COL['heart']), ('logWMH', COL['brain']), ('BrainAxis', COL['joint'])]
    ax = c.ax(24, 16, 23, 29); ax2 = c.ax(50, 16, 17, 29)
    rec = []
    for i, (x, col) in enumerate(order):
        r = one(g, exposure=x)
        vas = [k for k in r.vascular_outcomes.split(',') if len(sel(m, outcome=k, exposure=x))]
        non = [k for k in r.nonvascular_outcomes.split(',') if len(sel(m, outcome=k, exposure=x))]
        ax.plot([r.pooled_HR_nonvascular, r.pooled_HR_vascular], [i, i], color=col, lw=1.4, alpha=0.35, zorder=1,
                solid_capstyle='round')
        ax.plot([r.pooled_HR_vascular], [i], 'o', ms=3.6, mfc=col, mec=col, zorder=3)
        ax.plot([r.pooled_HR_nonvascular], [i], 'o', ms=3.6, mfc='white', mec=col, mew=0.8, zorder=3)
        ci_point(ax2, r.ratio_of_HR, i, r.ratio_lo, r.ratio_hi, col, filled=r.p_difference < 0.05, ms=3.2, lw=0.9)
        ax2.text(1.04, i, f'{r.ratio_of_HR:.2f} ({r.ratio_lo:.2f}–{r.ratio_hi:.2f})\nP = {r.p_difference:.3f}',
                 transform=ax2.get_yaxis_transform(), fontsize=4.9, va='center', ha='left', color=COL['ink'],
                 linespacing=1.1, fontweight='bold' if r.p_difference < 0.05 else 'normal')
        if i % 2 == 0:
            for a_ in (ax, ax2):
                a_.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
        rec.append(dict(exposure=x, vascular_modelled=';'.join(vas), nonvascular_modelled=';'.join(non),
                        pooled_HR_vascular=r.pooled_HR_vascular, pooled_HR_nonvascular=r.pooled_HR_nonvascular,
                        ratio_of_HR=r.ratio_of_HR, ratio_lo=r.ratio_lo, ratio_hi=r.ratio_hi, p_difference=r.p_difference,
                        note='estimates treated as independent; conservative'))
    for a_ in (ax, ax2):
        a_.set_ylim(len(order) - 0.5, -0.75); refline(a_, 1)
    clean_y(ax, [EXPO_LAB[o[0]] for o in order], fs=5.8, color=[o[1] for o in order])
    ax2.set_yticks([]); ax2.spines['left'].set_visible(False)
    log_x(ax, (0.8, 2.3), [1, 1.5, 2]); log_x(ax2, (0.45, 3.2), [0.5, 1, 2])
    ax.set_xlabel('Pooled HR per SD', labelpad=1); ax2.set_xlabel('Ratio of HRs', labelpad=1)
    ax.set_title('Vascular vs non-vascular', fontsize=5.6, loc='center', fontweight='normal', pad=1.5)
    ax2.set_title('Vascular ÷ non-vascular', fontsize=5.6, loc='center', fontweight='normal', pad=1.5)
    vas_txt = ', '.join(OUT_SHORT[k] for k in rec[0]['vascular_modelled'].split(';'))
    non_txt = ', '.join(OUT_SHORT[k] for k in rec[0]['nonvascular_modelled'].split(';'))
    c.legend(legend_handles([('pt', f'Vascular: {vas_txt}', COL['base'], True),
                             ('pt', f'Non-vascular: {non_txt}', COL['base'], False)]), 1, 0.3, ncol=2, fs=5.0,
             labelspacing=0.15, handletextpad=0.2, columnspacing=0.8)
    c.text(1, 5.6, 'Pooled by inverse-variance weighting of modelled outcomes (≥ 20 events)', fontsize=4.9, color=COL['mid'])
    c.sd(pd.DataFrame(rec), 'Pooled HRs (inverse-variance) of modelled vascular vs non-vascular brain outcomes; outcomes '
                            'with < 20 events (vascular dementia, Alzheimer disease, multiple sclerosis) not modelled')


LAYOUT = [(F, 'a', 0, 0), (F, 'b', 100, 0), (F, 'c', 100, 74), (F, 'd', 0, 132)]
HEIGHT = 194
