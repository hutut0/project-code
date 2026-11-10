"""Submission Fig. 7 (in-house Fig8) — v4: who carries the risk (B4, C5, C4, C7; 7a curves from the gated D2 re-derivation).
v4: end labels de-overlapped and number-at-risk rows labelled (a); population-share reference line (b); sequential
cardiac-axis strata and explicit pre-specified verdict (c); E-value panel reduced to the ranked robust associations (e).

Discipline: 8a curves are drawn only because the §1.7 gate passed (derived/D2_fig8a_gate_check.csv, all cells);
otherwise the panel falls back to the frozen 3/5-year point estimates. C5 reclassification claim not met → shown
descriptively. C4: no interaction survives FDR. C7: 'robust' only where E-value (CI) > strongest measured covariate.
"""
import numpy as np
import pandas as pd
from scipy import stats
from nclib import (panel, COL, ORGAN_COL, ci_point, refline, clean_y, legend_handles, fmt_ci, log_x, log_y)
from nc_data import t, sel, one, OUT_SHORT, OUT_ORDER, EXPO_LAB, organ_of

F = 'Fig8'
PAIR = '心脏轴 × WMH'
GRP = ['两者均非高', '仅心脏高', '仅脑高', '两者均高']
GLAB = {'两者均非高': 'Neither high', '仅心脏高': 'Cardiac axis high only', '仅脑高': 'WMH high only', '两者均高': 'Both high'}
GCOL = {'两者均非高': COL['ctrl'], '仅心脏高': COL['heart'], '仅脑高': COL['brain'], '两者均高': COL['joint']}
A_OUT = ['Stroke', 'P1_major_neurological', 'P2_major_cardiac']


def _gate():
    g = t('D', 'D2_fig8a_gate_check')
    return bool(g['pass'].all()), g


# ------------------------------------------------------------------ a  cumulative incidence
@panel(F, 'a', 183, 66, 'Joint cardiac-axis and WMH burden marks the highest cumulative incidence')
def a(c):
    ok, gate = _gate()
    J = sel(t('B', 'B4_joint_groups_RERI'), pair=PAIR)
    B4 = sel(t('B', 'B4_cumulative_incidence'), pair=PAIR)
    if ok:
        CU = t('D', 'D2_fig8a_cif_curves'); NR = t('D', 'D2_fig8a_numbers_at_risk')
    rec = []
    for j, k in enumerate(A_OUT):
        x0 = 17 + j * 57
        ax = c.ax(x0, 24, 42, 33)
        top = 0
        for g in GRP:
            if ok:
                q = sel(CU, outcome=k, group=g).sort_values('t_years')
                ax.step(q.t_years, q.cif * 100, where='post', color=GCOL[g], lw=1.0 if g == '两者均高' else 0.8,
                        zorder=3 if g == '两者均高' else 2)
                top = max(top, q.cif.max() * 100)
            else:   # fallback: frozen point estimates only
                r = one(B4, outcome=k, group=g)
                ax.plot([3, 5], [r.cif_3y * 100, r.cif_5y * 100], 'o', color=GCOL[g], ms=2.6)
                top = max(top, r.cif_5y * 100)
        ends = sorted([(one(B4, outcome=k, group=g).cif_5y * 100, g) for g in GRP])
        ypos = []; gap = top * 1.12 * 0.085
        for v_, g in ends:
            y_ = v_ if not ypos else max(v_, ypos[-1] + gap)
            ypos.append(y_)
            ax.text(5.1, y_, f'{v_:.1f}%', fontsize=5.0, color=GCOL[g], va='center', ha='left', clip_on=False,
                    fontweight='bold' if g == '两者均高' else 'normal')
        hr = one(J, outcome=k, group='两者均高')
        ax.text(0.03, 0.97, f'Both high vs neither\nHR {fmt_ci(hr.HR, hr.lo, hr.hi, 1)}', transform=ax.transAxes,
                fontsize=5.2, va='top', ha='left', color=COL['joint'], linespacing=1.15)
        ax.set_xlim(0, 5); ax.set_ylim(0, top * 1.12)
        ax.set_xticks(range(6)); ax.set_xlabel('Years since imaging', labelpad=1)
        ax.set_ylabel('Cumulative incidence (%)' if j == 0 else '')
        ax.set_title(OUT_SHORT[k], fontsize=6, pad=2, color=ORGAN_COL[organ_of(k)])
        # numbers at risk
        tb = c.ax(x0, 2.5, 42, 12); tb.set_xlim(0, 5); tb.set_ylim(0, 4); tb.axis('off')
        for i_, g in enumerate(GRP):
            for yr in range(6):
                if ok:
                    v = one(NR, outcome=k, group=g, year=yr).n_at_risk
                    tb.text(yr, 3.5 - i_, f'{v:,}', fontsize=5.0, ha='center', va='center', color=GCOL[g])
        if ok:
            q = sel(NR, outcome=k).copy(); q['outcome_en'] = OUT_SHORT[k]; q['group_en'] = q.group.map(GLAB)
            rec.append(q)
    c.text(1, 15.6, 'No. at risk', fontsize=5.2, color=COL['ink'], fontweight='bold')
    for i_, (g, lab) in enumerate(zip(GRP, ['Neither', 'Cardiac only', 'WMH only', 'Both high'])):
        c.text(1, 2.5 + 12 * (3.5 - i_) / 4, lab, fontsize=5.0, color=GCOL[g], va='center',
               fontweight='bold' if g == '两者均高' else 'normal')
    c.legend(legend_handles([('line', GLAB[g], GCOL[g], '-') for g in GRP]), 182, 65.6, loc='upper right', ncol=4,
             fs=5.3, columnspacing=0.9, handlelength=1.4)
    if ok:
        cu = CU.copy(); cu['outcome_en'] = cu.outcome.map(OUT_SHORT); cu['group_en'] = cu.group.map(GLAB)
        c.sd(cu, 'Aalen–Johansen cumulative incidence (death as competing event); groups: each exposure ≥ its 2/3 quantile '
                 '(Stage B definition); aggregate step function on a 0.02-year grid')
        c.sd(pd.concat(rec), 'Numbers at risk (aggregate)')
    gg = gate.copy(); gg['outcome_en'] = gg.outcome.map(OUT_SHORT); gg['group_en'] = gg.group.map(GLAB)
    c.sd(gg, 'Reproduction gate (§1.7): re-derived vs frozen B4 CIF at 3 and 5 years; pass = |Δ| < 0.1 pp and same n')
    c.sd(sel(J, outcome=A_OUT).assign(outcome_en=lambda d: d.outcome.map(OUT_SHORT), group_en=lambda d: d.group.map(GLAB)),
         'Joint-group Cox HR versus neither high (B4; RERI row = additive interaction)')


# ------------------------------------------------------------------ b  share of people vs share of events
@panel(F, 'b', 88, 54, 'A small high-burden group carries a large share of events')
def b(c):
    B4 = sel(t('B', 'B4_cumulative_incidence'), pair=PAIR, group='两者均高').copy()
    B4 = B4.sort_values('share_of_events', ascending=False).reset_index(drop=True)
    ax = c.ax(36, 9, 36, 36)
    for i, r in B4.iterrows():
        oc = ORGAN_COL[organ_of(r.outcome)]
        ax.plot([r.share_of_people * 100, r.share_of_events * 100], [i, i], color=COL['ctrl_l'], lw=1.6, zorder=1,
                solid_capstyle='round')
        ax.plot(r.share_of_people * 100, i, 'o', ms=2.6, color=COL['ctrl'], zorder=2)
        ax.plot(r.share_of_events * 100, i, 'o', ms=3.0, color=COL['joint'], zorder=3)
        ax.text(1.03, i, f'×{r.share_of_events / r.share_of_people:.1f}', transform=ax.get_yaxis_transform(),
                fontsize=5.2, va='center', ha='left', color=COL['joint'] if r.share_of_events > r.share_of_people else COL['mid'])
    sp = B4.share_of_people.mean() * 100
    ax.axvline(sp, color=COL['ctrl'], lw=0.6, ls=(0, (2.5, 1.5)), zorder=0)
    ax.text(sp, -1.05, f'{sp:.1f}% of participants', fontsize=5.0, color=COL['mid'], ha='center', va='bottom')
    ax.set_ylim(len(B4) - 0.5, -0.6)
    clean_y(ax, [OUT_SHORT[k] for k in B4.outcome], fs=5.3, color=[ORGAN_COL[organ_of(k)] for k in B4.outcome])
    ax.set_xlim(0, max(B4.share_of_events.max() * 100 * 1.1, 40))
    ax.set_xlabel('Share in the both-high group (%)', labelpad=1)
    ax.text(1.03, -0.9, 'events /\npeople', transform=ax.get_yaxis_transform(), fontsize=5.0, color=COL['mid'],
            va='bottom', ha='left', linespacing=1.0)
    c.legend(legend_handles([('pt', 'Share of people', COL['ctrl'], True), ('pt', 'Share of events', COL['joint'], True)]),
             1, 0.3, ncol=2, fs=5.2)
    sd = B4.copy(); sd['outcome_en'] = sd.outcome.map(OUT_SHORT); sd['events_to_people_ratio'] = sd.share_of_events / sd.share_of_people
    c.sd(sd, 'Both-high group (cardiac axis and WMH each ≥ 2/3 quantile): share of participants and of incident events')


# ------------------------------------------------------------------ c  PREVENT strata
STR = ['心脏轴下三分位', '心脏轴中三分位', '心脏轴上三分位', '心脏轴与WMH均上三分位']
SLAB = {'心脏轴下三分位': 'Cardiac axis T1', '心脏轴中三分位': 'T2', '心脏轴上三分位': 'T3',
        '心脏轴与WMH均上三分位': 'Cardiac axis T3 + WMH T3'}
SCOL = {'心脏轴下三分位': '#F0B7BE', '心脏轴中三分位': '#D9707E', '心脏轴上三分位': '#A3172C',
        '心脏轴与WMH均上三分位': COL['joint']}
PCAT = ['<5%', '5%–7.5%', '7.5%–20%', '≥20%']


@panel(F, 'c', 95, 54, 'Imaging grades risk within PREVENT categories')
def cpanel(c):
    C5 = t('C', 'C5_prevent_by_imaging_risk5'); RS = t('C', 'C5_reclassification_summary')
    rec = []
    for j, k in enumerate(['P2_major_cardiac', 'Stroke']):
        ax = c.ax(11 + j * 44, 13, 37, 32)
        for i, pc in enumerate(PCAT):
            for s_, dx in zip(STR, [-0.27, -0.09, 0.09, 0.27]):
                r = one(C5, outcome=k, prevent_cat=pc, stratum=s_)
                ci_point(ax, i + dx, r.risk5 * 100, r.lo * 100, r.hi * 100, SCOL[s_], filled=r.events >= 10, ms=2.4,
                         lw=0.6, orient='v')
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
        ax.set_xlim(-0.5, 3.5); ax.set_xticks(range(4)); ax.set_xticklabels([p_.replace('%–', '–') for p_ in PCAT], fontsize=5.3)
        ax.tick_params(axis='x', length=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('PREVENT 10-year CVD risk', labelpad=1)
        ax.set_ylabel('5-year risk (%)' if j == 0 else '')
        ax.set_title(OUT_SHORT[k], fontsize=6, pad=2, color=ORGAN_COL[organ_of(k)])
        q = sel(C5, outcome=k).copy(); q['outcome_en'] = OUT_SHORT[k]; q['stratum_en'] = q.stratum.map(SLAB); rec.append(q)
    c.legend(legend_handles([('pt', SLAB[s_], SCOL[s_], True) for s_ in STR]), 1, 0.3, ncol=4, fs=5.2,
             columnspacing=0.6, handletextpad=0.2)
    n_met = int((RS['预设判读'] != '不写该句').sum())
    vt = (f'Pre-specified reclassification criterion not met ({n_met}/{len(RS)} outcomes)' if n_met == 0 else
          f'Pre-specified reclassification criterion met: {n_met}/{len(RS)} outcomes')
    c.text(94, 49.3, vt + '; open: < 10 events', fontsize=5.0, ha='right', color=COL['heart'] if n_met == 0 else COL['mid'])
    c.sd(pd.concat(rec), '5-year Kaplan–Meier risk by PREVENT category and imaging stratum (C5)')
    rs = RS.copy(); rs['criterion_met'] = rs['预设判读'] != '不写该句'
    c.sd(rs, 'Pre-specified rule: claim only if risk(low PREVENT, cardiac T3) ≥ risk(intermediate PREVENT, cardiac T1)')


# ------------------------------------------------------------------ d  QQ of 189 interaction tests
EXS = {'CardiacAxis': (COL['heart'], 'o'), 'BrainAxis': (COL['brain'], 'o'), 'logWMH': (COL['brain'], '^'),
       'FA_r': (COL['brain'], 's'), 'GMVn_r': (COL['brain'], 'D')}


@panel(F, 'd', 62, 54, 'Effect modification no more than chance')
def d(c):
    C4 = t('C', 'C4_effect_modification').copy()
    n = len(C4)
    C4 = C4.sort_values('p_int').reset_index(drop=True)
    i = np.arange(1, n + 1)
    C4['expected'] = -np.log10((i - 0.5) / n)
    C4['observed'] = -np.log10(C4.p_int)
    C4['band_lo'] = -np.log10(stats.beta.ppf(0.975, i, n - i + 1))
    C4['band_hi'] = -np.log10(stats.beta.ppf(0.025, i, n - i + 1))
    ax = c.ax(11, 9, 46, 36)
    ax.fill_between(C4.expected, C4.band_lo, C4.band_hi, color=COL['light'], lw=0, zorder=0)
    mx = max(C4.expected.max(), C4.observed.max()) * 1.05
    ax.plot([0, mx], [0, mx], color=COL['mid'], lw=0.5, ls=(0, (2.5, 1.5)), zorder=1)
    for ex, (col, mk) in EXS.items():
        q = C4[C4.exposure == ex]
        if len(q):
            ax.plot(q.expected, q.observed, mk, ms=2.1, mfc=col, mec='white', mew=0.25, ls='none', zorder=3, alpha=0.9)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel('Expected −log$_{10}$ P', labelpad=1); ax.set_ylabel('Observed −log$_{10}$ P')
    nsig = int((C4.FDR_all189 < 0.05).sum())
    ax.text(0.04, 0.96, f'{n} pre-specified tests\n{nsig} with FDR < 0.05\nmin FDR = {C4.FDR_all189.min():.2f}',
            transform=ax.transAxes, fontsize=5.2, va='top', linespacing=1.15)
    used = [e for e in EXS if e in set(C4.exposure)]
    c.legend(legend_handles([('mk', EXPO_LAB.get(e, e), EXS[e][0], EXS[e][1], True) for e in used]), 61, 9.5,
             loc='lower right', ncol=1, fs=5.0, handletextpad=0.2, labelspacing=0.2)
    sd = C4.copy(); sd['outcome_en'] = sd.outcome.map(OUT_SHORT)
    c.sd(sd, 'Interaction P values (7 modifiers × exposures × outcomes); 95% band from order-statistic Beta distribution')


# ------------------------------------------------------------------ e  E-values (ranked robust associations)
@panel(F, 'e', 121, 54, 'Robustness to unmeasured confounding')
def e(c):
    C7 = t('C', 'C7_evalues').copy()
    C7['robust'] = C7['预设写法'] != '不写该句'
    R = C7[C7.robust].copy()
    R['margin'] = R.E_CI - R.benchmark_HR
    R = R.sort_values('E_CI', ascending=False).reset_index(drop=True)
    ax2 = c.ax(50, 9, 60, 38)
    for i_, r in R.iterrows():
        col = EXS.get(r.exposure, (COL['base'], 'o'))[0]
        ax2.plot([r.benchmark_HR, r.E_point], [i_, i_], color=COL['grid'], lw=1.4, zorder=1, solid_capstyle='round')
        ax2.plot([r.benchmark_HR, r.E_CI], [i_, i_], color=col, lw=0.9, alpha=0.45, zorder=1.5, solid_capstyle='round')
        ax2.plot(r.benchmark_HR, i_, 'd', ms=2.8, color=COL['ctrl'], zorder=2)
        ax2.plot(r.E_CI, i_, 'o', ms=2.8, color=col, zorder=3)
        ax2.plot(r.E_point, i_, 'o', ms=2.8, mfc='white', mec=col, mew=0.6, zorder=3)
        ax2.text(1.015, i_, f'{r.E_CI:.2f}', transform=ax2.get_yaxis_transform(), fontsize=5.0, va='center', ha='left',
                 color=col)
        if i_ % 2 == 0:
            ax2.axhspan(i_ - 0.5, i_ + 0.5, color=COL['light'], lw=0, zorder=0)
    ax2.set_ylim(len(R) - 0.5, -0.6)
    clean_y(ax2, [f"{OUT_SHORT[o]} · {EXPO_LAB.get(x, x)}" for o, x in zip(R.outcome, R.exposure)], fs=5.0,
            color=[ORGAN_COL[organ_of(o)] for o in R.outcome])
    ax2.set_xlabel('Risk-ratio scale', labelpad=1)
    ax2.set_xlim(1, R.E_point.max() * 1.08)
    ax2.text(1.015, -1.0, 'E (CI)', transform=ax2.get_yaxis_transform(), fontsize=5.0, va='bottom', ha='left',
             color=COL['mid'])
    c.text(1, 49.6, f'{len(R)}/{len(C7)} associations: E-value (CI) exceeds the strongest measured covariate',
           fontsize=5.1, color=COL['ink'])
    c.legend(legend_handles([('mk', 'E-value (CI limit)', COL['base'], 'o', True), ('mk', 'E-value (point)', COL['base'], 'o', False),
                             ('mk', 'Strongest measured covariate (HR)', COL['ctrl'], 'd', True)]), 120, 0.3,
             loc='lower right', ncol=3, fs=5.0, columnspacing=0.6, handletextpad=0.2)
    sd = C7.copy(); sd['outcome_en'] = sd.outcome.map(OUT_SHORT); sd['exposure_en'] = sd.exposure.map(EXPO_LAB)
    c.sd(sd, 'E-values for HR point estimate and CI limit; benchmark = largest HR of any measured covariate; '
             'robust if E-value (CI) > benchmark (pre-specified); panel shows the robust subset ranked by E-value (CI)')


LAYOUT = [(F, 'a', 0, 0), (F, 'b', 0, 66), (F, 'c', 88, 66), (F, 'd', 0, 120), (F, 'e', 62, 120)]
HEIGHT = 174
