"""Submission Fig. 4 (in-house Fig5) — v4: hypertrophic traits carry the densest disease signal, and cardiac and
white-matter risks for stroke are parallel and additive.

Panels: a  signal density (share of trait–outcome pairs at FDR < 0.05), trait group x outcome class  [replaces chord]
        b  96 imaging traits x 25 modelled outcomes (heat map)
        c  dose–response (5 curves) + P for non-linearity of all 18 pre-specified curves
        d  pathway decomposition of the cardiac-axis association
        e  cardiac axis and WMH alone vs mutually adjusted (moved from Fig. 3e in v3)
        f  joint cardiac-axis x WMH groups: HRs and RERI (additive interaction)                         [new]
Moved out: censoring at first incident cardiac event -> Supplementary Fig. 3b.
"""
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, LogNorm, to_rgba
from matplotlib.patches import Rectangle
from nclib import (panel, COL, CMAP_DIV, CMAP_SEQ, ci_point, refline, clean_y, legend_handles, heat_colorbar, log_x,
                   log_y, rounded_box)
from nc_data import t, sel, one, OUT_ORDER, OUT_SHORT, EXPO_LAB, TRAIT_GROUP_EN, organ_of

F = 'Fig5'
GROUPS = ['左室容积与功能', '左室质量与壁厚', '右心', '心房', '主动脉', '心肌应变', '脑体积与皮层', '白质微结构', '白质高信号']
GCOL = {'左室容积与功能': '#7A1F2D', '左室质量与壁厚': '#B83A4B', '右心': '#D9707F', '心房': '#E9A3AD',
        '主动脉': '#C9844F', '心肌应变': '#9C4F5E', '脑体积与皮层': '#7FA7CC', '白质微结构': '#2F6690', '白质高信号': '#163A5C'}
ABBR = {'左室容积与功能': 'LV', '左室质量与壁厚': 'LV mass & wall', '右心': 'RV', '心房': 'Atria', '主动脉': 'Ao',
        '心肌应变': 'Myocardial strain', '脑体积与皮层': 'Brain', '白质微结构': 'WM', '白质高信号': 'WMH'}
OCLS = [('心', 'Cardiac', COL['heart']), ('心脑', 'Heart–brain', COL['joint']), ('脑', 'Brain', COL['brain']),
        ('对照', 'Neg. control', COL['ctrl'])]
ORGC = {'心': COL['heart'], '脑': COL['brain'], '心脑': COL['joint'], '对照': COL['ctrl']}


# ------------------------------------------------------------------ a  signal density
@panel(F, 'a', 64, 72, 'Signal density by trait group and outcome class')
def a(c):
    d = t('C', 'C2_trait_disease_atlas').copy()
    d['sig'] = d.FDR_matrix < 0.05
    G = d.groupby(['trait_group', 'outcome_organ']).sig.agg(['sum', 'count']).reset_index()
    G['share'] = G['sum'] / G['count']
    dis = d[d.outcome_organ != '对照'].groupby('trait_group').sig.agg(['sum', 'count'])
    dis['share'] = dis['sum'] / dis['count']
    L, B0, W, H = 24, 15, 24, 42
    ax = c.ax(L, B0, W, H)
    norm = Normalize(0, 0.6)
    nmax = G['count'].max()
    for i, g in enumerate(GROUPS):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
        for j, (o, _, _) in enumerate(OCLS):
            r = G[(G.trait_group == g) & (G.outcome_organ == o)]
            if not len(r):
                continue
            r = r.iloc[0]
            sz = 4 + 70 * np.sqrt(r['count'] / nmax)
            ax.scatter(j, i, s=sz, c=[CMAP_SEQ(norm(min(r.share, 0.6)))], ec=COL['ink'] if r.share > 0 else COL['mid'],
                       lw=0.3, zorder=3)
    ax.axhline(5.5, color=COL['ink'], lw=0.5)
    ax.set_xlim(-0.6, len(OCLS) - 0.4); ax.set_ylim(len(GROUPS) - 0.5, -0.5)
    ax.set_xticks(range(len(OCLS))); ax.set_xticklabels([o[1] for o in OCLS], fontsize=5.0, rotation=40, ha='left',
                                                        rotation_mode='anchor')
    for tl, o in zip(ax.get_xticklabels(), OCLS):
        tl.set_color(o[2])
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1)
    clean_y(ax, [TRAIT_GROUP_EN[g] for g in GROUPS], fs=5.2, color=[GCOL[g] for g in GROUPS])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    # marginal: share across disease outcomes (negative controls excluded)
    mb = c.ax(L + W + 2.0, B0, 14, H)
    for i, g in enumerate(GROUPS):
        r = dis.loc[g]
        mb.barh(i, r.share * 100, height=0.62, color=GCOL[g], lw=0)
        mb.text(r.share * 100 + 1.5, i, f'{r.share * 100:.1f}%', fontsize=4.9, va='center')
    mb.set_ylim(len(GROUPS) - 0.5, -0.5); mb.set_xlim(0, 75); mb.set_yticks([]); mb.spines['left'].set_visible(False)
    mb.set_xticks([0, 25, 50]); mb.tick_params(axis='x', labelsize=5.0, pad=0.8)
    c.text(L + W + 2.0, B0 + H + 1.0, 'Disease\noutcomes (%)', fontsize=5.0, va='bottom', linespacing=0.95)
    # colour bar and size key
    sm = __import__('matplotlib.cm', fromlist=['ScalarMappable']).ScalarMappable(norm=norm, cmap=CMAP_SEQ)
    heat_colorbar(c, sm, 4, 3.2, 16, 1.3, '', ticks=[0, 0.3, 0.6], ticklabels=['0', '30', '60%'])
    c.text(4, 5.6, 'Share at FDR < 0.05', fontsize=5.0, va='bottom')
    for k, n_ in enumerate([20, 100, 400]):
        xk = 30 + k * 8
        ax_ = c.board()
        ax_.scatter([xk], [4.4], s=4 + 70 * np.sqrt(n_ / nmax), c='white', ec=COL['ink'], lw=0.3)
        ax_.text(xk, 1.4, f'{n_}', fontsize=4.9, ha='center')
    c.text(30, 7.4, 'Pairs tested', fontsize=5.0, va='bottom')
    tot = d.assign(nc=d.outcome_organ == '对照').groupby(['trait_organ', 'nc']).sig.agg(['sum', 'count']).reset_index()
    tot['share'] = tot['sum'] / tot['count']
    out = G.copy(); out['trait_group_en'] = out.trait_group.map(TRAIT_GROUP_EN)
    out['outcome_class'] = out.outcome_organ.map({'心': 'cardiac', '心脑': 'heart–brain composite', '脑': 'brain', '对照': 'negative control'})
    c.sd(out, 'Number and share of trait–outcome pairs at FDR < 0.05 (BH applied separately within cardiac and brain traits), '
              'by trait group and outcome class')
    dd = dis.reset_index(); dd['trait_group_en'] = dd.trait_group.map(TRAIT_GROUP_EN)
    c.sd(dd, 'Share at FDR < 0.05 across disease outcomes (negative controls excluded) — marginal bars')
    tot['trait_organ_en'] = tot.trait_organ.map({'心脏': 'cardiac traits', '脑': 'brain traits'})
    c.sd(tot, 'Totals: disease pairs vs negative-control pairs, cardiac and brain traits')


# ------------------------------------------------------------------ b  full atlas heat map
@panel(F, 'b', 119, 72, '96 imaging traits × 25 modelled outcomes')
def b(c):
    d = t('C', 'C2_trait_disease_atlas').copy()
    d['g'] = pd.Categorical(d.trait_group, GROUPS, ordered=True)
    traits = d.drop_duplicates('trait').sort_values(['g'], kind='stable')[['trait', 'trait_group']]
    outs = [k for k in OUT_ORDER if k in set(d.outcome)]
    Bm = d.pivot(index='outcome', columns='trait', values='logHR').loc[outs, traits.trait]
    Q = d.pivot(index='outcome', columns='trait', values='FDR_matrix').loc[outs, traits.trait]
    vmax = np.log(2.0); norm = Normalize(-vmax, vmax)
    rgba = CMAP_DIV(norm(np.clip(Bm.values, -vmax, vmax)))
    ns = ~(Q.values < 0.05)
    rgba[ns] = rgba[ns] * 0.22 + np.array(to_rgba('white')) * 0.78
    rgba[np.isnan(Bm.values)] = to_rgba('#FFFFFF')
    L, B0, W, H = 29, 9, 77, 51
    ax = c.ax(L, B0, W, H)
    ax.imshow(rgba, aspect='auto', interpolation='none')
    ax.set_yticks(range(len(outs))); ax.set_yticklabels([OUT_SHORT[k] for k in outs], fontsize=5.0)
    for tl, k in zip(ax.get_yticklabels(), outs):
        tl.set_color(ORGC[organ_of(k)])
    ax.set_xticks([]); ax.tick_params(length=0, pad=1)
    for s_ in ax.spines.values():
        s_.set_lw(0.4)
    gb = c.ax(L, B0 + H + 0.6, W, 1.6); gb.set_xlim(-0.5, len(traits) - 0.5); gb.set_ylim(0, 1); gb.axis('off')
    x0 = 0
    for g in GROUPS:
        n = int((traits.trait_group == g).sum())
        gb.add_patch(Rectangle((x0 - 0.5, 0), n, 1, fc=GCOL[g], ec='white', lw=0.4))
        ax.axvline(x0 - 0.5, color='white', lw=0.7)
        gb.text(x0 - 0.5 + (n / 2 if n > 1 else 0), 1.5, ABBR[g], ha='center' if n > 1 else 'left', va='bottom',
                fontsize=5.0, color=GCOL[g])
        x0 += n
    ax.axvline(int(sum((traits.trait_group == g).sum() for g in GROUPS[:6])) - 0.5, color=COL['ink'], lw=0.6)
    for i in range(1, len(outs)):
        if organ_of(outs[i]) != organ_of(outs[i - 1]):
            ax.axhline(i - 0.5, color=COL['ink'], lw=0.4)
    sm = __import__('matplotlib.cm', fromlist=['ScalarMappable']).ScalarMappable(norm=norm, cmap=CMAP_DIV)
    heat_colorbar(c, sm, L + W + 2.5, B0 + 22, 1.4, 20, 'HR per SD', ticks=np.log([0.5, 1, 2]), ticklabels=['0.5', '1', '2'],
                  orient='vertical')
    c.text(L + W + 1.5, B0 + 16, 'Faded:\nFDR ≥ 0.05', fontsize=5.0, color=COL['mid'], va='top', linespacing=1.1)
    n_tr = traits.groupby('trait_group').size()
    c.text(1, 3.6, 'Cardiac: ' + ' · '.join(f'{ABBR[g]} {n_tr[g]}' for g in GROUPS[:6]), fontsize=5.0, color=COL['heart'])
    c.text(1, 0.6, 'Brain: ' + ' · '.join(f'{ABBR[g]} {n_tr[g]}' for g in GROUPS[6:]) + '   (number of traits; LV, left ventricle; '
           'RV, right ventricle; Ao, aorta; WM, white matter)', fontsize=5.0, color=COL['brain'])
    c.sd(d.drop(columns='g'), 'All 2,400 trait × outcome Cox models (HR per SD higher trait value); FDR_matrix = BH applied '
                              'separately within cardiac traits and within brain traits')


# ------------------------------------------------------------------ c  dose–response
PAIRS = [('CardiacAxis', 'P2_major_cardiac'), ('CardiacAxis', 'Heart_failure'), ('CardiacAxis', 'Stroke'),
         ('logWMH', 'Stroke'), ('BrainAxis', 'P1_major_neurological')]
XC = {'CardiacAxis': COL['heart'], 'logWMH': COL['brain'], 'BrainAxis': COL['joint']}
TSHORT = {'P2_major_cardiac': 'P2', 'Heart_failure': 'heart failure', 'Stroke': 'stroke', 'P1_major_neurological': 'P1'}


@panel(F, 'c', 183, 44, 'Dose–response is approximately log-linear')
def cp(c):
    cur = t('C', 'C6_dose_response_curves'); ts = t('C', 'C6_dose_response_tests')
    rec = []
    w = 26.5
    for j, (x, o) in enumerate(PAIRS):
        ax = c.ax(11 + j * (w + 4.4), 9, w, 25)
        s = sel(cur, exposure=x, outcome=o).sort_values('z'); col = XC[x]
        ax.fill_between(s.z, s.lo, s.hi, color=col, alpha=0.16, lw=0)
        ax.plot(s.z, s.HR, color=col, lw=1.0)
        r = one(ts, exposure=x, outcome=o)
        zz = np.linspace(s.z.min(), s.z.max(), 50)
        ax.plot(zz, r.HR_linear_perSD ** zz, color=COL['ink'], lw=0.5, ls=(0, (2.5, 1.5)))
        refline(ax, 1, orient='h')
        lo_, hi_ = max(0.2, s.lo.min() * 0.9), min(12, s.hi.max() * 1.1)
        log_y(ax, (lo_, hi_), [v for v in [0.25, 0.5, 1, 2, 4, 8] if lo_ <= v <= hi_])
        ax.set_xlim(s.z.min(), s.z.max())
        ax.set_title(f'{EXPO_LAB[x]} → {TSHORT[o]}', fontsize=5.6, loc='left', pad=2, fontweight='normal', color=col)
        ax.text(0.04, 0.95, f"$P_{{nonlinear}}$ = {r.p_nonlinear:.2f}\n{int(r.events)} events", transform=ax.transAxes,
                fontsize=5.0, va='top', linespacing=1.15)
        ax.set_xlabel('Exposure (SD)', labelpad=1)
        if j == 0:
            ax.set_ylabel('HR (vs mean)')
        rec.append(s.assign(test_HR_linear=r.HR_linear_perSD, p_nonlinear=r.p_nonlinear))
    # all 18 pre-specified curves
    ax = c.ax(167, 9, 13, 25)
    T = ts.copy(); T['col'] = T.exposure.map(XC)
    order = {k: i for i, k in enumerate(['CardiacAxis', 'logWMH', 'BrainAxis'])}
    T = T.sort_values(['exposure', 'p_nonlinear'], key=lambda s_: s_.map(order) if s_.name == 'exposure' else s_)
    T['xj'] = 0.0
    for ex_, grp in T.groupby('exposure'):
        T.loc[grp.index, 'xj'] = order[ex_] + np.linspace(-0.22, 0.22, len(grp))
    for _, r in T.iterrows():
        ax.plot(r.xj, r.p_nonlinear, 'o', ms=2.2, mfc=r.col, mec='white', mew=0.25)
    ax.axhline(0.05, color=COL['bp'], lw=0.5, ls=(0, (2, 1.2)))
    ax.set_ylim(0, 1.02); ax.set_xlim(-0.6, 2.6); ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Cardiac', 'WMH', 'Brain'], fontsize=4.9, rotation=40, ha='right', rotation_mode='anchor')
    ax.set_yticks([0.05, 0.5, 1]); ax.set_yticklabels(['0.05', '0.5', '1'], fontsize=4.9)
    ax.set_title(f'All {len(T)} curves', fontsize=5.5, loc='left', pad=2, fontweight='normal')
    ax.set_ylabel('$P_{nonlinear}$', labelpad=0.5)
    c.legend(legend_handles([('line', 'Restricted cubic spline (95% CI)', COL['base'], '-'),
                             ('line', 'Log-linear fit', COL['ink'], (0, (2.5, 1.5)))]), 182, 43.8, ncol=2, fs=5.1,
             loc='upper right')
    c.sd(pd.concat(rec), 'Restricted cubic spline HR curves (reference = cohort mean) and likelihood-ratio test of non-linearity')
    c.sd(T.drop(columns='col'), 'Likelihood-ratio P for non-linearity, all 18 pre-specified exposure–outcome curves')


# ------------------------------------------------------------------ d  pathway decomposition
MED = [('logWMH', 'WMH', '影像（同日测量）'), ('FA_r', 'Lower FA', '影像（同日测量）'), ('ISOVF', 'ISOVF', '影像（同日测量）'),
       ('BrainAxis', 'Brain ageing axis', '影像（同日测量）'), ('白质合并块', 'All white-matter metrics', '影像（同日测量）'),
       ('AF', 'Incident AF', '时依临床事件（病史型）'), ('Heart_failure', 'Incident heart failure', '时依临床事件（病史型）'),
       ('MI', 'Incident MI', '时依临床事件（病史型）'), ('任一心脏事件', 'Any incident cardiac event', '时依临床事件（病史型）')]


@panel(F, 'd', 108, 62, 'Only a small share runs through white matter or incident cardiac events')
def dp(c):
    p = sel(t('C', 'C3_pathway_decomposition'), exposure='CardiacAxis')
    outs = ['Stroke', 'Ischaemic_stroke', 'P1_major_neurological']
    rec = []
    for j, o in enumerate(outs):
        ax = c.ax(35 + j * 24.2, 12, 21.0, 43)
        for i, (k, lab, typ) in enumerate(MED):
            r = one(p, outcome=o, mediator=k)
            yy = i + (0.5 if typ.startswith('时依') else 0)
            col = COL['brain'] if typ.startswith('影像') else COL['heart']
            ax.barh(yy, r.PM * 100, height=0.62, color=col if r.PM_lo > 0 else 'white', ec=col, lw=0.6, zorder=2)
            ax.plot([r.PM_lo * 100, r.PM_hi * 100], [yy, yy], color=COL['ink'], lw=0.55, zorder=3)
            if k in ('白质合并块', '任一心脏事件'):
                ax.text(31.5, yy, f'{r.PM * 100:.1f}', fontsize=4.9, va='center', ha='right',
                        fontweight='bold' if r.PM_lo > 0 else 'normal', color=COL['ink'])
            rec.append(dict(outcome=o, mediator=k, label=lab, mediator_type='same-day imaging' if typ.startswith('影像') else
                            'time-dependent clinical event', HR_total=r.HR_total, HR_adjusted=r.HR_adjusted, PM=r.PM,
                            PM_lo=r.PM_lo, PM_hi=r.PM_hi, n_boot=r.n_boot, n=r.n, events=r.events))
        ax.axhline(4.75, color=COL['grid'], lw=0.6)
        ax.set_ylim(len(MED) + 0.1, -0.7)
        refline(ax, 0)
        ax.set_xlim(-10, 32); ax.set_xticks([0, 10, 20])
        ax.set_title(OUT_SHORT[o], fontsize=5.8, loc='center', pad=2, fontweight='normal')
        ax.grid(axis='x', color=COL['grid'], lw=0.4, zorder=0)
        if j == 0:
            ys = [i + (0.5 if typ.startswith('时依') else 0) for i, (_, _, typ) in enumerate(MED)]
            clean_y(ax, [m_[1] for m_ in MED], ys, fs=5.3,
                    color=[COL['brain'] if m_[2].startswith('影像') else COL['heart'] for m_ in MED])
        else:
            ax.set_yticks([]); ax.spines['left'].set_visible(False)
        if j == 1:
            ax.set_xlabel('Share of the cardiac-axis HR explained (%)', labelpad=1)
    c.legend(legend_handles([('bar', 'Same-day imaging (cross-sectional)', COL['brain'], True),
                             ('bar', 'Incident cardiac event (time-varying)', COL['heart'], True),
                             ('bar', '95% CI includes 0', COL['base'], False)]), 1, 0.2, ncol=3, fs=5.0, columnspacing=0.7)
    dem = sel(t('C', 'C3_pathway_decomposition'), outcome='Dementia')
    for _, r in dem.iterrows():
        rec.append(dict(outcome='Dementia', mediator=r.mediator, PM=r.PM, PM_lo=r.PM_lo, PM_hi=r.PM_hi, events=r.events,
                        note='descriptive (47 events); not plotted'))
    c.sd(pd.DataFrame(rec), 'Difference-method proportion explained, bootstrap 95% CI; imaging "mediators" are measured on '
                            'the same day as the exposure and do not establish temporal mediation')


# ------------------------------------------------------------------ e  mutual adjustment (moved from Fig. 3e)
@panel(F, 'e', 75, 62, 'Cardiac axis and WMH carry independent risk')
def e(c):
    b4 = t('B', 'B4_mutual_adjustment'); m = t('B', 'B1_full_matrix_cox')
    mod = '心脏轴 + WMH'; modi = '心脏轴 + WMH（乘法交互项）'
    keys = [k for k in OUT_ORDER if k in set(b4.outcome)]
    rec = []
    for j, (x, col) in enumerate([('CardiacAxis', COL['heart']), ('logWMH', COL['brain'])]):
        ax = c.ax(26 + j * 21.5, 11, 19.5, 45)
        for i, k in enumerate(keys):
            r0 = one(m, outcome=k, exposure=x); r1 = one(b4, outcome=k, model=mod, exposure=x)
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
            ax.plot([r0.HR, r1.HR], [i - 0.18, i + 0.18], color=COL['mid'], lw=0.4, zorder=1)
            ci_point(ax, r0.HR, i - 0.18, r0.lo, r0.hi, col, filled=False, ms=2.2, lw=0.5)
            ci_point(ax, r1.HR, i + 0.18, r1.lo, r1.hi, col, filled=True, ms=2.2, lw=0.5)
            rec.append(dict(outcome=k, exposure=x, HR_alone=r0.HR, lo_alone=r0.lo, hi_alone=r0.hi, HR_mutual=r1.HR,
                            lo_mutual=r1.lo, hi_mutual=r1.hi, p_mutual=r1.p, n=r1.n, events=r1.events))
        ax.set_ylim(len(keys) - 0.5, -0.6); refline(ax, 1); log_x(ax, (0.7, 4.2), [1, 2, 4])
        ax.set_title(EXPO_LAB[x], fontsize=5.8, color=col, loc='center', pad=2)
        if j == 0:
            clean_y(ax, [OUT_SHORT[k] for k in keys], fs=5.1, color=[ORGC[organ_of(k)] for k in keys])
        else:
            ax.set_yticks([]); ax.spines['left'].set_visible(False)
    c.text(47.5, 4.6, 'HR per SD (95% CI)', fontsize=5.8, ha='center', va='bottom')
    ax3 = c.ax(68.5, 11, 6, 45); ax3.set_ylim(len(keys) - 0.5, -0.6); ax3.set_xlim(0, 1); ax3.axis('off')
    ax3.text(0.5, -1.25, '$P_{int}$', ha='center', va='center', fontsize=5.4)
    for i, k in enumerate(keys):
        r = one(b4, outcome=k, model=modi, exposure='interaction')
        ps = '<0.001' if r.p < 0.001 else (f'{r.p:.3f}' if r.p < 0.01 else f'{r.p:.2f}')
        ax3.text(0.5, i, ps, ha='center', va='center', fontsize=5.0, fontweight='bold' if r.p < 0.05 else 'normal')
        rec.append(dict(outcome=k, exposure='CardiacAxis × WMH', HR_mutual=r.HR, lo_mutual=r.lo, hi_mutual=r.hi, p_mutual=r.p))
    c.legend(legend_handles([('pt', 'Alone', COL['base'], False), ('pt', 'Mutually adjusted', COL['base'], True)]),
             1, 0.3, ncol=2, fs=5.0)
    c.text(74.5, 0.6, 'Bold: P < 0.05', fontsize=4.9, color=COL['mid'], ha='right')
    c.sd(pd.DataFrame(rec), 'HR alone (B1) vs mutually adjusted (cardiac axis + WMH, B4); multiplicative interaction P')


# ------------------------------------------------------------------ f  joint groups and additive interaction
JG = [('两者均非高', 0, 0), ('仅心脏高', 1, 0), ('仅脑高', 0, 1), ('两者均高', 1, 1)]
F_OUT = ['Stroke', 'P1_major_neurological', 'P2_major_cardiac', 'Heart_failure']


@panel(F, 'f', 183, 46, 'Joint cardiac and white-matter burden is additive for stroke but not for cardiac disease')
def fpanel(c):
    J = sel(t('B', 'B4_joint_groups_RERI'), pair='心脏轴 × WMH')
    norm = LogNorm(1, 6)
    rec = []
    for j, k in enumerate(F_OUT):
        x0 = 15 + j * 42.5
        ax = c.ax(x0, 12, 21, 23)
        for g, cx, wy in JG:
            r = one(J, outcome=k, group=g)
            fc = CMAP_SEQ(norm(max(r.HR, 1.0))) if g != '两者均非高' else '#FFFFFF'
            ax.add_patch(Rectangle((cx, wy), 1, 1, fc=fc, ec='white', lw=1.2))
            tc = 'white' if r.HR > 3.2 else COL['ink']
            lab = 'Ref.' if g == '两者均非高' else f'{r.HR:.2f}'
            ax.text(cx + 0.5, wy + 0.6, lab, ha='center', va='center', fontsize=6.0, color=tc,
                    fontweight='bold' if (g != '两者均非高' and r.lo > 1) else 'normal')
            ax.text(cx + 0.5, wy + 0.27, f'{int(r.events)} ev.', ha='center', va='center', fontsize=4.8,
                    color=tc if g != '两者均非高' else COL['mid'])
            rec.append(dict(outcome=k, outcome_en=OUT_SHORT[k], group=g,
                            group_en={'两者均非高': 'Neither high', '仅心脏高': 'Cardiac axis high only', '仅脑高': 'WMH high only',
                                      '两者均高': 'Both high'}[g], n=r.n, events=r.events, HR=r.HR, lo=r.lo, hi=r.hi, p=r.p))
        ax.add_patch(Rectangle((0, 0), 2, 2, fill=False, ec=COL['ink'], lw=0.5))
        ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect('auto')
        ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['Not high', 'High'], fontsize=5.0)
        ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(['Not high', 'High'] if j == 0 else ['', ''], fontsize=5.0)
        ax.tick_params(length=0, pad=1)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_xlabel('Cardiac axis', labelpad=1, fontsize=5.4, color=COL['heart'])
        if j == 0:
            ax.set_ylabel('WMH', labelpad=1, fontsize=5.4, color=COL['brain'])
        ax.set_title(OUT_SHORT[k], fontsize=6.0, color=ORGC[organ_of(k)], pad=2.0)
        rr = one(J, outcome=k, group='RERI')
        h10, h01, h11 = [one(J, outcome=k, group=g).HR for g in ['仅心脏高', '仅脑高', '两者均高']]
        exp_add = h10 + h01 - 1
        sig = rr.lo > 0 or rr.hi < 0
        bx = c.board(x0 + 22.5, 12, 18.5, 23)
        bx.text(0.3, 18.5, 'RERI', fontsize=5.4, color=COL['mid'], va='bottom')
        mn = lambda v: f'{v:.2f}'.replace('-', '−')
        bx.text(0.3, 13.8, mn(rr.HR), fontsize=8.0, fontweight='bold' if sig else 'normal',
                color=COL['joint'] if sig else COL['base'], va='bottom')
        bx.text(0.3, 10.6, f'({mn(rr.lo)} to {mn(rr.hi)})', fontsize=5.0, color=COL['ink'], va='bottom')
        bx.text(0.3, 3.4, f'Observed {h11:.2f}\nadditive {exp_add:.2f}', fontsize=4.9, color=COL['mid'], va='bottom',
                linespacing=1.15)
        rec.append(dict(outcome=k, outcome_en=OUT_SHORT[k], group='RERI', group_en='RERI (additive interaction)', n=rr.n,
                        events=rr.events, HR=rr.HR, lo=rr.lo, hi=rr.hi, p=rr.p, joint_HR_observed=h11,
                        joint_HR_expected_if_additive=exp_add, note='expected = HR10 + HR01 − 1 (arithmetic on frozen HRs)'))
    sm = __import__('matplotlib.cm', fromlist=['ScalarMappable']).ScalarMappable(norm=norm, cmap=CMAP_SEQ)
    cb = heat_colorbar(c, sm, 15, 2.3, 22, 1.3, '', ticks=[1, 2, 4, 6], ticklabels=['1', '2', '4', '6'])
    cb.ax.minorticks_off()
    c.text(14, 3.0, 'HR', fontsize=5.4, ha='right', va='center')
    c.text(45, 1.4, 'High = top tertile of the cardiac axis or WMH; reference = neither high. Bold HR: 95% CI excludes 1. '
                    'RERI > 0: joint risk exceeds the sum of the separate excesses.', fontsize=5.0, color=COL['mid'])
    c.sd(pd.DataFrame(rec), 'Joint cardiac-axis × WMH groups (top tertile each) vs neither high (B4); RERI with 95% CI')


LAYOUT = [(F, 'a', 0, 0), (F, 'b', 64, 0), (F, 'c', 0, 72), (F, 'd', 0, 116), (F, 'e', 108, 116), (F, 'f', 0, 178)]
HEIGHT = 224
