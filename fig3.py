"""Submission Figure 2 (in-house Fig3) — Q2 (mechanism layer): what links the two organs.

v3 redesign (00_v3_图1图2绘图构思与引导词_NC标准.md): a attenuation with the pre-specified 20% threshold,
b organ age gaps (bubble map), c protein profiles over a density background, d tissue enrichment dumbbells
(incl. the body-size-free axis), e genetic-correlation matrix, f SBP conditioning, g evidence synthesis.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from nclib import panel, COL, CMAP_DIV, ci_point, refline, clean_y, legend_handles, heat_colorbar
from nc_data import t, sel, one

F = 'Fig3'
THR = 20.0     # pre-specified attenuation threshold for a partial explanation (Methods)


def _attenuation():
    r3 = t('R', 'R3b_inflammation_attenuation_of_coupling')
    r4h = t('R', 'R4h_imagingage_organ_age')
    r4g = sel(t('R', 'R4g_shared_protein_attenuation'), outcome='BrainAxis')
    rows = []
    for k, lab in [('BloodInfl', 'Blood-cell inflammation'), ('ProtInfl', 'Proteomic inflammation'),
                   ('Immune_AG', 'Immune age gap')]:
        r = one(r3, exposure=k)
        rows.append(dict(group='Inflammation', label=lab, source='R3b', key=k, n=r.n, beta_without=r.beta_without,
                         beta_with=r.beta_with, att=r.attenuation_pct, lo=r.att_lo, hi=r.att_hi, prespecified=True))
    for k, lab in [('Heart_AG', 'Heart age gap'), ('Artery_AG', 'Artery age gap'), ('Brain_AG', 'Brain age gap'),
                   ('Organismal_AG', 'Organismal age gap'), ('全部12器官', 'All 12 organ age gaps')]:
        r = one(r4h, adjusted_for=k)
        rows.append(dict(group='Proteomic organ ageing', label=lab, source='R4h', key=k, n=np.nan,
                         beta_without=r.beta_without, beta_with=r.beta_with, att=r.attenuation_pct, lo=np.nan,
                         hi=np.nan, prespecified=True))
    for k, lab in [('方向一致的共享蛋白', 'Six concordant proteins'), ('全部共享蛋白', 'All shared proteins')]:
        r = one(r4g, adjusted_for=k)
        rows.append(dict(group='Shared proteins (post hoc)', label=lab, source='R4g', key=k, n=r.n,
                         beta_without=r.beta_without, beta_with=r.beta_with, att=r.attenuation_pct, lo=r.att_lo,
                         hi=r.att_hi, prespecified=False))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ a  attenuation of the coupling
@panel(F, 'a', 72, 64, 'Inflammation and organ ageing do not explain the coupling')
def a(c):
    D = _attenuation()
    ax = c.ax(35, 12, 34, 44)
    ax.axvspan(THR, 70, color=COL['bp'], alpha=0.10, lw=0, zorder=0)
    ax.axvline(THR, color=COL['bp'], lw=0.6, ls=(0, (2.5, 1.5)), zorder=1)
    y = 0; ys, labs = [], []
    for g, sub in D.groupby('group', sort=False):
        ax.text((0.5 - 35) / 34, y, g, transform=ax.get_yaxis_transform(), fontsize=5.7, fontweight='bold',
                color=COL['base'], va='center', clip_on=False)
        y += 1
        for _, r in sub.iterrows():
            col = COL['joint'] if r.prespecified else COL['mid']
            if np.isfinite(r.lo):
                ci_point(ax, r.att, y, r.lo, r.hi, col, filled=not (r.lo <= 0 <= r.hi), ms=2.8, lw=0.7)
            else:
                ax.plot([r.att], [y], marker='s', ms=2.6, mec=col, mfc=COL['joint_l'], mew=0.6, ls='none', zorder=4)
            if not r.prespecified and r.att > THR:
                ax.text(r.hi + 1.5, y, 'upper bound', fontsize=5.0, va='center', color=COL['mid'], style='italic')
            ys.append(y); labs.append(r.label); y += 1
        y += 0.3
    ax.set_ylim(y - 0.5, -0.7)
    clean_y(ax, labs, ys, fs=5.6)
    refline(ax, 0)
    ax.set_xlim(-25, 70); ax.set_xticks([-20, 0, 20, 40, 60])
    ax.text(THR + 1.5, -0.45, '≥20%: pre-specified\npartial explanation', fontsize=5.0, color=COL['bp'], va='top',
            linespacing=0.95)
    ax.set_xlabel('Attenuation of coupling (%)')
    c.legend(legend_handles([('mk', '95% CI excludes 0', COL['joint'], 'o', True),
                             ('mk', 'includes 0', COL['joint'], 'o', False),
                             ('mk', 'CI not estimated', COL['joint'], 's', True)]), 1, 0.6, ncol=3, fs=5.3,
             columnspacing=0.5, handletextpad=0.2)
    c.sd(D, 'Attenuation (%) of the cardiac axis → brain ageing axis coupling (L5) after adding each candidate; '
            f'pre-specified threshold for a partial explanation = {THR:.0f}%; shared proteins selected post hoc in the '
            'same sample (upper bound)')


# ------------------------------------------------------------------ b  organ age gaps vs imaging
ORG = [('Heart_AG', 'Heart'), ('Artery_AG', 'Artery'), ('Brain_AG', 'Brain'), ('Organismal_AG', 'Organismal'),
       ('NonspecOrganismal_AG', 'Non-specific organismal'), ('Kidney_AG', 'Kidney'), ('Liver_AG', 'Liver'),
       ('Lung_AG', 'Lung'), ('Intestine_AG', 'Intestine'), ('Pancreas_AG', 'Pancreas'), ('Muscle_AG', 'Muscle'),
       ('Immune_AG', 'Immune')]
IMG4 = [('CardiacAxis', 'Cardiac\naxis', COL['heart']), ('BrainAxis', 'Brain\naxis', COL['joint']),
        ('FA', 'FA', COL['brain']), ('logWMH', 'WMH', COL['brain'])]


@panel(F, 'b', 50, 64, 'Organ age gaps track their own organ')
def b(c):
    r = sel(t('R', 'R4c_organ_age_gap_vs_imaging'), level='L5')
    B = r.pivot(index='age_gap', columns='outcome', values='beta').loc[[o[0] for o in ORG], [i[0] for i in IMG4]]
    Q = r.pivot(index='age_gap', columns='outcome', values='fdr').loc[B.index, B.columns]
    ax = c.ax(23, 13, 26, 40)
    vmax = 0.07; smax = 62.0
    for i in range(len(ORG)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
    yy, xx = np.meshgrid(np.arange(len(ORG)), np.arange(4), indexing='ij')
    v = B.values; sig = Q.values < 0.05
    sz = np.clip(np.abs(v) / vmax, 0.03, 1) * smax
    ax.scatter(xx[~sig], yy[~sig], s=sz[~sig], c=v[~sig], cmap=CMAP_DIV, vmin=-vmax, vmax=vmax, lw=0, zorder=2)
    sc = ax.scatter(xx[sig], yy[sig], s=sz[sig], c=v[sig], cmap=CMAP_DIV, vmin=-vmax, vmax=vmax,
                    edgecolors=COL['ink'], linewidths=0.45, zorder=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(len(ORG) - 0.5, -0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([i[1] for i in IMG4], fontsize=5.3, linespacing=0.95)
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1)
    for tl, (_, _, col) in zip(ax.get_xticklabels(), IMG4):
        tl.set_color(col)
    ax.set_yticks(range(len(ORG)))
    ax.set_yticklabels([o[1] + ' age gap' if i < 3 else o[1] for i, o in enumerate(ORG)], fontsize=5.6)
    heat_colorbar(c, sc, 25, 7.0, 20, 1.4, '', ticks=[-0.07, 0, 0.07], ticklabels=['−0.07', '0', '0.07'])
    c.text(23.5, 7.7, 'β', fontsize=5.8, ha='right', va='center')
    c.text(49, 0.9, 'outlined: FDR < 0.05; area scales with |β|', fontsize=4.9, color=COL['mid'], ha='right')
    c.sd(sel(r, outcome=[i[0] for i in IMG4]), 'Proteomic organ age gap (per SD) vs imaging trait, L5 model; '
                                               'FDR across 60 tests; bubble area proportional to |beta|')


# ------------------------------------------------------------------ c  protein association profiles
@panel(F, 'c', 61, 64, 'Proteome-wide associations with both axes')
def cc(c):
    from adjustText import adjust_text
    p = sel(t('R', 'R4a_proteome_scan'), level='L5')
    X = p[p.outcome == 'CardiacAxis'].set_index('protein')
    Y = p[p.outcome == 'BrainAxis'].set_index('protein')
    D = X[['beta', 'fdr']].join(Y[['beta', 'fdr']], lsuffix='_cardiac', rsuffix='_brain', how='inner').reset_index()
    sc, sb = D.fdr_cardiac < 0.05, D.fdr_brain < 0.05
    D['class'] = np.select([sc & sb, sc, sb], ['both', 'cardiac axis only', 'brain axis only'], 'neither')
    ax = c.ax(11, 11, 47, 43)
    lim = np.nanmax(np.abs(D[['beta_cardiac', 'beta_brain']].values)) * 1.08
    nb = D[D['class'] == 'neither']
    grey = LinearSegmentedColormap.from_list('dens', ['#E9ECF0', '#9AA3AE', '#4A4E57'])
    ax.hexbin(nb.beta_cardiac, nb.beta_brain, gridsize=30, extent=(-lim, lim, -lim, lim), bins='log',
              cmap=grey, mincnt=1, linewidths=0.0, zorder=1, rasterized=True)
    sty = {'cardiac axis only': (COL['heart'], 3.0, 2), 'brain axis only': (COL['brain'], 3.0, 2), 'both': (COL['joint'], 9, 4)}
    for k in ['cardiac axis only', 'brain axis only', 'both']:
        s = D[D['class'] == k]; col, sz, z = sty[k]
        ax.scatter(s.beta_cardiac, s.beta_brain, s=sz, c=col, lw=0.25 if k == 'both' else 0,
                   edgecolors='white', zorder=z, alpha=1 if k == 'both' else 0.8)
    refline(ax, 0); refline(ax, 0, orient='h')
    both = D[D['class'] == 'both']
    txt = [ax.text(r.beta_cardiac, r.beta_brain, r.protein, fontsize=5.2, color=COL['joint'], style='italic', zorder=5)
           for _, r in both.iterrows()]
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    adjust_text(txt, ax=ax, x=both.beta_cardiac.values, y=both.beta_brain.values,
                arrowprops=dict(arrowstyle='-', color=COL['joint'], lw=0.3), expand=(1.3, 1.5), force_text=(0.4, 0.6))
    ax.set_xlabel('β, protein → cardiac axis'); ax.set_ylabel('β, protein → brain ageing axis', labelpad=1)
    ax.locator_params(nbins=5)
    n = D['class'].value_counts()
    c.legend(legend_handles([('pt', f"Both FDR < 0.05 (n = {n.get('both', 0)})", COL['joint'], True),
                             ('pt', f"Cardiac axis only ({n.get('cardiac axis only', 0)})", COL['heart'], True),
                             ('pt', f"Brain axis only ({n.get('brain axis only', 0)})", COL['brain'], True)]),
             12.0, 54.6, loc='upper left', fs=5.2, handletextpad=0.1, labelspacing=0.12)
    c.text(57.5, 12.5, f'grey density: {len(nb):,} other proteins', fontsize=4.9, ha='right', color=COL['mid'])
    c.sd(D, f'Olink proteome scan (n proteins = {len(D)}), L5 model, discovery cohort; FDR within each imaging trait; '
            'proteins significant for neither axis are shown as a hexagonal density (log counts)')


# ------------------------------------------------------------------ d  tissue enrichment (dumbbells)
@panel(F, 'd', 62, 60, 'Brain-enriched proteins track the cardiac axis')
def dd(c):
    b_ = sel(t('R', 'R4b_tissue_enrichment'), level='L5')
    sf = t('R', 'R4f_tissue_enrichment_sizefree')
    outs = [('CardiacAxis', 'Cardiac axis', COL['heart']), ('SizeFree', 'Body-size-free axis', COL['heart']),
            ('BrainAxis', 'Brain ageing axis', COL['joint']), ('FA', 'FA', COL['brain']), ('GMVn', 'GMV', COL['brain']),
            ('logWMH', 'WMH', COL['brain'])]
    rows = []
    for j, (tis, tlab, xmax, xt) in enumerate([('脑高表达', 'Brain-enriched', 66, [0, 20, 40, 60]),
                                               ('心肌高表达', 'Heart-enriched', 22, [0, 10, 20])]):
        ax = c.ax(25 + j * 20, 12, 16.5 if j == 0 else 14, 40)
        bg = one(b_, outcome='CardiacAxis', tissue=tis).pct_background
        for i, (o, lab, col) in enumerate(outs):
            if o == 'SizeFree':
                r = one(sf, tissue=tis); nsig, nin, pct, OR, p = r.n_sig, r.n_in, r.pct, r.OR, r.p
            else:
                r = one(b_, outcome=o, tissue=tis); nsig, nin, pct, OR, p = r.n_sig, r.n_tissue_in_sig, r.pct_in_sig, r.OR, r.p
            ci = binomtest(int(nin), int(nsig)).proportion_ci(method='wilson')
            lo, hi = ci.low * 100, ci.high * 100
            ax.plot([bg, pct], [i, i], color=col, lw=1.4 if p < 0.05 else 0.6, alpha=0.35, solid_capstyle='round', zorder=1)
            ci_point(ax, pct, i, lo, hi, col, filled=p < 0.05, ms=2.8, lw=0.6)
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
            rows.append(dict(outcome=o, tissue=tlab, n_associated=nsig, n_tissue=nin, pct=pct, wilson_lo=lo,
                             wilson_hi=hi, pct_background=bg, OR=OR, fisher_p=p,
                             source='R4f' if o == 'SizeFree' else 'R4b'))
        refline(ax, bg, color=COL['ink'], lw=0.5)
        ax.set_ylim(len(outs) - 0.5, -0.6)
        ax.set_xlim(0, xmax); ax.set_xticks(xt)
        ax.set_title(tlab, fontsize=6, fontweight='normal', loc='center', pad=2)
        if j == 0:
            clean_y(ax, [o[1] for o in outs], fs=5.6, color=[o[2] for o in outs])
        else:
            ax.set_yticks([]); ax.spines['left'].set_visible(False)
    c.text(42, 4.6, '% of associated proteins (95% CI)', ha='center', fontsize=6.0)
    c.legend(legend_handles([('pt', 'Fisher P < 0.05', COL['base'], True), ('pt', 'P ≥ 0.05', COL['base'], False),
                             ('line', 'All proteins', COL['ink'], (0, (2.5, 1.5)))]),
             1, 0.3, ncol=3, fs=5.2, columnspacing=0.5, handletextpad=0.3)
    c.sd(pd.DataFrame(rows), 'Tissue enrichment (HPA) of proteins associated (FDR < 0.05, L5); Wilson 95% CI added from the '
                             'counts; dumbbell = background share (all proteins) to observed share; body-size-free axis from R4f')


# ------------------------------------------------------------------ e  genetic correlation matrix
@panel(F, 'e', 56, 60, 'Little shared heritability')
def e(c):
    s = t('S', 'S6_ldsc_rg')
    gb = t('S', 'S10_genetic_contribution_bound')
    H3 = ['LVEDV', 'LVESV', 'LVEF']; W3 = ['FA_genu', 'ISOVF_genu', 'WMH']
    en = {'LVEDV': 'LVEDV', 'LVESV': 'LVESV', 'LVEF': 'LVEF', 'FA_genu': 'FA', 'ISOVF_genu': 'ISOVF', 'WMH': 'WMH',
          'Stroke': 'Stroke'}
    q = sel(s, family='GQ3_heart_x_WM').copy()
    q['heart'] = np.where(q.trait1.isin(H3), q.trait1, q.trait2)
    q['wm'] = np.where(q.trait1.isin(H3), q.trait2, q.trait1)
    M = q.pivot(index='heart', columns='wm', values='rg').loc[H3, W3]
    SE = q.pivot(index='heart', columns='wm', values='rg_se').loc[H3, W3]
    FD = q.pivot(index='heart', columns='wm', values='FDR_family').loc[H3, W3]
    vmax = 0.6
    ax = c.ax(14, 27, 27, 24)
    im = ax.imshow(M.values, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax, aspect='auto', interpolation='none')
    for i in range(3):
        for j in range(3):
            v = M.values[i, j]
            ax.text(j, i - 0.1, f'{v:.2f}'.replace('-', '−'), ha='center', va='center', fontsize=5.6, color=COL['ink'],
                    fontweight='bold' if FD.values[i, j] < 0.05 else 'normal')
            ax.text(j, i + 0.27, f'±{SE.values[i, j]:.2f}', ha='center', va='center', fontsize=4.6, color=COL['mid'])
    ax.set_xticks(range(3)); ax.set_xticklabels([en[w] for w in W3], fontsize=5.6, color=COL['brain'])
    ax.set_yticks(range(3)); ax.set_yticklabels([en[h] for h in H3], fontsize=5.6, color=COL['heart'])
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=1)
    ax.set_xticks(np.arange(-.5, 3), minor=True); ax.set_yticks(np.arange(-.5, 3), minor=True)
    ax.grid(which='minor', color='white', lw=0.8); ax.tick_params(which='minor', length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    heat_colorbar(c, im, 44, 33, 1.4, 16, '', ticks=[-0.6, 0, 0.6], ticklabels=['−0.6', '0', '0.6'], orient='vertical')
    c.text(44.7, 51.5, '$r_g$', fontsize=5.8, ha='center')
    c.text(27.5, 22.3, 'All 9 pairs FDR ≥ 0.05' if (FD.values >= 0.05).all() else 'Some pairs FDR < 0.05',
           ha='center', fontsize=5.3, color=COL['mid'])
    ref = [sel(s, family='WM_x_disease', trait1='WMH', trait2='Stroke'), sel(s, family='WM_x_WM', trait1='FA_genu', trait2='WMH'),
           sel(s, family='heart_x_heart', trait1='LVEDV', trait2='LVEF')]
    ref = pd.concat(ref)
    g = c.board(2, 9.5, 54, 10)
    g.text(0.5, 9.2, 'Reference pairs', fontsize=5.4, fontweight='bold', color=COL['base'], va='top')
    for k, (_, r) in enumerate(ref.iterrows()):
        x0 = 1 + k * 17.6
        col = CMAP_DIV((r.rg + vmax) / (2 * vmax))
        g.add_patch(Rectangle((x0, 0.8), 5.2, 4.6, fc=col, ec='white', lw=0.5))
        g.text(x0 + 2.6, 3.1, f'{r.rg:.2f}'.replace('-', '−'), ha='center', va='center', fontsize=5.0,
               color='white' if abs(r.rg) > 0.4 else COL['ink'], fontweight='bold' if r.FDR_family < 0.05 else 'normal')
        g.text(x0 + 6.0, 3.1, f'{en[r.trait1]}–\n{en[r.trait2]}', ha='left', va='center', fontsize=4.9, linespacing=0.95)
    lo_, hi_ = gb.share_point_of_B1.min() * 100, gb.share_point_of_B1.max() * 100
    blo, bhi = gb.share_bound_of_B1.min() * 100, gb.share_bound_of_B1.max() * 100
    c.text(2.5, 4.4, f'Share of coupling attributable to $r_g$: {lo_:.0f}–{hi_:.0f}%', fontsize=5.2, color=COL['joint'])
    c.text(2.5, 1.0, f'(upper bounds {blo:.0f}–{bhi:.0f}%; bold = FDR < 0.05)', fontsize=5.0, color=COL['mid'])
    c.sd(pd.concat([q.assign(block='heart x white matter'), ref.assign(block='reference pair')])
         [['block', 'family', 'trait1', 'trait2', 'nsnp', 'rg', 'rg_se', 'p', 'FDR_family', 'gcov_int', 'qc']],
         'LDSC genetic correlations; FDR within pre-specified family; share of coupling attributable to rg in '
         'S10_genetic_contribution_bound (point and upper bound)')


# ------------------------------------------------------------------ f  conditioning on SBP, same instruments
@panel(F, 'f', 65, 60, 'Blood pressure as a shared upstream load')
def f(c):
    b5 = t('BP', 'B05_same_instrument_comparison_without_BP')
    b2 = sel(t('BP', 'B02_all_model_estimates'), BP='SBP', method='IVW')
    spec = [('CardiacAxisS4+SBP__FA_genu', 'CardiacAxisS4', 'Cardiac axis → FA', COL['heart']),
            ('CardiacAxisS4+SBP__ISOVF_genu', 'CardiacAxisS4', 'Cardiac axis → ISOVF', COL['heart']),
            ('CardiacAxisS4+SBP__WMH', 'CardiacAxisS4', 'Cardiac axis → WMH', COL['heart']),
            ('CardiacAxisS4+SBP__Stroke', 'CardiacAxisS4', 'Cardiac axis → stroke', COL['heart']),
            ('WMH+SBP__Stroke', 'WMH', 'WMH → stroke', COL['brain'])]
    ax = c.ax(25, 12, 31, 41)
    rec = []
    for i, (m, ex, lab, col) in enumerate(spec):
        w = one(b5, model=m, exposure=ex); s_ = one(b2, model=m, exposure=ex)
        wlo, whi = w.beta - 1.96 * w.se, w.beta + 1.96 * w.se
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
        ax.add_patch(FancyArrowPatch((w.beta, i - 0.17), (s_.beta, i + 0.17), arrowstyle='-|>', mutation_scale=4.5,
                                     color=COL['mid'], lw=0.5, shrinkA=2.2, shrinkB=2.2, zorder=1))
        ci_point(ax, w.beta, i - 0.17, wlo, whi, col, filled=False, ms=2.8, lw=0.7)
        ci_point(ax, s_.beta, i + 0.17, s_.lo, s_.hi, col, filled=True, ms=2.8, lw=0.7)
        ax.text(1.12, i, f'{s_.F_conditional_standard:.1f}', transform=ax.get_yaxis_transform(), fontsize=5.4,
                va='center', ha='center', color=COL['bp'] if s_.F_conditional_standard < 10 else COL['ink'])
        rec.append(dict(model=m, exposure=ex, label=lab, nsnp=w.nsnp, beta_without_SBP=w.beta, lo_without=wlo, hi_without=whi,
                        p_without=w.p, beta_with_SBP=s_.beta, lo_with=s_.lo, hi_with=s_.hi, p_with=s_.p,
                        conditional_F_exposure=s_.F_conditional_standard))
    ax.text(1.12, -0.75, '$F_{cond}$', transform=ax.get_yaxis_transform(), fontsize=5.6, ha='center', va='center')
    ax.set_ylim(len(spec) - 0.5, -0.9)
    clean_y(ax, [s[2] for s in spec], fs=5.6, color=[s[3] for s in spec])
    refline(ax, 0)
    ax.set_xlim(-0.45, 0.65); ax.set_xticks([-0.4, 0, 0.4])
    ax.set_xlabel('β or log OR per SD (95% CI)')
    c.legend(legend_handles([('pt', 'Same SNPs, no SBP', COL['base'], False), ('pt', '+ SBP (MVMR)', COL['base'], True),
                             ('patch', '$F_{cond}$ < 10', COL['bp'])]),
             1, 0.3, ncol=3, fs=5.2, columnspacing=0.5, handletextpad=0.3)
    c.sd(pd.DataFrame(rec), 'Same final instruments with and without SBP (IVW-MVMR); conditional F < 10 = weak conditional instrument; '
                            'attenuation is not evidence of mediation')


# ------------------------------------------------------------------ g  evidence synthesis
def _box(g, x, y, w, h, fc, ec, lw=0.6, r=1.2):
    g.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=2))


def _arrow(g, p0, p1, col, lw=0.9, ls='-', style='-|>', ms=6):
    g.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=col, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0, zorder=3))


@panel(F, 'g', 183, 48, 'Synthesis: blood pressure as a shared upstream load of the coupling')
def g(c):
    D = _attenuation()
    d1 = sel(t('R', 'R1d_imagingage_hierarchical'), outcome='BrainAxis', cohort='Meta')
    l5 = list(dict.fromkeys(d1.level))[-1]
    coup = one(d1, level=l5).beta
    gb = t('S', 'S10_genetic_contribution_bound')
    s6 = sel(t('S', 'S6_ldsc_rg'), family='GQ3_heart_x_WM')
    b2 = sel(t('BP', 'B02_all_model_estimates'), BP='SBP', method='IVW')
    st = one(b2, model='CardiacAxisS4+SBP__Stroke', exposure='CardiacAxisS4')
    G = c.board(0, 0, 183, 43)
    _box(G, 3, 14, 37, 15, '#FBF1DC', COL['bp'], lw=0.8)
    G.text(21.5, 25.6, 'Blood pressure', ha='center', va='center', fontsize=6.6, fontweight='bold', color=COL['bp'])
    G.text(21.5, 21.0, 'shared upstream load', ha='center', va='center', fontsize=5.6, color=COL['ink'])
    G.text(21.5, 17.4, 'genetic evidence (Fig. 2f)', ha='center', va='center', fontsize=5.0, color=COL['mid'])
    _box(G, 56, 27, 44, 12.5, '#F8E6E9', COL['heart'], lw=0.8)
    G.text(78, 35.4, 'Cardiac remodelling', ha='center', va='center', fontsize=6.4, fontweight='bold', color=COL['heart'])
    G.text(78, 30.6, 'hypertrophy-led axis (Fig. 1a)', ha='center', va='center', fontsize=5.4, color=COL['ink'])
    _box(G, 56, 3.5, 44, 12.5, '#E4EDF5', COL['brain'], lw=0.8)
    G.text(78, 11.9, 'Deep white-matter injury', ha='center', va='center', fontsize=6.4, fontweight='bold', color=COL['brain'])
    G.text(78, 7.1, 'lower FA, higher MD/ISOVF, WMH (Fig. 1d,e)', ha='center', va='center', fontsize=5.4, color=COL['ink'])
    _arrow(G, (40.3, 24.5), (55.6, 32.5), COL['bp'], lw=1.0)
    _arrow(G, (40.3, 18.5), (55.6, 10.5), COL['bp'], lw=1.0)
    _arrow(G, (78, 26.6), (78, 16.4), COL['mid'], lw=0.7, ls=(0, (2.2, 1.4)), style='<|-|>', ms=5)
    G.text(81.2, 21.5, f'phenotypic coupling\nβ = {coup:.3f} per SD (L5)', ha='left', va='center', fontsize=5.3,
           color=COL['base'], linespacing=1.35)
    G.text(108, 41.5, 'Candidate explanations of the coupling', fontsize=6.0, fontweight='bold', color=COL['ink'], va='top')
    infl = D[D.group == 'Inflammation'].att.max(); org = D[D.group == 'Proteomic organ ageing'].att.max()
    prot = one(D, key='方向一致的共享蛋白')
    gen_txt = (f'$r_g$ n.s.; share {gb.share_point_of_B1.min() * 100:.0f}–{gb.share_point_of_B1.max() * 100:.0f}%'
               if (s6.FDR_family >= 0.05).all() else 'see Fig. 2e')
    rows = [('Inflammation (3 measures)', f'max {infl:.1f}%'.replace('-', '−'), 'not explanatory', COL['ctrl']),
            ('Proteomic organ ageing (5)', f'max {org:.1f}%'.replace('-', '−'), 'not explanatory', COL['ctrl']),
            ('Shared genetics (9 pairs)', gen_txt, 'small', COL['ctrl']),
            ('Six shared proteins (post hoc)', f'{prot.att:.1f}% (upper bound)', 'partial', COL['joint'])]
    for k, (nm, val, verdict, col) in enumerate(rows):
        yk = 33.6 - k * 6.4
        G.add_patch(Rectangle((108, yk - 2.3), 73, 4.6, fc=COL['light'] if k % 2 == 0 else 'white', ec='none', zorder=1))
        G.text(109.5, yk, nm, fontsize=5.4, va='center', color=COL['ink'])
        G.text(140, yk, val, fontsize=5.2, va='center', color=COL['base'])
        G.add_patch(Rectangle((178.6, yk - 1.6), 1.6, 3.2, fc=col, ec='none', zorder=2))
        G.text(177.6, yk, verdict, fontsize=5.0, va='center', ha='right', color=col if col != COL['ctrl'] else COL['mid'])
    G.text(108, 5.6, f'Pre-specified threshold for a partial explanation: ≥{THR:.0f}% attenuation (Fig. 2a).',
           fontsize=5.0, color=COL['mid'], va='center')
    G.text(108, 2.2, 'Genetic estimates support a shared load; they do not establish mediation.', fontsize=5.0,
           color=COL['mid'], va='center', style='italic')
    c.sd(pd.DataFrame([dict(element='coupling_beta_L5_meta', value=coup),
                       dict(element='cardiac_axis_to_stroke_beta_after_SBP', value=st.beta),
                       dict(element='max_attenuation_inflammation_pct', value=infl),
                       dict(element='max_attenuation_organ_ageing_pct', value=org),
                       dict(element='shared_protein_attenuation_pct_post_hoc_upper_bound', value=prot.att),
                       dict(element='genetic_share_point_min_pct', value=gb.share_point_of_B1.min() * 100),
                       dict(element='genetic_share_point_max_pct', value=gb.share_point_of_B1.max() * 100)]),
         'Numbers displayed in the synthesis schematic (taken from the files used in panels a–f)')


LAYOUT = [(F, 'a', 0, 0), (F, 'b', 72, 0), (F, 'c', 122, 0), (F, 'd', 0, 64), (F, 'e', 62, 64), (F, 'f', 118, 64),
          (F, 'g', 0, 124)]
HEIGHT = 172
