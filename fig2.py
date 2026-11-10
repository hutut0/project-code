"""Submission Figure 1 (in-house Fig2) — Q1: how cardiac and brain structure are related.

v3 redesign (00_v3_图1图2绘图构思与引导词_NC标准.md): a axis anatomy, b bubble association map,
c adjustment cascade, d JHU tract projections, e regional distributions, f head/body size, g age equivalence.
Every number is read from frozen result files; the JHU atlas is used for display only.
"""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from nclib import (panel, COL, CMAP_DIV, ci_point, refline, clean_y, legend_handles, heat_colorbar)
from nc_data import t, sel, one, H
import brainmap as bm

F = 'Fig2'
CARD_EN = {'CardiacAxis': 'Cardiac axis', 'LVM': 'LV mass', 'WT_global': 'LV wall thickness', 'LVEDV': 'LVEDV',
           'LVEF': 'LVEF', 'Ell_global': 'GLS (higher = worse)', 'Ecc_global': 'GCS (higher = worse)',
           'AoStiff': 'Ascending aortic stiffness', 'AAo_max': 'Ascending aorta area',
           'DAo_max': 'Descending aorta area', 'LAV_max': 'LA maximal volume'}
BRAIN_EN = {'GMVn': 'GMV', 'HCV': 'Hippocampus', 'Thal': 'Thalamus', 'LatVent': 'Lat. ventricles',
            'CTh': 'Cortical thickness', 'FA': 'FA', 'MD': 'MD', 'ICVF': 'ICVF', 'OD': 'OD', 'ISOVF': 'ISOVF',
            'logWMH': 'WMH'}
GREY = ['GMVn', 'HCV', 'Thal', 'LatVent', 'CTh']
WHITE = ['FA', 'MD', 'ICVF', 'OD', 'ISOVF', 'logWMH']
CMAP_FA = LinearSegmentedColormap.from_list('fa_seq', ['#0B2A4A', '#1E5A8C', '#4F86B8', '#93B8D8', '#DCE8F3'], N=256)


def swarm(x, dx, step, cap):
    """Deterministic beeswarm offsets: points closer than dx in x are stacked at multiples of step."""
    x = np.asarray(x, float)
    order = np.argsort(x)
    off = np.zeros(len(x))
    placed = []
    for i in order:
        cand = [0.0] + [s * k * step for k in range(1, 40) for s in (1, -1)]
        for o in cand:
            if abs(o) > cap:
                o = cand[0]
                break
            if all(abs(x[i] - x[j]) >= dx or abs(o - off[j]) >= step * 0.95 for j in placed):
                break
        off[i] = o
        placed.append(i)
    return off


# ------------------------------------------------------------------ a  cardiac axis anatomy
GROUPS = [('LV mass and wall thickness', 'LV mass & wall'), ('Right ventricle', 'Right ventricle'),
          ('LV volumes and function', 'LV volume & function'), ('Aorta', 'Aorta'), ('Atria', 'Atria'),
          ('Myocardial strain', 'Myocardial strain')]


@panel(F, 'a', 60, 62, 'A hypertrophy-led cardiac axis')
def a(c):
    info = t('R', 'R1_axis_info').set_index('axis').loc['心脏重构轴']
    L = sel(t('R', 'R1_axis_loadings'), axis='心脏重构轴')
    st2 = pd.read_csv(H / 'manuscript/supp/tables_v2/ST2_Cardiac_82_traits.csv')
    L = L.merge(st2[['Variable (analysis name)', 'Original field title', 'Trait group']],
                left_on='feature', right_on='Variable (analysis name)', how='left')
    assert L['Trait group'].notna().all() and len(L) == 82
    ax = c.ax(22, 9, 35, 37)
    norm = Normalize(-0.21, 0.21)
    rec = []
    for i, (g, lab) in enumerate(GROUPS):
        s = L[L['Trait group'] == g]
        off = swarm(s.loading_discovery.values, 0.0085, 0.12, 0.42)
        ax.scatter(s.loading_discovery, i + off, s=5.5, c=CMAP_DIV(norm(s.loading_discovery.values)),
                   edgecolors=COL['ink'], linewidths=0.25, zorder=3)
        med = s.loading_discovery.median()
        ax.plot([med, med], [i - 0.38, i + 0.38], color=COL['ink'], lw=0.8, zorder=4, solid_capstyle='butt')
        for (_, r), o in zip(s.iterrows(), off):
            rec.append(dict(trait_group=g, feature=r.feature, title=r['Original field title'],
                            loading_discovery=r.loading_discovery, loading_replication=r.loading_replication,
                            y_offset_display=o))
    refline(ax, 0)
    ax.set_ylim(len(GROUPS) - 0.45, -0.55)
    clean_y(ax, [g[1] for g in GROUPS], fs=5.6)
    ax.set_xlim(-0.14, 0.235); ax.set_xticks([-0.1, 0, 0.1, 0.2])
    ax.set_xlabel('Loading on cardiac axis (PC1)')
    for f_, txt, dx, dy, ha in [('LVM', 'LV mass ', -0.004, -0.62, 'center'), ('WT_global', 'Wall thickness ', -0.06, 0.56, 'right'),
                                ('LVEF', 'LVEF ', 0.012, 0.52, 'left')]:
        r = one(L, feature=f_); y0 = [g[0] for g in GROUPS].index(r['Trait group'])
        yo = [x['y_offset_display'] for x in rec if x['feature'] == f_][0]
        ax.annotate(f'{txt}{r.loading_discovery:.3f}'.replace('-', '−'), (r.loading_discovery, y0 + yo),
                    xytext=(r.loading_discovery + dx, y0 + yo + dy), fontsize=5.2, ha=ha, va='center', linespacing=0.95,
                    color=COL['ink'], arrowprops=dict(arrowstyle='-', lw=0.35, color=COL['ink'], shrinkA=0, shrinkB=1.5),
                    zorder=6)
    c.text(22, 54.0, f'PC1 of 82 cardiac MRI traits: {info.var_exp_discovery * 100:.1f}% of variance',
           fontsize=5.4, color=COL['base'])
    c.text(22, 50.6, f'replication {info.var_exp_replication * 100:.1f}%; Tucker φ = {info.tucker:.3f}',
           fontsize=5.4, color=COL['base'])
    c.legend([Line2D([], [], color=COL['ink'], lw=0.8, label='Group median')], 1, 0.4, fs=5.3)
    c.sd(pd.DataFrame(rec), 'Discovery-sample loadings of 82 cardiac MRI traits on the cardiac remodelling axis (PC1); '
                            f'variance explained {info.var_exp_discovery:.4f} (replication {info.var_exp_replication:.4f}); '
                            f'Tucker congruence {info.tucker:.4f}; trait groups from Supplementary Table 2')


# ------------------------------------------------------------------ b  bubble association map
@panel(F, 'b', 123, 62, 'Strongest associations: hypertrophic traits with white-matter metrics')
def b(c):
    r = t('R', 'R2b_cardiac_brain_matrix')
    rows = list(CARD_EN); cols = GREY + WHITE
    B = r.pivot(index='cardiac', columns='brain', values='beta').loc[rows, cols]
    Q = r.pivot(index='cardiac', columns='brain', values='fdr').loc[rows, cols]
    c.sd(r[['cardiac', 'cardiac_label', 'brain', 'brain_label', 'beta', 'lo', 'hi', 'p', 'fdr', 'n']],
         'Standardised beta per 1 SD cardiac trait, fully adjusted (L5); FDR across the matrix; '
         'bubble area proportional to |beta|, outline = FDR < 0.05')
    vmax = 0.15; smax = 78.0
    ax = c.ax(31, 9, 57, 37.5)
    yy, xx = np.meshgrid(np.arange(len(rows)), np.arange(len(cols)), indexing='ij')
    for i in range(len(rows)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
    v = B.values; q = Q.values
    sz = np.clip(np.abs(v) / vmax, 0.02, 1) * smax
    sig = q < 0.05
    ax.scatter(xx[~sig], yy[~sig], s=sz[~sig], c=v[~sig], cmap=CMAP_DIV, vmin=-vmax, vmax=vmax, lw=0, zorder=2)
    sc = ax.scatter(xx[sig], yy[sig], s=sz[sig], c=v[sig], cmap=CMAP_DIV, vmin=-vmax, vmax=vmax,
                    edgecolors=COL['ink'], linewidths=0.45, zorder=3)
    ax.axvline(len(GREY) - 0.5, color=COL['ink'], lw=0.5)
    ax.set_xlim(-0.6, len(cols) - 0.4); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([CARD_EN[x] for x in rows], fontsize=5.6)
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([BRAIN_EN[x] for x in cols], rotation=45, ha='left',
                                                        rotation_mode='anchor', fontsize=5.6)
    for tl, k in zip(ax.get_xticklabels(), cols):
        tl.set_color(COL['brain'] if k in WHITE else COL['base'])
    ax.tick_params(length=0, pad=1.2)
    for s in ax.spines.values():
        s.set_visible(False)
    g = c.board(31, 2.6, 57, 5.6)
    w = 57 / (len(cols) + 0.2)
    for x0, n, lab, col in [(0, len(GREY), 'Grey matter & CSF', COL['base']), (len(GREY), len(WHITE), 'White matter', COL['brain'])]:
        g.plot([(x0 + 0.1) * w + 0.8, (x0 + n + 0.1) * w - 0.8], [4.6, 4.6], color=col, lw=0.9, solid_capstyle='butt')
        g.text((x0 + n / 2 + 0.1) * w, 3.7, lab, ha='center', va='top', fontsize=5.8, color=col, fontweight='bold')
    heat_colorbar(c, sc, 96, 40, 22, 1.5, 'β per SD (L5)', ticks=[-0.15, 0, 0.15], ticklabels=['−0.15', '0', '0.15'])
    lg = c.board(94, 6, 29, 28)
    lg.text(1, 26.5, '|β| (bubble area)', fontsize=5.6, va='top')
    for k, val in enumerate([0.05, 0.10, 0.15]):
        yk = 20.5 - k * 5.2
        lg.scatter([4.5], [yk], s=np.clip(val / vmax, 0.02, 1) * smax, c='white', edgecolors=COL['mid'], linewidths=0.5)
        lg.text(9, yk, f'{val:.2f}', fontsize=5.4, va='center')
    lg.scatter([4.5], [3.2], s=40, c='white', edgecolors=COL['ink'], linewidths=0.6)
    lg.text(9, 3.2, 'outlined: FDR < 0.05', fontsize=5.4, va='center')


# ------------------------------------------------------------------ c  adjustment cascade
@panel(F, 'c', 72, 64, 'Stable after adjustment and replicated')
def cc(c):
    d = t('R', 'R1d_imagingage_hierarchical')
    outs = [('BrainAxis', 'Brain ageing axis', (0.0, 0.245)), ('FA', 'FA', (-0.2, 0.0)), ('logWMH', 'WMH (log)', (0.0, 0.245))]
    lev = list(dict.fromkeys(d.level)); xs = np.arange(len(lev))
    c.sd(sel(d, outcome=[o[0] for o in outs])[['outcome', 'level', 'cohort', 'beta', 'lo', 'hi', 'p', 'n', 'I2']],
         'Standardised beta per 1 SD cardiac axis by adjustment level (L0–L5) and cohort; Meta = fixed-effect')
    for j, (o, lab, yl) in enumerate(outs):
        ax = c.ax(8.5 + j * 22.8, 13, 16.2, 42)
        m = sel(d, outcome=o, cohort='Meta').set_index('level').loc[lev]
        col = COL['joint'] if o == 'BrainAxis' else COL['brain']
        ax.fill_between(xs, m.lo, m.hi, color=col, alpha=0.18, lw=0, zorder=1)
        for coh, ls, mk, lw in [('Discovery', (0, (2.0, 1.2)), 'o', 0.5), ('Replication', '-', 's', 0.5)]:
            s = sel(d, outcome=o, cohort=coh).set_index('level').loc[lev]
            ax.plot(xs, s.beta, color=COL['mid'], ls=ls, lw=lw, zorder=2)
            ax.plot(xs, s.beta, marker=mk, ms=2.0, mfc='white', mec=COL['mid'], mew=0.5, ls='none', zorder=3)
        ax.plot(xs, m.beta, color=col, lw=1.0, zorder=4)
        ax.plot(xs, m.beta, marker='D', ms=2.4, mfc=col, mec=col, ls='none', zorder=5)
        refline(ax, 0, orient='h')
        ax.set_ylim(*yl)
        tk = [-0.2, -0.1, 0] if o == 'FA' else [0, 0.1, 0.2]
        ax.set_yticks(tk); ax.set_yticklabels([f'{v:g}'.replace('-', '−') for v in tk])
        ax.set_xlim(-0.4, len(lev) - 0.6)
        ax.set_xticks(xs); ax.set_xticklabels([str(x).split()[0][:2] for x in lev], fontsize=5.2)
        ax.set_title(lab, fontsize=6, fontweight='normal', loc='center', pad=2, color=col)
        if j == 0:
            ax.set_ylabel('β per SD cardiac axis', labelpad=1)
            b0, b5 = m.beta.iloc[0], m.beta.iloc[-1]
            ax.annotate(f'{b0:.3f}', (0, b0), xytext=(0.5, b0 + 0.022), fontsize=5.2, ha='left', color=col)
            ax.annotate(f'{b5:.3f}', (len(lev) - 1, b5), xytext=(len(lev) - 1.2, b5 - 0.045), fontsize=5.2,
                        ha='center', color=col)
    c.text(41, 8.2, 'Adjustment level', ha='center', fontsize=6.2)
    c.legend(legend_handles([('mk', 'Discovery', COL['mid'], 'o', False), ('mk', 'Replication', COL['mid'], 's', False),
                             ('mk', 'Meta (95% CI band)', COL['base'], 'D', True)]), 2, 0.4, ncol=3, fs=5.3,
             columnspacing=0.6)


# ------------------------------------------------------------------ d  tract-level FA on the JHU atlas
def _fa_tracts():
    fa = sel(t('R', 'R2c_regional_tract_cortex'), category='WM tract FA').copy()
    fa['stem'] = fa.name.str.replace('IDP_dMRI_TBSS_FA_', '', regex=False)
    fa['jhu_label'] = fa.stem.map(bm.STEM2LAB)
    assert fa.jhu_label.notna().all() and len(fa) == 48
    return fa


@panel(F, 'd', 111, 64, 'Lower FA across deep white-matter tracts')
def d(c):
    fa = _fa_tracts()
    sigd = {int(r.jhu_label): float(r.beta) for _, r in fa.iterrows() if r.fdr < 0.05}
    ns = {int(r.jhu_label) for _, r in fa.iterrows() if r.fdr >= 0.05}
    vmin = -0.13
    s_mm = 0.211
    lab, mask = bm.atlas()
    views = []
    for v in ['sagittal', 'coronal', 'axial']:
        img, nsm, br = bm.project(sigd, ns, v)
        rr, cc_ = np.where(br)
        r0, r1, c0, c1 = max(rr.min() - 2, 0), rr.max() + 3, max(cc_.min() - 2, 0), cc_.max() + 3
        views.append((v, img[r0:r1, c0:c1], nsm[r0:r1, c0:c1], br[r0:r1, c0:c1], (r0, c0)))
    x = 2.0; gap = 2.6
    from scipy.ndimage import gaussian_filter
    for v, img, nsm, br, (r0, c0) in views:
        hh, ww = img.shape[0] * s_mm, img.shape[1] * s_mm
        ax = c.ax(x, 13, ww, hh)
        ax.imshow(np.where(br, 1.0, np.nan), cmap=LinearSegmentedColormap.from_list('bg', ['#EEF0F3', '#EEF0F3']),
                  origin='lower', interpolation='nearest', vmin=0, vmax=1)
        ax.imshow(np.where(nsm & np.isnan(img), 1.0, np.nan),
                  cmap=LinearSegmentedColormap.from_list('ns', [COL['ctrl'], COL['ctrl']]), origin='lower',
                  interpolation='nearest', vmin=0, vmax=1, alpha=0.75)
        im = ax.imshow(img, cmap=CMAP_FA, vmin=vmin, vmax=0, origin='lower', interpolation='nearest')
        ax.contour(gaussian_filter(br.astype(float), 1.2), levels=[0.5], colors=['#9AA3AE'], linewidths=0.45)
        ax.set_axis_off()
        ax.text(0.5, 1.02, v.capitalize(), transform=ax.transAxes, ha='center', va='bottom', fontsize=5.6, color=COL['mid'])
        if v == 'sagittal':
            ax.text(0.0, 0.02, 'P', transform=ax.transAxes, fontsize=5.2, color=COL['mid'])
            ax.text(1.0, 0.02, 'A', transform=ax.transAxes, fontsize=5.2, color=COL['mid'], ha='right')
            cx, cy = bm.centroid(6, 'sagittal')
            ax.annotate('Fornix\n(spared)', (cx - c0, cy - r0), xytext=(0.52, 0.93), textcoords='axes fraction',
                        fontsize=5.0, ha='center', va='top', color=COL['base'],
                        arrowprops=dict(arrowstyle='-', lw=0.35, color=COL['base'], shrinkA=0, shrinkB=0))
        else:
            ax.text(0.0, 0.02, 'L', transform=ax.transAxes, fontsize=5.2, color=COL['mid'])
            ax.text(1.0, 0.02, 'R', transform=ax.transAxes, fontsize=5.2, color=COL['mid'], ha='right')
        if v == 'axial':
            top_lab = int(_fa_tracts().sort_values('beta').iloc[0].jhu_label)
            cx, cy = bm.centroid(top_lab, 'axial')
            ax.annotate('PCR', (cx - c0, cy - r0), xytext=(-0.12, 0.33), textcoords='axes fraction', fontsize=5.2,
                        ha='center', va='center', color=COL['ink'],
                        arrowprops=dict(arrowstyle='-', lw=0.35, color=COL['ink'], shrinkA=0, shrinkB=0))
        x += ww + gap
    heat_colorbar(c, im, 5, 6.2, 30, 1.5, '', ticks=[-0.12, -0.08, -0.04, 0], ticklabels=['−0.12', '−0.08', '−0.04', '0'])
    c.text(36.5, 6.9, 'FA β per SD cardiac axis', fontsize=5.6, va='center')
    nsig = len(sigd); nneg = sum(1 for v_ in sigd.values() if v_ < 0)
    top = fa.loc[fa.beta.idxmin()]
    c.text(5, 1.0, f'{nsig}/48 tracts FDR < 0.05 ({nneg} lower FA); largest: posterior corona radiata '
                   f'(β = {top.beta:.3f})'.replace('= -', '= −'), fontsize=5.4, color=COL['brain'])
    c.legend(legend_handles([('patch', f'FDR ≥ 0.05 (n = {len(ns)})', COL['ctrl'])]), 110, 5.0, loc='lower right', fs=5.4)
    c.sd(fa[['IDP', 'name', 'stem', 'jhu_label', 'beta', 'lo', 'hi', 'p', 'fdr', 'n']],
         'Tract-level FA (JHU ICBM-DTI-81 labels, TBSS) per 1 SD cardiac axis, L5; FDR across 210 regional IDPs. '
         'Atlas used for display only: maximum-|beta| projection of significant tracts onto the MNI152 template')


# ------------------------------------------------------------------ e  regional distributions
@panel(F, 'e', 64, 56, 'Deep white matter, not cortex')
def e(c):
    r = t('R', 'R2c_regional_tract_cortex')
    cats = [('WM tract FA', 'FA', COL['brain']), ('WM tract diffusivity', 'MD', COL['brain']),
            ('WM tract ISOVF', 'ISOVF', COL['brain']), ('cortical thickness', 'Cortical\nthickness', COL['base'])]
    ax = c.ax(15, 10, 33, 38)
    rec = []
    for i, (k, lab, col) in enumerate(cats):
        s = sel(r, category=k)
        off = swarm(s.beta.values, 0.0042, 0.105, 0.43)
        sg = (s.fdr < 0.05).values
        ax.scatter(s.beta[sg], i + off[sg], s=4.2, c=col, lw=0, zorder=3)
        ax.scatter(s.beta[~sg], i + off[~sg], s=4.2, facecolors='white', edgecolors=col, linewidths=0.4, zorder=3)
        npos = int(((s.fdr < 0.05) & (s.beta > 0)).sum()); nneg = int(((s.fdr < 0.05) & (s.beta < 0)).sum())
        ax.text(1.03, i, f'{int(sg.sum())}/{len(s)}', transform=ax.get_yaxis_transform(), fontsize=5.6,
                va='bottom', ha='left', color=col, fontweight='bold')
        ax.text(1.03, i + 0.05, f'{nneg} −, {npos} +', transform=ax.get_yaxis_transform(), fontsize=5.0,
                va='top', ha='left', color=COL['mid'])
        for (_, x_), o in zip(s.iterrows(), off):
            rec.append(dict(category=k, name=x_['name'], beta=x_.beta, lo=x_.lo, hi=x_.hi, p=x_.p, fdr=x_.fdr, n=x_.n))
    refline(ax, 0)
    ax.set_ylim(len(cats) - 0.5, -0.55)
    clean_y(ax, [x[1] for x in cats], fs=5.6, color=[x[2] for x in cats])
    ax.set_xlim(-0.14, 0.12); ax.set_xticks([-0.1, 0, 0.1])
    ax.set_xlabel('β per SD cardiac axis (L5)')
    c.text(55.5, 49.5, 'FDR < 0.05', fontsize=5.2, ha='center', color=COL['mid'])
    c.legend(legend_handles([('pt', 'FDR < 0.05', COL['base'], True), ('pt', 'FDR ≥ 0.05', COL['base'], False)]),
             1, 0.4, ncol=2, fs=5.3)
    c.sd(pd.DataFrame(rec), 'All 210 regional IDPs per 1 SD cardiac axis (L5): 48 tracts each for FA, MD and ISOVF, '
                            '66 Desikan cortical-thickness regions; FDR across the 210 IDPs')


# ------------------------------------------------------------------ f  head- and body-size sensitivity
@panel(F, 'f', 69, 56, 'Not explained by head or body size')
def f(c):
    h = t('R', 'R2_headsize_bodysize_sensitivity')
    br = list(dict.fromkeys(h.brain))
    en = {'主脑轴（头颅归一化体积）': 'Brain axis (head-size normalised)', 'ICV残差法脑轴': 'Brain axis (ICV residualised)',
          '原始灰质体积 + ICV + ICV²': 'Raw GMV + ICV + ICV²', '原始白质体积 + ICV + ICV²': 'Raw WMV + ICV + ICV²',
          '原始海马体积 + ICV + ICV²': 'Raw hippocampus + ICV + ICV²', '归一化灰质体积': 'Normalised GMV',
          '白质 FA': 'FA', '白质 MD': 'MD', '白质 ISOVF': 'ISOVF', '白质高信号(log)': 'WMH (log)'}
    wmk = ['白质 FA', '白质 MD', 'ISOVF', '白质高信号']
    ax = c.ax(35, 10, 31, 38)
    rec = []
    for i, b_ in enumerate(br):
        wm = any(s in b_ for s in wmk)
        col = COL['brain'] if wm else COL['base']
        for dy, ck, filled in [(-0.18, '主心脏轴', True), (0.18, '体型无关心脏轴', False)]:
            r = one(h, cardiac=ck, brain=b_)
            ci_point(ax, r.beta, i + dy, r.lo, r.hi, col, filled=filled, ms=2.5, lw=0.6)
            rec.append(dict(r, brain_en=en[b_], cardiac_en='Cardiac axis' if filled else 'Body-size-free cardiac axis'))
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=COL['light'], lw=0, zorder=0)
    ax.set_ylim(len(br) - 0.5, -0.5)
    clean_y(ax, [en[b_] for b_ in br], fs=5.5, color=[COL['brain'] if any(s in b_ for s in wmk) else COL['ink'] for b_ in br])
    refline(ax, 0)
    ax.set_xlim(-0.12, 0.11); ax.set_xticks([-0.1, 0, 0.1])
    ax.set_xlabel('β per SD (95% CI)')
    c.legend(legend_handles([('pt', 'Cardiac axis', COL['base'], True), ('pt', 'Body-size-free axis', COL['base'], False)]),
             2, 0.4, ncol=2, fs=5.3)
    c.sd(pd.DataFrame(rec), 'Sensitivity to head-size handling and to a cardiac axis built without body-size-dependent traits (L5)')


# ------------------------------------------------------------------ g  age equivalence
@panel(F, 'g', 50, 56, 'Years of white-matter ageing per SD')
def g(c):
    gq = t('R', 'R1g_imagingage_age_equivalence')
    order = [('FA', 'FA'), ('MD', 'MD'), ('ISOVF', 'ISOVF'), ('logWMH', 'WMH'), ('BrainAxis_res', 'Brain axis*')]
    ax = c.ax(15, 13, 31, 35)
    for i, (k, lab) in enumerate(order):
        for dy, ax_k, filled in [(-0.18, 'CardiacAxis', True), (0.18, 'CardiacAxis_sizefree', False)]:
            r = one(gq, cardiac=ax_k, outcome=k)
            ax.barh(i + dy, r.age_equiv_years, height=0.32, color=COL['heart'] if filled else 'white', ec=COL['heart'],
                    lw=0.6, zorder=2)
            ax.plot([r.lo, r.hi], [i + dy, i + dy], color=COL['ink'], lw=0.6, zorder=3)
            if k == 'FA' and filled:
                ax.text(r.hi + 0.08, i + dy, f'{r.age_equiv_years:.2f} y', fontsize=5.3, va='center', color=COL['heart'])
    ax.set_ylim(len(order) - 0.5, -0.6)
    clean_y(ax, [o[1] for o in order], fs=5.6)
    ax.set_xlim(0, 3.6); ax.set_xticks([0, 1, 2, 3])
    ax.grid(axis='x', color=COL['grid'], lw=0.4, zorder=0)
    ax.set_xlabel('Age-equivalent years (95% CI)')
    c.legend(legend_handles([('bar', 'Cardiac axis', COL['heart'], True), ('bar', 'Body-size-free axis', COL['heart'], False)]),
             49, 13.6, loc='lower right', ncol=1, fs=5.2, handlelength=1.0, handletextpad=0.35)
    c.text(49, 49.6, '*ICV-residualised', fontsize=5.0, ha='right', color=COL['mid'])
    c.sd(gq, 'Age-equivalent years = beta per SD cardiac axis / beta per year of age; 95% CI by bootstrap')


LAYOUT = [(F, 'a', 0, 0), (F, 'b', 60, 0), (F, 'c', 0, 62), (F, 'd', 72, 62),
          (F, 'e', 0, 126), (F, 'f', 64, 126), (F, 'g', 133, 126)]
HEIGHT = 182
