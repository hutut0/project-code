"""Figure 1 — Study design and the shared-load, asymmetric-consequence model.

a  cohort and six evidence layers (counts read from result tables)
b  four questions (v3) × six evidence layers (QUALITATIVE: encodes framework v4 tables 0.2/2.2; recorded as such)
c  30 pre-specified outcomes and their events (pyCirclize; A1 + B1_followup_events + A5 tiers)
d  conceptual model; every number on an arrow is read from a frozen result file
"""
import numpy as np
import pandas as pd
from nclib import (panel, COL, ORGAN_COL, legend_handles, rounded_box, arrow, fmt_ci)
from nc_data import t, sel, one, OUT_SHORT, OUT_ORDER, outcome_dict

F = 'Fig1'

# ------------------------------------------------------------------ a  cohort and evidence layers
LAYER = [('L-I', 'Imaging', COL['joint']), ('L-M', 'Molecular', '#7A8B99'), ('L-D', 'Disease', COL['heart']),
         ('L-P', 'Pathway', '#5B8C85'), ('L-G', 'Genetic', COL['brain']), ('L-C', 'Clinical', COL['bp'])]


@panel(F, 'a', 183, 44, 'One cohort with heart and brain MRI, six independent evidence layers')
def a(c):
    T1 = t('R', 'Table1_baseline_characteristics').set_index('变量')
    S0 = t('R', 'S0_dates_summary').iloc[0]
    L = t('R', 'R1_axis_loadings'); info = t('R', 'R1_axis_info').set_index('axis')
    R2c = t('R', 'R2c_regional_tract_cortex'); r4e = t('R', 'R4e_proteome_scan_sizefree_cardiac_axis')
    r4c = t('R', 'R4c_organ_age_gap_vs_imaging'); o = outcome_dict()
    n_all, n_dis, n_rep = (T1.loc['人数', k] for k in ['全部', '发现队列', '复制队列'])
    n_cmr = int((L.axis == '心脏重构轴').sum()); n_bax = int((L.axis == '脑结构老化轴').sum())
    n_tr = int((R2c.category == 'WM tract FA').sum()); n_cx = int((R2c.category == 'cortical thickness').sum())
    n_prot, n_pn = len(r4e), int(r4e.n.max()); n_org = r4c.age_gap.nunique()
    ve = info.loc['心脏重构轴']
    fu_lo, fu_hi = o.median_fu.min(), o.median_fu.max()
    bd = c.board(0, 0, 183, 39)
    # cohort box
    rounded_box(bd, 1.5, 3, 37, 32, fc=COL['light'], ec=COL['base'], lw=0.5)
    bd.text(20, 31.2, 'UK Biobank imaging visit', ha='center', va='top', fontsize=6.2, fontweight='bold')
    bd.text(20, 27.0, f"{S0.imaging_first[:4]}–{S0.imaging_last[:4]}", ha='center', va='top', fontsize=5.6, color=COL['mid'])
    bd.text(20, 22.6, f'N = {n_all}', ha='center', va='top', fontsize=8.5, fontweight='bold', color=COL['ink'])
    bd.text(20, 15.6, f'heart + brain MRI', ha='center', va='top', fontsize=5.6, color=COL['base'])
    bd.text(20, 11.6, f'Discovery {n_dis} · Replication {n_rep}', ha='center', va='top', fontsize=5.4, color=COL['base'])
    bd.text(20, 7.6, f'median follow-up {fu_lo:.1f}–{fu_hi:.1f} y', ha='center', va='top', fontsize=5.4, color=COL['base'])
    arrow(bd, (39.5, 19), (44.5, 19), color=COL['base'], lw=0.8, head=3.5)
    body = {
        'L-I': [f'{n_cmr} cardiac MRI traits', f'axis: {ve.var_exp_discovery * 100:.1f}% variance',
                f'{n_bax}-IDP brain axis', f'{n_tr} tracts · {n_cx} cortices'],
        'L-M': ['blood counts · CRP', f'{n_prot:,} Olink proteins', f'(n = {n_pn:,})', f'{n_org} organ-age clocks'],
        'L-D': [f'{len(o)} pre-specified', 'outcomes: cardiac,', 'brain, composite,', 'negative controls'],
        'L-P': ['same-day', 'white-matter block', 'incident cardiac', 'events (time-varying)'],
        'L-G': ['GWAS: 82 CMR,', 'BIG40 white matter,', 'stroke ×2, HF, AF,', 'CAD, AD, PD, BP'],
        'L-C': ['risk-factor model', 'PREVENT (baseline)', 'out-of-fold ΔC, NRI', 'calibration, DCA'],
    }
    w, gap, x0 = 21.6, 1.5, 45.5
    for i, (code, name, col) in enumerate(LAYER):
        x = x0 + i * (w + gap)
        rounded_box(bd, x, 3, w, 32, fc='white', ec=col, lw=0.6)
        rounded_box(bd, x, 28.5, w, 6.5, fc=col, ec=col, lw=0.6)
        bd.text(x + w / 2, 31.75, f'{code}  {name}', ha='center', va='center', fontsize=5.9, color='white',
                fontweight='bold')
        for j, s in enumerate(body[code]):
            bd.text(x + w / 2, 24.6 - j * 5.0, s, ha='center', va='center', fontsize=5.1, color=COL['ink'])
    c.sd(pd.DataFrame([dict(item='participants (all / discovery / replication)', value=f'{n_all} / {n_dis} / {n_rep}', source='Table1'),
                       dict(item='imaging dates', value=f'{S0.imaging_first} to {S0.imaging_last}', source='S0_dates_summary'),
                       dict(item='cardiac MRI traits in cardiac axis', value=n_cmr, source='R1_axis_loadings'),
                       dict(item='cardiac axis variance explained (discovery / replication)',
                            value=f'{ve.var_exp_discovery:.4f} / {ve.var_exp_replication:.4f}', source='R1_axis_info'),
                       dict(item='brain-ageing axis IDPs', value=n_bax, source='R1_axis_loadings'),
                       dict(item='DTI tracts / cortical regions', value=f'{n_tr} / {n_cx}', source='R2c'),
                       dict(item='Olink proteins (n)', value=f'{n_prot} ({n_pn})', source='R4e'),
                       dict(item='organ-age clocks', value=n_org, source='R4c'),
                       dict(item='pre-specified outcomes', value=len(o), source='A1_outcome_dictionary'),
                       dict(item='median follow-up range across outcomes (y)', value=f'{fu_lo:.2f}–{fu_hi:.2f}',
                            source='B1_followup_events')]),
         'Counts shown in panel a (all read from result tables)')


# ------------------------------------------------------------------ b  sub-questions × layers (qualitative)
# Figure references: in-house numbering (Fig1 = design) by default; HB_FIG_NUMBERING=submission gives the
# manuscript numbering, in which the design figure moves to Supplementary Fig. 1 and Figs 2–8 become Figs 1–7.
_SUB = __import__('os').environ.get('HB_FIG_NUMBERING', '') == 'submission'
_FR = (['Fig. 1', 'Figs 2–4', 'Fig. 5', 'Figs 6–7'] if _SUB else ['Fig. 2', 'Figs 3–5', 'Fig. 6', 'Figs 7–8'])
QS = [('Q1', 'Association', _FR[0], 'specific to deep white\nmatter; beyond measured\nrisk factors'),
      ('Q2', 'Mutual influence', _FR[1], 'shared upstream load\n(BP); asymmetric\ndisease consequences'),
      ('Q3', 'Causality', _FR[2], 'WMH → IS triangulated\n(MR grade D); cardiac\naxis inconclusive'),
      ('Q4', 'Clinical value', _FR[3], 'incremental value;\nrecalibration required')]
# analyses per layer, transcribed from framework v4 tables 2.2 / 2.3 (qualitative design information);
# v3 manuscript structure: the former mechanism and mutual-influence sub-questions are merged into Q2.
QMAP = {'Q1': {'L-I': ['R1 hierarchical adjustment', 'R1 internal replication', 'R2 tract localisation',
                       'R2 body-size-free axis', 'R1 12 sensitivity analyses']},
        'Q2': {'L-M': ['R3 inflammation', 'R4 organ ages', 'R4 proteome', 'R5 glial markers'],
               'L-G': ['LDSC genetic correlation', 'BP MVMR'],
               'L-D': ['B1 30-outcome matrix', 'B3 vascular gradient', 'C2 trait atlas', 'C6 dose–response', 'B4 joint effects'],
               'L-P': ['B2 cross-organ sensitivity', 'C3 pathway decomposition']},
        'Q3': {'L-G': ['S3/S9 univariable MR', 'M5 independent replication', 'MVMR (cardiac, BP)', 'reverse MR',
                       'M1 full 82-trait axis', 'G1–G6 disease-level bidirectional MR']},
        'Q4': {'L-C': ['C1 out-of-fold ΔC', 'B5 PREVENT comparison', 'B5 NRI and decision curves', 'B5 calibration',
                       'C5 reclassification', 'C4 subgroups', 'C7 E-values'],
               'L-D': ['B4 joint groups']}}


@panel(F, 'b', 100, 62, 'Four questions answered by independent evidence layers')
def b(c):
    bd = c.board(0, 0, 100, 56)
    xs = {code: 26 + i * 6.6 for i, (code, _, _) in enumerate(LAYER)}
    ytop = 46
    for code, name, col in LAYER:
        bd.text(xs[code], ytop + 3.2, code, ha='center', va='bottom', fontsize=5.4, color=col, fontweight='bold')
    bd.text(67.5, ytop + 3.2, 'Shown in', ha='left', va='bottom', fontsize=5.2, color=COL['mid'])
    bd.text(77.5, ytop + 3.2, 'Answer', ha='left', va='bottom', fontsize=5.2, color=COL['mid'])
    rec = []
    for i, (q, name, fig_, ans) in enumerate(QS):
        y = ytop - i * 11.2 - 1.0
        if i % 2 == 0:
            bd.add_patch(__import__('matplotlib.patches', fromlist=['Rectangle']).Rectangle((0.5, y - 5.3), 99, 10.6,
                         fc=COL['light'], ec='none', zorder=0))
        bd.text(1.5, y, f'{q}', ha='left', va='center', fontsize=6.0, fontweight='bold', color=COL['ink'])
        bd.text(7.0, y, name, ha='left', va='center', fontsize=5.6, color=COL['ink'])
        for code, _, col in LAYER:
            n = len(QMAP[q].get(code, []))
            primary = n >= 2
            if n:
                r = 0.9 + 0.32 * n
                bd.add_patch(__import__('matplotlib.patches', fromlist=['Circle']).Circle(
                    (xs[code], y), r, fc=col if primary else 'white', ec=col, lw=0.7, zorder=3))
                if primary:
                    bd.text(xs[code], y, str(n), ha='center', va='center', fontsize=5.0, color='white',
                            fontweight='bold', zorder=4)
            else:
                bd.plot(xs[code], y, 'o', ms=1.2, color=COL['ctrl_l'], zorder=2)
            for a_ in QMAP[q].get(code, []):
                rec.append(dict(sub_question=f'{q} {name}', layer=code, analysis=a_, role='primary' if primary else 'supporting'))
        bd.text(67.5, y, fig_, ha='left', va='center', fontsize=5.4, color=COL['base'])
        bd.text(77.5, y, ans, ha='left', va='center', fontsize=5.0, color=COL['ink'], linespacing=1.0)
    c.legend(legend_handles([('pt', 'primary layer (number = analyses)', COL['base'], True),
                             ('pt', 'supporting layer', COL['base'], False)]), 1, 0.3, ncol=2, fs=5.1)
    c.sd(pd.DataFrame(rec), 'QUALITATIVE design information transcribed from the v4 framework (tables 0.2, 2.2, 2.3); '
                            'not a computed result')


# ------------------------------------------------------------------ c  30 outcomes (pyCirclize)
ABBR = {'P1_major_neurological': 'P1', 'P2_major_cardiac': 'P2', 'MACE': 'MACE', 'Heart_brain_multimorbidity': 'H–B',
        'Stroke': 'Stroke', 'Ischaemic_stroke': 'IS', 'ICH': 'ICH', 'TIA': 'TIA', 'Vascular_dementia': 'VaD',
        'Dementia': 'Dementia', 'Alzheimers_disease': 'AD', 'Parkinsons_disease': 'PD', 'Depression': 'Depression',
        'Anxiety_disorder': 'Anxiety', 'Bipolar_disorder': 'Bipolar', 'Schizophrenia': 'SCZ', 'Epilepsy': 'Epilepsy',
        'Sleep_apnoea': 'OSA', 'Insomnia_other_sleep': 'Insomnia', 'Migraine': 'Migraine', 'Multiple_sclerosis': 'MS',
        'Heart_failure': 'HF', 'AF': 'AF', 'Ventricular_arrhythmia': 'VA', 'MI': 'MI', 'CHD': 'CHD',
        'Cardiomyopathy': 'CMP', 'Aortic_stenosis': 'AS', 'NC_inguinal_hernia': 'Hernia', 'NC_senile_cataract': 'Cataract'}
FAM_SHORT = {'Composite': 'Composite', 'Cerebrovascular': 'Cerebro-\nvascular', 'Neurodegenerative': 'Neuro-\ndegen.',
             'Psychiatric': 'Psychiatric', 'Other neurological': 'Other\nneuro', 'Heart failure': 'HF',
             'Arrhythmia': 'Arrhythmia', 'Coronary': 'Coronary', 'Structural': 'Structural',
             'Negative control': 'Controls'}


@panel(F, 'c', 83, 62, '30 pre-specified outcomes, locked before analysis')
def cpanel(c):
    from pycirclize import Circos
    o = outcome_dict().copy()
    tiers = t('A', 'A5_power_feasibility_tiers').set_index('key')['feasibility_tier']
    o['tier'] = o.key.map(tiers)
    fam_order = list(dict.fromkeys(o.family_en))
    sectors = {f: int((o.family_en == f).sum()) for f in fam_order}
    circos = Circos(sectors, start=10, end=350, space=2.2)
    vmax = np.log10(2000)
    for sec in circos.sectors:
        ks = o[o.family_en == sec.name].sort_values('order')
        x = np.arange(len(ks)) + 0.5
        tr = sec.add_track((44, 76))
        tr.axis(fc=COL['light'], ec='none')
        for ref in (10, 100, 1000):
            tr.line([0, len(ks)], [np.log10(ref)] * 2, vmax=vmax, lw=0.3, color='#C9CED6', ls=(0, (2, 1.5)))
        cols, fcs = [], []
        for _, r in ks.iterrows():
            oc = ORGAN_COL[r['器官']]
            cols.append(oc)
            if r.events < 20:
                fcs.append('white')
            elif str(r.tier).startswith('仅描述'):
                fcs.append(ORGAN_COL[r['器官']] + '66')
            else:
                fcs.append(oc)
        tr.bar(x, np.log10(ks.events.clip(lower=1)), width=0.72, vmax=vmax, color=fcs, ec=cols, lw=0.5)
        for xi, (_, r) in zip(x, ks.iterrows()):
            sec.text(ABBR[r.key], x=xi, r=79, size=5.0, orientation='vertical', color=ORGAN_COL[r['器官']], ha='left'
                     if False else 'center', va='center', adjust_rotation=True)
        inner = sec.add_track((36, 42))
        inner.axis(fc=COL['ctrl_l'], ec='none')
    ax = c.ax(4, 2.5, 54, 54, projection='polar')
    circos.plotfig(ax=ax)
    # radial reference labels in the top gap
    for ref, rr in zip((10, 100, 1000), [44 + 32 * np.log10(v) / vmax for v in (10, 100, 1000)]):
        ax.text(0, rr / 100 * ax.get_rmax() if False else rr, f'{ref:,}', fontsize=5.0, ha='center', va='center',
                color=COL['mid'])
    c.text(61, 50.5, 'Bar height: events\n(log scale; reference\nrings 10 / 100 / 1,000)\n\nGaps separate 10\npre-specified disease\nfamilies (Source Data)', fontsize=5.0, color=COL['mid'],
           ha='left', va='top', linespacing=1.15)
    from matplotlib.patches import Patch
    hs = [Patch(fc=COL['heart'], ec=COL['heart'], label='Cardiac'), Patch(fc=COL['brain'], ec=COL['brain'], label='Brain'),
          Patch(fc=COL['joint'], ec=COL['joint'], label='Heart–brain'), Patch(fc=COL['ctrl'], ec=COL['ctrl'], label='Control'),
          Patch(fc=COL['base'] + '66', ec=COL['base'], lw=0.5, label='descriptive tier'),
          Patch(fc='white', ec=COL['base'], lw=0.5, label='< 20 events\n(not modelled)')]
    c.legend(hs, 62, 2.0, ncol=1, fs=5.0, handlelength=0.9, handletextpad=0.35, labelspacing=0.3)
    sd = o[['key', 'short', '中文名', '器官', 'organ_en', 'family_en', 'events', 'n_at_risk', 'median_fu', 'person_years', 'tier']].copy()
    sd['abbreviation'] = sd.key.map(ABBR); sd['modelled'] = sd.events >= 20
    c.sd(sd, 'Outcome dictionary (A1), events and follow-up after imaging (B1_followup_events), power tier fixed before '
             'association analysis (A5)')


# ------------------------------------------------------------------ d  conceptual model
@panel(F, 'd', 183, 72, 'Shared load, asymmetric consequences: the working model with study estimates')
def d(c):
    B1 = t('B', 'B1_full_matrix_cox')
    def hr(o_, e_):
        return one(B1, outcome=o_, exposure=e_)
    R1g = t('R', 'R1g_imagingage_age_equivalence'); fa_age = one(R1g, cardiac='CardiacAxis', outcome='FA')
    fa = sel(t('R', 'R2c_regional_tract_cortex'), category='WM tract FA')
    n_neg = int(((fa.fdr < 0.05) & (fa.beta < 0)).sum())
    s3 = one(t('S', 'S3_univariable_summary'), exposure='WMH', outcome='Stroke')
    m5 = t('M5', 'M5_univariable_summary'); wmeg = one(m5, exposure='WMH', outcome='MEGA_AIS')
    b2 = sel(t('BP', 'B02_all_model_estimates'), method='IVW'); b5 = t('BP', 'B05_same_instrument_comparison_without_BP')
    wsbp = one(b2, model='WMH+SBP__Stroke', exposure='WMH')
    s4o = one(b5, model='CardiacAxisS4+SBP__Stroke', exposure='CardiacAxisS4')
    s4s = one(b2, model='CardiacAxisS4+SBP__Stroke', exposure='CardiacAxisS4')
    c3 = one(t('C', 'C3_pathway_decomposition'), outcome='Stroke', exposure='CardiacAxis', mediator='白质合并块')
    g6 = t('G', 'mvmr_SBP/G6_SBP_conditioned_MVMR')
    g6r = {o_: one(g6, model=f'WMH+SBP__{o_}', exposure='WMH', method='IVW_MVMR') for o_ in ['HF_HERMES', 'AF_Nielsen', 'CAD_Nikpay']}
    rev = one(t('S', 'S9_axis_univariable_summary'), exposure='WMH', outcome='CardiacAxisS4')
    C1 = t('C', 'C1_prediction_atlas'); m = C1[(C1.model != '基线') & (~C1.outcome.str.startswith('NC_'))]
    n_mod = m.outcome.nunique(); cnt = m[m.dC_FDR < 0.05].groupby('model').outcome.nunique()
    hs, hw, hhf = hr('Stroke', 'CardiacAxis'), hr('Ischaemic_stroke', 'logWMH'), hr('Heart_failure', 'CardiacAxis')
    hp2, hnc = hr('P2_major_cardiac', 'BrainAxis'), hr('NC_senile_cataract', 'BrainAxis')

    bd = c.board(0, 0, 183, 66)
    # nodes
    rounded_box(bd, 66, 52, 51, 11, fc=COL['bp_l'], ec=COL['bp'], lw=0.7)
    bd.text(91.5, 59.4, 'Shared upstream load', ha='center', va='center', fontsize=6.4, fontweight='bold')
    bd.text(91.5, 55.2, 'genetic evidence implicates blood pressure', ha='center', va='center', fontsize=5.5, color=COL['base'])
    rounded_box(bd, 6, 30, 50, 11, fc=COL['heart_l'], ec=COL['heart'], lw=0.7)
    bd.text(31, 37.4, 'Cardiac remodelling', ha='center', va='center', fontsize=6.4, fontweight='bold')
    bd.text(31, 33.2, '82-trait axis, hypertrophy-led', ha='center', va='center', fontsize=5.5, color=COL['base'])
    rounded_box(bd, 127, 30, 50, 11, fc=COL['brain_l'], ec=COL['brain'], lw=0.7)
    bd.text(152, 37.4, 'Deep white-matter injury', ha='center', va='center', fontsize=6.4, fontweight='bold')
    bd.text(152, 33.2, 'FA↓  ISOVF↑  WMH↑', ha='center', va='center', fontsize=5.5, color=COL['base'])
    rounded_box(bd, 6, 8, 50, 9, fc='white', ec=COL['heart'], lw=0.7)
    bd.text(31, 12.5, 'Cardiac disease (HF, AF, CHD)', ha='center', va='center', fontsize=6.2, fontweight='bold',
            color=COL['heart'])
    rounded_box(bd, 127, 8, 50, 9, fc='white', ec=COL['brain'], lw=0.7)
    bd.text(152, 12.5, 'Ischaemic stroke', ha='center', va='center', fontsize=6.2, fontweight='bold', color=COL['brain'])
    lab = dict(fontsize=5.1, ha='center', va='center', linespacing=1.15,
               bbox=dict(fc='white', ec='none', pad=0.4, alpha=0.92), zorder=6)
    # BP -> heart (genetic, same instruments)
    arrow(bd, (72, 52), (40, 41.3), color=COL['bp'], lw=0.9, head=4)
    bd.text(43, 49.5, f'genetic cardiac axis → stroke OR {np.exp(s4o.beta):.2f} → {s4s.OR:.2f}\n'
                      f'after conditioning on SBP  [G]', color=COL['bp'], **lab)
    # BP -> white matter (proposed)
    arrow(bd, (111, 52), (143, 41.3), color=COL['bp'], lw=0.9, ls=(0, (3, 2)), head=4)
    bd.text(141, 49.5, 'proposed; cumulative-BP test not possible\n(single BP measurement)', color=COL['bp'], **lab)
    # coupling
    arrow(bd, (56.5, 35.5), (126.5, 35.5), color=COL['joint'], lw=1.3, head=4.5)
    arrow(bd, (126.5, 35.5), (56.5, 35.5), color=COL['joint'], lw=1.3, head=4.5)
    bd.text(91.5, 40.2, f'coupling ≈ {fa_age.age_equiv_years:.1f} years of FA ageing per SD; '
                        f'FA lower in {n_neg}/{len(fa)} tracts  [O, replicated]', color=COL['joint'], **lab)
    bd.text(91.5, 31.0, f'MR heart → WM: direction-consistent, imprecise; WMH → heart β {rev.b_IVW:+.3f} SD  [G]'.replace('-', '−'),
            color=COL['joint'], **lab)
    # heart -> cardiac disease
    arrow(bd, (31, 30), (31, 17.3), color=COL['heart'], lw=1.3, head=4.5)
    bd.text(31, 23.5, f'HF HR {fmt_ci(hhf.HR, hhf.lo, hhf.hi)}  [O]', color=COL['heart'], **lab)
    # WM -> stroke (triangulated)
    arrow(bd, (152, 30), (152, 17.3), color=COL['brain'], lw=2.0, head=5)
    bd.text(149, 23.5, f'IS HR {hw.HR:.2f} · MR OR {np.exp(s3.b_IVW):.2f} (grade {s3.grade})\n'
                       f'MEGASTROKE {wmeg.OR_IVW:.2f} · | SBP {wsbp.OR:.2f}  [O, G, Rep]', color=COL['brain'],
            **dict(lab, ha='right'))
    # heart -> stroke (observational; MR inconclusive)
    arrow(bd, (56.5, 30.5), (126.5, 16.5), color=COL['heart'], lw=1.0, ls=(0, (3, 2)), head=4.5)
    bd.text(78, 26.3, f'stroke HR {hs.HR:.2f}; {c3.PM * 100:.1f}% via same-day WM block\n'
                      f'MR inconclusive (positive control failed)  [O, G]', color=COL['heart'], **lab)
    # WM -> cardiac disease (weak / null)
    arrow(bd, (126.5, 30.5), (56.5, 16.5), color=COL['mid'], lw=0.8, ls=(0, (1, 1.5)), head=4)
    bd.text(91.5, 12.2, f'brain axis → P2 HR {hp2.HR:.2f} ≈ negative control {hnc.HR:.2f}\n'
                       f'WMH | SBP → HF/AF/CAD OR {g6r["HF_HERMES"].OR:.2f}/{g6r["AF_Nielsen"].OR:.2f}/'
                       f'{g6r["CAD_Nikpay"].OR:.2f}  [O, G]', color=COL['mid'], **lab)
    # clinical strip
    rounded_box(bd, 6, 0.6, 171, 5.2, fc=COL['light'], ec='none')
    bd.text(91.5, 3.2, f'Clinical layer — out-of-fold ΔC with FDR < 0.05 in {n_mod} modelled outcomes:  cardiac MRI '
                       f'{cnt.get("+心脏MRI", 0)}  ·  brain MRI {cnt.get("+脑MRI", 0)}  ·  both {cnt.get("+心脏MRI+脑MRI", 0)}',
            ha='center', va='center', fontsize=5.4, color=COL['ink'])
    from matplotlib.lines import Line2D
    c.legend([Line2D([], [], color=COL['base'], lw=2.0, label='triangulated (MR grade D)'),
              Line2D([], [], color=COL['base'], lw=1.0, label='supported'),
              Line2D([], [], color=COL['base'], lw=1.0, ls=(0, (3, 2)), label='proposed / MR inconclusive'),
              Line2D([], [], color=COL['mid'], lw=0.8, ls=(0, (1, 1.5)), label='weak / no evidence')], 182, 71.6,
             loc='upper right', ncol=4, fs=5.0, columnspacing=0.8, handlelength=1.8)
    c.text(182, 67.6, '[O] observational  [G] genetic  [Rep] independent replication', fontsize=5.0, ha='right',
           va='top', color=COL['mid'])
    rows = [('Coupling, FA age-equivalent per SD (years)', fa_age.age_equiv_years, 'R1g'),
            ('Tracts with lower FA (FDR < 0.05) / total', f'{n_neg}/{len(fa)}', 'R2c'),
            ('Genetic cardiac axis → stroke OR, same instruments without SBP', np.exp(s4o.beta), 'B05'),
            ('Genetic cardiac axis → stroke OR, with SBP (IVW-MVMR)', s4s.OR, 'B02'),
            ('MR WMH → cardiac axis (beta, SD)', rev.b_IVW, 'S9_axis_univariable_summary'),
            ('Heart failure HR per SD cardiac axis', fmt_ci(hhf.HR, hhf.lo, hhf.hi), 'B1_full_matrix_cox'),
            ('Ischaemic stroke HR per SD WMH', fmt_ci(hw.HR, hw.lo, hw.hi), 'B1_full_matrix_cox'),
            ('MR WMH → ischaemic stroke OR (IVW) and grade', f'{np.exp(s3.b_IVW):.3f}; grade {s3.grade}', 'S3_univariable_summary'),
            ('MEGASTROKE any ischaemic stroke OR', wmeg.OR_IVW, 'M5_univariable_summary'),
            ('WMH → stroke OR conditional on SBP', wsbp.OR, 'B02'),
            ('Stroke HR per SD cardiac axis', fmt_ci(hs.HR, hs.lo, hs.hi), 'B1_full_matrix_cox'),
            ('Proportion via same-day white-matter block', fmt_ci(c3.PM, c3.PM_lo, c3.PM_hi, 3), 'C3_pathway_decomposition'),
            ('Brain axis → P2 HR', fmt_ci(hp2.HR, hp2.lo, hp2.hi), 'B1_full_matrix_cox'),
            ('Brain axis → cataract (negative control) HR', fmt_ci(hnc.HR, hnc.lo, hnc.hi), 'B1_full_matrix_cox')]
    rows += [(f'WMH | SBP → {k} OR', fmt_ci(v.OR, v.OR_lo, v.OR_hi), 'G6_SBP_conditioned_MVMR') for k, v in g6r.items()]
    rows += [(f'C1 outcomes with FDR < 0.05, {k}', int(v), 'C1_prediction_atlas') for k, v in cnt.items()]
    c.sd(pd.DataFrame(rows, columns=['quantity', 'value', 'source']), 'Every estimate printed on the model diagram')


LAYOUT = [(F, 'a', 0, 0), (F, 'b', 0, 44), (F, 'c', 100, 44), (F, 'd', 0, 106)]
HEIGHT = 178
