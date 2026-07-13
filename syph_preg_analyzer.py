"""
PregnancyLog — a per-pregnancy log for every pregnancy in the sim.

For each pregnancy, records:
    - conception, entry (when syph exposure begins), and delivery/loss timings
    - mother's syph status through pregnancy (stages seen, syph_positive_ever)
    - every syph ANC visit, syph test (positive/negative), and syph treatment
    - final delivery outcome, read from ti_{nnd,stillborn,congenital} rather
      than syph.cs_outcome — because successful maternal treatment cancels
      the ti_ events but leaves cs_outcome stale. Both are captured so the
      reversal is visible.
    - preterm (Pregnancy.preterm on newborn) and lbw (FetalHealth.lbw on
      newborn), read at delivery so timing-shift/growth-restriction reversal
      via sti_fetal is reflected.

Prenatal loss (fetus dies before delivery) is detected via
sim.people.ti_dead on the tracked fetus UID and classified by mother's
gestation at death (miscarriage <20w, stillbirth_prenatal >=20w).

Usage:
    from syph_preg_analyzer import PregnancyLog
    analyzer = PregnancyLog()
    sim = build_scenario_sim(..., extra_analyzers=[analyzer])
    sim.run()
    records = sim.analyzers['pregnancy_log'].to_records()
"""

import numpy as np
import starsim as ss


SYPH_TEST_NAMES = [
    'syph_symp_test', 'syph_symp_test_poc', 'syph_pn_test',
    'syph_rash_test', 'syph_anc_rpr', 'syph_anc_rdt', 'syph_anc_confirm',
]
SYPH_TX_NAME = 'syph_tx'
SYPH_ANC_TIMER_NAME = 'syph_anc_timer'
STAGE_ATTRS = ['primary', 'secondary', 'early', 'late']


def _stage_of(syph, uid):
    for s in STAGE_ATTRS:
        arr = getattr(syph, s, None)
        if arr is not None and bool(arr[uid]):
            return s
    return 'infected_unknown' if bool(syph.infected[uid]) else None


class PregnancyLog(ss.Analyzer):
    """Log every pregnancy from conception to close (delivery or loss)."""

    def __init__(self, name='pregnancy_log', **kwargs):
        super().__init__(name=name, **kwargs)
        self.records = {}   # mother_uid -> record dict

    def init_pre(self, sim):
        super().init_pre(sim)
        self._preg = sim.demographics.pregnancy
        self._syph = sim.diseases.syph
        self._fh = sim.custom['fetal_health']
        self._tests = {n: sim.interventions[n]
                       for n in SYPH_TEST_NAMES if n in sim.interventions}
        self._tx = sim.interventions.get(SYPH_TX_NAME)
        self._anc_timer = sim.interventions.get(SYPH_ANC_TIMER_NAME)
        self._mat_net = sim.networks.get('maternalnet')
        self._preg.add_delivery_callback(self._on_live_birth)

    def _find_current_fetus(self, mother_uid):
        if self._mat_net is None:
            return -1
        contacts = list(self._mat_net.find_contacts([int(mother_uid)]))
        if not contacts:
            return -1
        age = self.sim.people.age
        unborn = [int(u) for u in contacts if age[int(u)] < 0]
        return unborn[-1] if unborn else -1

    def _open(self, mother_uid, ti, syph_positive_at_entry):
        preg = self._preg
        mid = int(mother_uid)
        fetus_uid = self._find_current_fetus(mid)
        conception_ti = int(preg.ti_pregnant[mid]) if not np.isnan(preg.ti_pregnant[mid]) else int(ti)
        sched_del = preg.ti_delivery[mid]
        self.records[mid] = dict(
            mother_uid=mid,
            fetus_uid=fetus_uid,
            conception_ti=conception_ti,
            conception_year=float(self.sim.t.yearvec[conception_ti])
                if 0 <= conception_ti < self.sim.t.npts else float('nan'),
            scheduled_delivery_ti=int(sched_del) if not np.isnan(sched_del) else -1,
            syph_positive_at_entry=bool(syph_positive_at_entry),
            syph_positive_ever=bool(syph_positive_at_entry),
            entry_stage=_stage_of(self._syph, mid) if syph_positive_at_entry else None,
            syph_ti_infected=(int(self._syph.ti_infected[mid])
                              if not np.isnan(self._syph.ti_infected[mid]) else -1),
            anc_visits=[],       # [ti, ...]
            tests=[],            # [(ti, iv_name, positive_bool), ...]
            treatments=[],       # [ti, ...]
            stages_seen=[],      # [(ti, stage), ...]
            treated=False,
            mtct_ti=None,        # ti when cs_outcome first set (set_congenital fired)
            mtct_stage=None,     # mother's stage at that ti
            outcome=None,
            outcome_ti=None,
            outcome_year=None,
            cs_outcome=None,
            preterm=None,
            lbw=None,
            gestation_weeks=None,
        )

    def step(self):
        sim = self.sim
        ti = int(sim.ti)
        preg = self._preg
        syph = self._syph

        # 1. Open a log for every new conception (regardless of syph status)
        newly_preg = preg.ti_pregnant == ti
        for uid in newly_preg.uids:
            if int(uid) not in self.records:
                self._open(uid, ti, syph_positive_at_entry=bool(syph.infected[uid]))

        # 2. For every open log, update tracked events + detect syph acquisition
        newly_inf = syph.ti_infected == ti
        newly_inf_set = set(int(u) for u in newly_inf.uids)

        for mid, rec in list(self.records.items()):
            if rec['outcome'] is not None:
                continue

            # (a) Newly syph+ during pregnancy
            if mid in newly_inf_set and not rec['syph_positive_ever']:
                rec['syph_positive_ever'] = True
                rec['syph_ti_infected'] = ti
                rec['stages_seen'].append((ti, _stage_of(syph, mid)))

            # (b) ANC visits (any SyphilisANCTimer window firing this step)
            if self._anc_timer is not None:
                for k in range(len(self._anc_timer.windows)):
                    arr = getattr(self._anc_timer, f'ti_visit_{k}', None)
                    if arr is None:
                        continue
                    v = arr[mid]
                    if not np.isnan(v) and int(v) == ti:
                        rec['anc_visits'].append(ti)

            # (c) Syph tests fired this step
            for iv_name, iv in self._tests.items():
                if not hasattr(iv, 'ti_tested'):
                    continue
                t_tested = iv.ti_tested[mid]
                if not np.isnan(t_tested) and int(t_tested) == ti:
                    t_pos = iv.ti_positive[mid] if hasattr(iv, 'ti_positive') else np.nan
                    positive = bool(not np.isnan(t_pos) and int(t_pos) == ti)
                    rec['tests'].append((ti, iv_name, positive))

            # (d) Treatment this step
            if self._tx is not None and hasattr(self._tx, 'ti_treated'):
                t_tx = self._tx.ti_treated[mid]
                if not np.isnan(t_tx) and int(t_tx) == ti:
                    rec['treatments'].append(ti)
                    rec['treated'] = True

            # (e) Track stage transitions (only meaningful once syph+)
            if rec['syph_positive_ever']:
                stage = _stage_of(syph, mid)
                if not rec['stages_seen'] or rec['stages_seen'][-1][1] != stage:
                    rec['stages_seen'].append((ti, stage))

            # Detect MTCT event: cs_outcome newly set on the tracked fetus
            fetus_uid = rec['fetus_uid']
            if fetus_uid >= 0 and rec['mtct_ti'] is None:
                cs = syph.cs_outcome[fetus_uid]
                if not np.isnan(cs):
                    rec['mtct_ti'] = ti
                    rec['mtct_stage'] = _stage_of(syph, mid)

            # (f) Prenatal death detection
            fetus_uid = rec['fetus_uid']
            if fetus_uid >= 0:
                t_dead = sim.people.ti_dead[fetus_uid]
                if not np.isnan(t_dead) and int(t_dead) == ti:
                    ga_wk = float(preg.gestation[mid])
                    if np.isnan(ga_wk):
                        ga_wk = (ti - rec['conception_ti']) * float(preg.dt.weeks)
                    threshold = float(preg.pars.loss_threshold.weeks)
                    outcome = 'miscarriage' if ga_wk < threshold else 'stillbirth_prenatal'
                    self._close(rec, outcome, ti, ga_wk, live_birth=False)

    def _on_live_birth(self, mother_uids, newborn_uids):
        ti = int(self.sim.ti)
        for m, n in zip(mother_uids, newborn_uids):
            mid = int(m)
            if mid not in self.records:
                continue
            rec = self.records[mid]
            if rec['outcome'] is not None:
                continue
            fetus_uid = int(n)
            rec['fetus_uid'] = fetus_uid

            outcome = 'normal'
            if rec['syph_positive_ever']:
                for label, attr in [('nnd', 'ti_nnd'),
                                    ('stillborn', 'ti_stillborn'),
                                    ('congenital', 'ti_congenital')]:
                    arr = getattr(self._syph, attr, None)
                    if arr is None:
                        continue
                    v = arr[fetus_uid]
                    if not np.isnan(v) and int(v) <= ti:
                        outcome = label
                        break

            ga_wk = float(self._preg.gestation_at_birth[fetus_uid])
            self._close(rec, outcome, ti, ga_wk, live_birth=True, fetus_uid=fetus_uid)

    def _close(self, rec, outcome, ti, ga_wk, live_birth, fetus_uid=None):
        rec['outcome'] = outcome
        rec['outcome_ti'] = int(ti)
        rec['outcome_year'] = float(self.sim.t.yearvec[ti])
        rec['gestation_weeks'] = float(ga_wk)
        if fetus_uid is None:
            fetus_uid = rec['fetus_uid']
        if live_birth and fetus_uid >= 0:
            rec['preterm'] = bool(self._preg.preterm[fetus_uid])
            rec['lbw'] = bool(self._fh.lbw[fetus_uid])
            rec['cs_outcome'] = (float(self._syph.cs_outcome[fetus_uid])
                                 if rec['syph_positive_ever'] else float('nan'))
        else:
            rec['preterm'] = None
            rec['lbw'] = None
            rec['cs_outcome'] = (float(self._syph.cs_outcome[fetus_uid])
                                 if fetus_uid is not None and fetus_uid >= 0
                                 and rec['syph_positive_ever']
                                 else float('nan'))

    def to_records(self):
        return list(self.records.values())
