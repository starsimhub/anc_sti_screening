"""
PregnancyLog — a per-pregnancy log for every pregnancy in the sim.

Each pregnancy is a separate record (so a mother with n pregnancies produces n
records). Linkage relies on starsim / stisim primitives:

  - `Pregnancy.add_delivery_callback` gives us (mother_uids, newborn_uids)
    authoritatively at delivery.
  - stisim `Syphilis.set_congenital` sets per-fetus BoolArrs
    `mtct_from_{mat_active, early, late}` at MTCT time. At delivery we read
    those directly to get the mother's stage bucket.
  - `syph.cs_outcome[fetus_uid]` is the authoritative outcome
    (ti_{outcome} arrays get cleared inside syph.step before the delivery
    callback fires).

Fields per pregnancy record:
  - mother_uid, fetus_uid, conception_ti, conception_year, scheduled_delivery_ti
  - syph_positive_at_entry, syph_positive_ever, entry_stage, syph_ti_infected
  - anc_visits [ti, ...]
  - tests [(ti, iv_name, positive), ...]
  - treatments [ti, ...]
  - treated (bool)
  - mtct_stage ∈ {mat_active, early, late, None}
  - cs_outcome (0=misc, 1=nnd, 2=stillborn, 3=congenital, 4=normal, or NaN)
  - outcome ∈ {normal, nnd, stillborn, congenital, miscarriage, stillbirth_prenatal}
  - outcome_ti, outcome_year, gestation_weeks, preterm, lbw
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
CS_OUTCOME_LABELS = {0: 'miscarriage', 1: 'nnd', 2: 'stillborn',
                     3: 'congenital', 4: 'normal'}


def _stage_of(syph, uid):
    for s in STAGE_ATTRS:
        arr = getattr(syph, s, None)
        if arr is not None and bool(arr[uid]):
            return s
    return 'infected_unknown' if bool(syph.infected[uid]) else None


def _mtct_stage_of(syph, fetus_uid):
    for tag in ('mat_active', 'early', 'late'):
        arr = getattr(syph, f'mtct_from_{tag}', None)
        if arr is not None and bool(arr[fetus_uid]):
            return tag
    return None


class PregnancyLog(ss.Analyzer):
    """Log every pregnancy from conception to close (delivery or loss)."""

    def __init__(self, name='pregnancy_log', **kwargs):
        super().__init__(name=name, **kwargs)
        self.records = []
        self._open_by_mother = {}

    def init_pre(self, sim):
        super().init_pre(sim)
        self._preg = sim.demographics.pregnancy
        self._syph = sim.diseases.syph
        self._fh = sim.custom['fetal_health']
        self._tests = {n: sim.interventions[n]
                       for n in SYPH_TEST_NAMES if n in sim.interventions}
        self._tx = sim.interventions.get(SYPH_TX_NAME)
        self._anc_timer = sim.interventions.get(SYPH_ANC_TIMER_NAME)
        self._preg.add_delivery_callback(self._on_live_birth)

    def _open(self, mother_uid, ti, syph_positive_at_entry):
        preg = self._preg
        mid = mother_uid
        conception_ti = int(preg.ti_pregnant[mid]) if not np.isnan(preg.ti_pregnant[mid]) else int(ti)
        sched_del = preg.ti_delivery[mid]
        rec = dict(
            mother_uid=mid,
            fetus_uid=-1,
            conception_ti=conception_ti,
            conception_year=float(self.sim.t.yearvec[conception_ti])
                if 0 <= conception_ti < self.sim.t.npts else float('nan'),
            scheduled_delivery_ti=int(sched_del) if not np.isnan(sched_del) else -1,
            syph_positive_at_entry=bool(syph_positive_at_entry),
            syph_positive_ever=bool(syph_positive_at_entry),
            entry_stage=_stage_of(self._syph, mid) if syph_positive_at_entry else None,
            syph_ti_infected=(int(self._syph.ti_infected[mid])
                              if not np.isnan(self._syph.ti_infected[mid]) else -1),
            anc_visits=[],
            tests=[],
            treatments=[],
            treated=False,
            mtct_stage=None,
            cs_outcome=None,
            outcome=None,
            outcome_ti=None,
            outcome_year=None,
            preterm=None,
            lbw=None,
            gestation_weeks=None,
        )
        self.records.append(rec)
        self._open_by_mother[mid] = rec

    def step(self):
        sim = self.sim
        ti = int(sim.ti)
        preg = self._preg
        syph = self._syph

        newly_preg = preg.ti_pregnant == ti
        for uid in newly_preg.uids:
            self._open(uid, ti, syph_positive_at_entry=bool(syph.infected[uid]))

        newly_inf_set = set(int(u) for u in (syph.ti_infected == ti).uids)

        for mid, rec in list(self._open_by_mother.items()):
            if rec['outcome'] is not None:
                continue

            if mid in newly_inf_set and not rec['syph_positive_ever']:
                rec['syph_positive_ever'] = True
                rec['syph_ti_infected'] = ti

            if self._anc_timer is not None:
                for k in range(len(self._anc_timer.windows)):
                    arr = getattr(self._anc_timer, f'ti_visit_{k}', None)
                    if arr is None:
                        continue
                    v = arr[mid]
                    if not np.isnan(v) and int(v) == ti:
                        rec['anc_visits'].append(ti)

            for iv_name, iv in self._tests.items():
                if not hasattr(iv, 'ti_tested'):
                    continue
                t_tested = iv.ti_tested[mid]
                if not np.isnan(t_tested) and int(t_tested) == ti:
                    t_pos = iv.ti_positive[mid] if hasattr(iv, 'ti_positive') else np.nan
                    positive = bool(not np.isnan(t_pos) and int(t_pos) == ti)
                    rec['tests'].append((ti, iv_name, positive))

            if self._tx is not None and hasattr(self._tx, 'ti_treated'):
                t_tx = self._tx.ti_treated[mid]
                if not np.isnan(t_tx) and int(t_tx) == ti:
                    rec['treatments'].append(ti)
                    rec['treated'] = True

            # Prenatal loss: mother stopped being pregnant, no delivery callback
            if not bool(preg.pregnant[mid]):
                ga_wk = float(preg.gestation[mid])
                if np.isnan(ga_wk):
                    ga_wk = (ti - rec['conception_ti']) * float(preg.dt.weeks)
                threshold = float(preg.pars.loss_threshold.weeks)
                outcome = 'miscarriage' if ga_wk < threshold else 'stillbirth_prenatal'
                self._close(rec, outcome, ti, ga_wk, fetus_uid=None)

    def _on_live_birth(self, mother_uids, newborn_uids):
        ti = int(self.sim.ti)
        syph = self._syph
        for m, n in zip(mother_uids, newborn_uids):
            mid = int(m)
            if mid not in self._open_by_mother:
                continue
            rec = self._open_by_mother[mid]
            if rec['outcome'] is not None:
                continue
            fetus_uid = int(n)
            rec['fetus_uid'] = fetus_uid

            rec['mtct_stage'] = _mtct_stage_of(syph, fetus_uid)

            cs = syph.cs_outcome[fetus_uid]
            if not np.isnan(cs):
                rec['cs_outcome'] = float(cs)
                outcome = CS_OUTCOME_LABELS.get(int(cs), 'normal')
            else:
                outcome = 'normal'

            ga_wk = float(self._preg.gestation_at_birth[fetus_uid])
            self._close(rec, outcome, ti, ga_wk, fetus_uid=fetus_uid, live_birth=True)

    def _close(self, rec, outcome, ti, ga_wk, fetus_uid=None, live_birth=False):
        rec['outcome'] = outcome
        rec['outcome_ti'] = int(ti)
        rec['outcome_year'] = float(self.sim.t.yearvec[ti])
        rec['gestation_weeks'] = float(ga_wk)
        if live_birth and fetus_uid is not None and fetus_uid >= 0:
            rec['preterm'] = bool(self._preg.preterm[fetus_uid])
            rec['lbw'] = bool(self._fh.lbw[fetus_uid])
        self._open_by_mother.pop(rec['mother_uid'], None)

    def to_records(self):
        return self.records
