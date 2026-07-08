"""Local copy of the edge-stratified PartnerNotification class.

Mirrors the contents of the upstream feat/baseline-pn branch (stisim PR 505).
That work isn't landing in stisim 1.5.7; rather than wait, we maintain a
local copy here. Drop this file and re-parent SyndromicPN / POCPN on
``sti.PartnerNotification`` once 505 lands in a future stisim release.

Differences from the simpler ``sti.PartnerNotification`` (rc1.5.7):
  - Walks per-edge with edge_type info preserved (``self.current_partner_edges``).
  - ``pn_rates(rates)`` helper builds an edge-type / sex-stratified callable
    suitable as the ``p`` arg of ``ss.bernoulli`` on ``p_notify_current``
    or ``p_attends_current``.
  - Backwards-compatible: scalar Bernoullis still work.

The previous-partner channel is left scalar-only here (``pn_rates`` returns
zeros when ``self.current_partner_edges`` is empty, e.g. on the prior
channel). Networks without an ``edge_type`` column (like ``PriorPartners``)
get ``edge_type=-1`` and pn_rates emits probability 0 for those edges.
"""

from collections import defaultdict

import numpy as np
import starsim as ss


__all__ = ['PartnerNotification', 'pn_rates']


class PartnerNotification(ss.Intervention):
    """Notify and follow up sexual partners of index cases.

    Two channels: current-partner (via the active sexual network) and
    prior-partner (via a :class:`stisim.PriorPartners` recall network).
    Each contact is offered notification with probability ``p_notify_<scope>``;
    notified contacts attend follow-up with probability ``p_attends_<scope>``.
    Attendees are scheduled for the supplied ``test`` intervention on the
    next timestep. Index cases are not notified about themselves; partners
    reachable via both channels are not double-notified.

    Args:
        eligibility: Callable ``f(sim) -> uids`` returning index cases.
        test: Test intervention to schedule for attending partners. Optional
            — subclasses overriding :meth:`notify_attendees` may set this to
            ``None``.
        pars: Optional dict of parameter overrides.
        **kwargs: Forwarded to ``ss.Intervention``.
    """

    def __init__(self, eligibility, test=None, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            p_notify_current=ss.bernoulli(p=0.5),
            p_attends_current=ss.bernoulli(p=0.5),
            p_notify_previous=ss.bernoulli(p=0.05),
            p_attends_previous=ss.bernoulli(p=0.01),
            current_network='structuredsexual',
            previous_network='priorpartners',
        )
        self.update_pars(pars=pars, **kwargs)
        self.eligibility = eligibility
        self.test = test

        self.define_states(
            ss.FloatArr('ti_notified'),
        )
        self._cur_nw = None
        self._prev_nw = None
        self._use_previous = False
        self.current_partner_edges = defaultdict(list)
        # Opt-in dyad-level event log for chain tracing. Default None (off).
        # Set to a list ([]) on the live module to enable: step() then
        # appends (ti, index_uid, partner_uid, notified, attended) tuples
        # for every current-channel (index, partner) dyad. Recomputed from
        # network topology only — consumes no RNG, changes no behaviour.
        self.trace_events = None
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self._cur_nw = sim.networks.get(self.pars.current_network)
        if self._cur_nw is None:
            raise ValueError(
                f"PartnerNotification requires network '{self.pars.current_network}' in the sim."
            )
        self._use_previous = self.pars.p_notify_previous.pars.p > 0
        if self._use_previous:
            self._prev_nw = sim.networks.get(self.pars.previous_network)
            if self._prev_nw is None:
                raise ValueError(
                    f"PartnerNotification with p_notify_previous>0 requires network "
                    f"'{self.pars.previous_network}' in the sim. Set p_notify_previous=0 to disable."
                )
        return

    def init_results(self):
        super().init_results()
        # Sex-stratified versions of the index/notify/attend funnel are added
        # for every count below so VDS (female) vs UDS (male) channels can be
        # separated downstream. Under SOC syndromic management the BV-dominant
        # VDS etiology drives the female false-alarm rate well above the male
        # UDS one; aggregating across sexes hides this.
        self.define_results(
            ss.Result('new_notified', dtype=int, label='Partners notified'),
            ss.Result('new_notified_f', dtype=int, label='Partners notified (F)', auto_plot=False),
            ss.Result('new_notified_m', dtype=int, label='Partners notified (M)', auto_plot=False),
            ss.Result('new_attending', dtype=int, label='Partners attending'),
            ss.Result('new_attending_f', dtype=int, label='Partners attending (F)', auto_plot=False),
            ss.Result('new_attending_m', dtype=int, label='Partners attending (M)', auto_plot=False),
            ss.Result('new_notified_current', dtype=int, label='Current partners notified', auto_plot=False),
            ss.Result('new_notified_previous', dtype=int, label='Prior partners notified', auto_plot=False),
            # ---- PN justification: did the index actually need to alert anyone? ----
            # Total PN triggers this step (size of eligibility pool).
            ss.Result('new_index_total', dtype=int,
                      label='PN triggers (index cases)', auto_plot=False),
            ss.Result('new_index_total_f', dtype=int,
                      label='PN triggers, female index', auto_plot=False),
            ss.Result('new_index_total_m', dtype=int,
                      label='PN triggers, male index', auto_plot=False),
            # Index false-alarm: PN triggered but the index has no current STI.
            # The headline "was PN warranted?" metric. Under SOC syndromic,
            # BV-only women presenting at VDS become indices despite no STI →
            # high false-alarm rate. Under POC test-based dx, only test-positive
            # agents trigger PN → false-alarm rate ~0 (modulo test specificity).
            ss.Result('new_index_no_sti', dtype=int,
                      label='PN triggers with no current STI (false alarm)',
                      auto_plot=False),
            ss.Result('new_index_no_sti_f', dtype=int,
                      label='PN triggers, female index, no STI',
                      auto_plot=False),
            ss.Result('new_index_no_sti_m', dtype=int,
                      label='PN triggers, male index, no STI',
                      auto_plot=False),
            # ---- Partner outcome (downstream signal, NOT a PN-warranted metric) ----
            # Of partners notified/attending, what fraction currently have no STI.
            # Reflects transmission-link enrichment + Bernoulli filters. A negative
            # partner test from a CORRECTLY-triggered PN is still a worthwhile test.
            ss.Result('new_notified_no_sti', dtype=int,
                      label='PN notified, no current STI',
                      auto_plot=False),
            ss.Result('new_notified_no_sti_f', dtype=int,
                      label='PN notified, female, no STI', auto_plot=False),
            ss.Result('new_notified_no_sti_m', dtype=int,
                      label='PN notified, male, no STI', auto_plot=False),
            ss.Result('new_attended_no_sti', dtype=int,
                      label='PN attendees with no current STI',
                      auto_plot=False),
            ss.Result('new_attended_no_sti_f', dtype=int,
                      label='PN attendees, female, no STI', auto_plot=False),
            ss.Result('new_attended_no_sti_m', dtype=int,
                      label='PN attendees, male, no STI', auto_plot=False),
        )
        return

    def notify_attendees(self, uids):
        """Schedule ``self.test`` for attendees on the next timestep.

        Subclasses can override to apply per-partner-sex syndromic treatment,
        set treatment eligibility on a specific intervention, etc.
        """
        if self.test is None:
            raise ValueError(
                'PartnerNotification.notify_attendees: no `test` set and '
                'no override provided. Either pass test=intervention or '
                'subclass and override notify_attendees().'
            )
        self.test.schedule(uids, self.ti + 1)
        return

    def find_partners(self, nw, index_uids):
        """Walk edges of ``nw`` and return ``(partner_uids, edge_types)``.

        One entry per edge where exactly one endpoint is in ``index_uids``.
        Edges connecting two index cases drop out — index cases aren't
        notified about themselves. ``edge_types`` is filled with ``-1`` if
        ``nw.edges`` has no ``edge_type`` column (e.g. PriorPartners);
        ``pn_rates`` callables map unrecognized edge_type ints to
        probability 0, gracefully disabling edge-type stratification.
        """
        p1_is_index = np.isin(nw.p1, index_uids)
        p2_is_index = np.isin(nw.p2, index_uids)
        # XOR: keep edges where exactly one endpoint is an index.
        has_one_index = p1_is_index ^ p2_is_index
        partner_uids = np.where(p1_is_index, nw.p2, nw.p1)[has_one_index]
        if 'edge_type' in nw.edges:
            edge_types = nw.edges.edge_type[has_one_index]
        else:
            edge_types = np.full(len(partner_uids), -1, dtype=int)
        return partner_uids, edge_types

    def step(self):
        index_uids = self.eligibility(self.sim)
        if len(index_uids) == 0:
            return

        # Current-partner channel: walk edges, group edge_types by partner uid.
        # Callables on p_notify_current.p / p_attends_current.p can read
        # `self.current_partner_edges` to stratify by edge type.
        cur_uids, cur_edge_types = self.find_partners(self._cur_nw, index_uids)
        self.current_partner_edges = defaultdict(list)
        for uid, et in zip(cur_uids, cur_edge_types):
            self.current_partner_edges[int(uid)].append(int(et))
        cur_partners = ss.uids(np.unique(cur_uids))

        cur_notified = self.pars.p_notify_current.filter(cur_partners)
        cur_attending = self.pars.p_attends_current.filter(cur_notified)

        # Prior-partner channel (skip partners already notified as current).
        # Edge-type stratification is not currently supported here — pn_rates
        # helpers see an empty current_partner_edges and return zeros.
        if self._use_previous:
            prev_uids, _ = self.find_partners(self._prev_nw, index_uids)
            prev_partners = ss.uids(np.unique(prev_uids)) - cur_partners
            prev_notified = self.pars.p_notify_previous.filter(prev_partners)
            prev_attending = self.pars.p_attends_previous.filter(prev_notified)
        else:
            prev_notified = ss.uids()
            prev_attending = ss.uids()

        all_attending = cur_attending | prev_attending
        all_notified = cur_notified | prev_notified
        ti = self.ti

        # PN funnel precision: of notified / attending partners, count those
        # with no current NG/CT/TV/syph infection. BV doesn't justify PN, so
        # BV-only counts as "no STI". Measured BEFORE notify_attendees() — both
        # SyndromicPN and POCPN override notify_attendees to fire
        # vds.step / panel.step, which append uids to tx.eligibility queues.
        # In stisim's current SyndromicManagement flow, the actual tx.step
        # fires on the NEXT timestep (deferred), so this ordering doesn't
        # affect the count today — but measure before to be robust to any
        # future change that fires treatment inline.
        # Index-side: was PN warranted? Restricted to NG + CT + syph, the
        # three discrete tests where "unnecessary" treatment unambiguously
        # means "the test was wrong". Metronidazole is excluded because it
        # treats both TV and BV — a "no-TV" outcome doesn't imply unnecessary
        # tx (BV may be the indication).
        #
        # An index has "no STI" iff their own treatment was in
        # tx.outcomes[d].unnecessary for at least one of ng_tx / ct_tx / syph_tx,
        # AND was NOT in outcomes[d].(successful | unsuccessful) for any of them.
        #
        # We CAN'T just check `dis.infected[index_uids]` here, because
        # STITreatment.change_states -> clear_infection has already run this
        # step. A successfully-treated confirmed-positive index would look
        # no-STI by the time pn.step is reached. tx.outcomes is the right
        # source: it freezes "had STI at treatment time" before clearance.
        clean_index_set = ss.uids()
        had_sti_set = ss.uids()
        for tx_name, d_key in (('ng_tx', 'ng'), ('ct_tx', 'ct'), ('syph_tx', 'syph')):
            tx = self.sim.interventions.get(tx_name)
            if tx is None:
                continue
            # All agents whose tx fired this step.
            clean_index_set = clean_index_set | (tx.ti_treated == ti).uids
            out = getattr(tx, 'outcomes', None)
            if out is None or d_key not in out or not hasattr(out[d_key], 'get'):
                continue
            had_sti_set = (had_sti_set
                | out[d_key].get('successful', ss.uids())
                | out[d_key].get('unsuccessful', ss.uids()))
        false_alarm = clean_index_set.remove(had_sti_set)
        female = self.sim.people.female
        self.results['new_index_total'][ti] += len(clean_index_set)
        self.results['new_index_no_sti'][ti] += len(false_alarm)
        if len(clean_index_set):
            f_index = female[clean_index_set]
            self.results['new_index_total_f'][ti] += int(f_index.sum())
            self.results['new_index_total_m'][ti] += int((~f_index).sum())
        if len(false_alarm):
            f_false = female[false_alarm]
            self.results['new_index_no_sti_f'][ti] += int(f_false.sum())
            self.results['new_index_no_sti_m'][ti] += int((~f_false).sum())

        # Notify / attend totals broken out by partner sex. The aggregate
        # `new_notified` / `new_attending` are set unconditionally below (line
        # ~323), so we only record the _f / _m splits here.
        if len(all_notified):
            f_not = female[all_notified]
            self.results['new_notified_f'][ti] += int(f_not.sum())
            self.results['new_notified_m'][ti] += int((~f_not).sum())
        if len(all_attending):
            f_att = female[all_attending]
            self.results['new_attending_f'][ti] += int(f_att.sum())
            self.results['new_attending_m'][ti] += int((~f_att).sum())

        # Partner-side STI status: of partners notified / attending, what
        # fraction have no current STI. A negative partner test from a
        # correctly-triggered PN is still a worthwhile test, so this is a
        # transmission-link signal, NOT a "was PN warranted?" signal. Partners
        # are NOT in tx.outcomes this step (they get queued for next-step tx),
        # so dis.infected is safe.
        any_sti = None
        for d in ('ng', 'ct', 'tv', 'syph'):
            dis = self.sim.diseases.get(d)
            if dis is None or not hasattr(dis, 'infected'):
                continue
            any_sti = dis.infected if any_sti is None else (any_sti | dis.infected)
        if any_sti is not None:
            if len(all_notified):
                no_sti_not = ~any_sti[all_notified]
                self.results['new_notified_no_sti'][ti] += int(no_sti_not.sum())
                f_not = female[all_notified]
                self.results['new_notified_no_sti_f'][ti] += int((no_sti_not & f_not).sum())
                self.results['new_notified_no_sti_m'][ti] += int((no_sti_not & ~f_not).sum())
            if len(all_attending):
                no_sti_att = ~any_sti[all_attending]
                self.results['new_attended_no_sti'][ti] += int(no_sti_att.sum())
                f_att = female[all_attending]
                self.results['new_attended_no_sti_f'][ti] += int((no_sti_att & f_att).sum())
                self.results['new_attended_no_sti_m'][ti] += int((no_sti_att & ~f_att).sum())

        if len(all_attending):
            self.ti_notified[all_attending] = self.ti
            self.notify_attendees(all_attending)

        if self.trace_events is not None:
            self._log_trace(index_uids, cur_notified, cur_attending)

        self.results['new_notified_current'][ti] = len(cur_notified)
        self.results['new_notified_previous'][ti] = len(prev_notified)
        self.results['new_notified'][ti] = len(cur_notified) + len(prev_notified)
        self.results['new_attending'][ti] = len(all_attending)
        return

    def _log_trace(self, index_uids, cur_notified, cur_attending):
        """Append current-channel (index, partner) dyads to ``trace_events``.

        For chain tracing. Re-derives the index→partner edge mapping from
        network topology (no RNG draws), then tags each dyad with the
        actual notified / attended outcomes computed in ``step``. One
        record per (index, partner) edge:
        ``(ti, index_uid, partner_uid, notified, attended)``.
        """
        nw = self._cur_nw
        p1_is_index = np.isin(nw.p1, index_uids)
        p2_is_index = np.isin(nw.p2, index_uids)
        has_one_index = p1_is_index ^ p2_is_index
        partner = np.where(p1_is_index, nw.p2, nw.p1)[has_one_index]
        index = np.where(p1_is_index, nw.p1, nw.p2)[has_one_index]
        notified = set(int(u) for u in cur_notified)
        attending = set(int(u) for u in cur_attending)
        ti = self.ti
        for ip, pp in zip(index, partner):
            pp = int(pp)
            self.trace_events.append(
                (ti, int(ip), pp, int(pp in notified), int(pp in attending)))
        return


def pn_rates(rates):
    """Build a per-UID probability callable for :class:`PartnerNotification`.

    Stratifies notification or attendance rates by partnership edge type
    (and optionally partner sex). Returned callable is suitable as the
    ``p`` arg of ``ss.bernoulli`` on ``p_notify_current`` / ``p_attends_current``.

    Args:
        rates: dict mapping edge-type name to a probability OR to a per-sex
            dict. Two supported shapes:

            - ``{edge_name: prob}`` — sex-independent.
              Example: ``{'stable': 0.20, 'casual': 0.10}``.
            - ``{edge_name: {'f': prob, 'm': prob}}`` — sex-dependent.
              Example: ``{'stable': {'f': 0.80, 'm': 0.50}, ...}``.

            Edge names must match the active network's ``edge_types`` keys.
            Edges whose type is not in ``rates`` get probability 0.

    Returns:
        callable f(module, sim, uids) -> ndarray of probabilities, one per
        UID. If a partner is reachable via multiple current-channel edges,
        per-edge probabilities are **summed and capped at 1**.
    """
    sample = next(iter(rates.values()), None)
    sex_dependent = isinstance(sample, dict)
    if sex_dependent:
        for k, v in rates.items():
            if not isinstance(v, dict):
                raise ValueError(
                    f"pn_rates: inconsistent rates spec; key '{k}' has "
                    f"non-dict value while another key is a dict."
                )

    def func(module, sim, uids):
        partner_edges = getattr(module, 'current_partner_edges', None) or {}
        if not partner_edges:
            return np.zeros(len(uids))
        nw = module._cur_nw
        int_to_name = {int(v): k for k, v in nw.edge_types.items()}
        if sex_dependent:
            female = sim.people.female
        out = np.zeros(len(uids))
        for i, u in enumerate(uids):
            uid = int(u)
            edge_types = partner_edges.get(uid, ())
            if not edge_types:
                continue
            total = 0.0
            if sex_dependent:
                sex_key = 'f' if female[uid] else 'm'
            for et in edge_types:
                name = int_to_name.get(int(et))
                if name is None or name not in rates:
                    continue
                spec = rates[name]
                if sex_dependent:
                    total += float(spec.get(sex_key, 0.0))
                else:
                    total += float(spec)
            out[i] = min(total, 1.0)
        return out

    return func
