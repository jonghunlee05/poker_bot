"""
Heads-up 25bb bot: expected-value decisions against a modelled opponent.

- Preflop: all-in equities between the 169 hand classes (precomputed,
  pf_data.py) with Nash-like priors. Every option (fold, limp, shove, call)
  is scored by expected chips.
- Opponent model per team: action frequencies in each spot and the strength
  of hands they show down. Ranges are "top x% of hands", mixed with a uniform
  range when showdowns say they play randomly. Priors are Nash-like and
  observed counts pull estimates toward the opponent (safe exploitation:
  more data, more exploitation).
- Postflop: Monte Carlo against the opponent's range, reweighted by their
  aggression this hand; bets and raises chosen by EV using their observed
  fold rates.
"""

import random
from bisect import bisect_left, bisect_right

from player_action import PlayerAction
from fasteval import evaluate, card_int

DECK = list(range(52))
RANK_CH = "23456789TJQKA"


def _class_name(a, b):
    ra, rb = a >> 2, b >> 2
    if ra < rb:
        ra, rb = rb, ra
    if ra == rb:
        return RANK_CH[ra] * 2
    return RANK_CH[ra] + RANK_CH[rb] + ("s" if (a & 3) == (b & 3) else "o")


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# Priors: (probability, pseudo-count).
PRIORS = {
    "sb_fold": (0.30, 4), "sb_limp": (0.20, 4), "sb_raise": (0.35, 4),
    "sb_shove": (0.15, 4),
    "callshove": (0.30, 4),     # calls our all-in when it had not raised
    "rcallshove": (0.45, 4),    # calls our all-in after raising itself
    "bl_raise": (0.25, 4),      # raises after we limp
    "vr_fold": (0.40, 4), "vr_call": (0.40, 4), "vr_raise": (0.20, 4),
    "po_bet": (0.45, 5),        # bets postflop when checked to / first
    "fs_fold": (0.30, 5),       # folds to our small bet
    "fb_fold": (0.38, 5),       # folds to our big bet
    "fs_fold_b": (0.15, 5),     # same, after it called a bet earlier
    "fb_fold_b": (0.22, 5),
    "fc_raise": (0.12, 5),      # raises our postflop bet
    "postshove": (0.50, 4),     # calls our postflop all-in
    "fr_fold": (0.18, 5),       # folds when we raise its postflop bet
}

DEFAULT_PARAMS = {
    "r_ip": 0.95, "r_oop": 0.85, "r_flop": 0.92, "r_turn": 0.96,
    "limp_ok": 1, "samples": 300, "margin": 0.0, "opens": 1,
    "agg_m": 0.2, "raise_eps": 0.35, "call_eps": 0.4,
    "bayes": 1, "sd_n0": 6.0, "ref": 100, "look": 1, "opp_bet": 0.7,
}


class Model:
    def __init__(self, params=None):
        import pf_data as D  # data is copied into the object (pickled)
        self.names = list(D.NAMES)
        self.combos = list(D.COMBOS)
        self.M = [list(r) for r in D.M]
        self.pct = list(D.PCT)
        total = float(sum(self.combos))
        self.frac = [c / total for c in self.combos]
        self.lo = [self.pct[i] - self.frac[i] / 2 for i in range(169)]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.rng = random.Random(20261107)
        self.stats = {}
        self.hand = []
        self.hole = []
        self.cls = 0
        self.player_id = None
        self.hist = []

    # ------------------------------------------------------------ API
    def seed(self, hand, player_id):
        self.hand = hand
        self.player_id = player_id
        self.hist = []
        try:
            self.hole = [card_int(c) for c in hand]
            self.cls = self.idx[_class_name(*self.hole)]
        except Exception:
            self.hole = []

    def run(self, actions, game_state):
        me = self.player_id
        try:
            self.hist.extend(actions)
            action = self._decide(game_state)
            if action is not None:
                game_state.validate_action(action)
                return action
        except Exception:
            pass
        try:
            if game_state.call_amount == 0:
                return PlayerAction.call(me)
        except Exception:
            pass
        return PlayerAction.fold(me)

    def end_game(self, actions, game_state):
        try:
            self.hist.extend(actions)
            if game_state.num_players == 2:
                self._learn(game_state)
        except Exception:
            pass

    # ------------------------------------------------------ the model
    def _st(self, team):
        d = self.stats.get(team)
        if d is None:
            d = {"n": {}, "k": {}, "show": {}}
            self.stats[team] = d
        return d

    def _count(self, d, key, hit):
        d["n"][key] = d["n"].get(key, 0) + 1
        if hit:
            d["k"][key] = d["k"].get(key, 0) + 1

    def _count_k(self, d, key):
        d["k"][key] = d["k"].get(key, 0) + 1

    def _p(self, d, key):
        p0, n0 = self.params.get("pri", {}).get(key, PRIORS[key])
        return (d["k"].get(key, 0) + p0 * n0) / (d["n"].get(key, 0) + n0)

    def _p_group(self, d, prefix, keys):
        """Frequencies of exclusive outcomes counted under prefix."""
        n = d["n"].get(prefix, 0)
        out = {}
        for k in keys:
            p0, n0 = self.params.get("pri", {}).get(k, PRIORS[k])
            out[k] = (d["k"].get(k, 0) + p0 * n0) / (n + n0)
        s = sum(out.values())
        return {k: v / s for k, v in out.items()}

    def _show(self, d, key, value):
        lst = d["show"].setdefault(key, [])
        lst.append(round(value, 4))
        if len(lst) > 60:
            del lst[0]

    def _rho(self, d, key, x):
        """How much a 'top x' range is really top x (1) vs random (0)."""
        if x >= 0.97:
            return 1.0
        lst = d["show"].get(key, [])
        n0 = 3.0
        m = (sum(lst) + (x / 2) * n0) / (len(lst) + n0)
        return _clip((0.5 - m) / (0.5 - x / 2), 0.0, 1.0)

    # ------------------------------------------------------ hand parse
    def _parse(self, gs):
        """Replays this hand's actions. Returns a context dict."""
        n = gs.num_players
        button = gs.button
        if n == 2:
            sbp, bbp = button, 1 - button
        else:
            sbp, bbp = (button + 1) % n, (button + 2) % n
        bets = [0] * n
        bets[sbp] = gs.small_blind
        bets[bbp] = gs.big_blind
        cur = gs.big_blind
        committed = [0] * n
        street = 0
        nboard = 0
        events = []
        for a in self.hist:
            if a.kind == "reveal_card":
                if a.player_id == -1:
                    nboard += 1
                    st = {3: 1, 4: 2, 5: 3}.get(nboard)
                    if st is not None and st != street:
                        for p in range(n):
                            committed[p] += bets[p]
                        bets = [0] * n
                        cur = 0
                        street = st
                continue
            if a.kind not in ("fold", "call", "raise"):
                continue
            p = a.player_id
            ev = {"p": p, "kind": a.kind, "street": street,
                  "facing": cur - bets[p],
                  "pot": sum(committed) + sum(bets), "cur": cur,
                  "prev": committed[p], "amount": a.amount}
            if a.kind == "call":
                bets[p] = cur
            elif a.kind == "raise":
                bets[p] = a.amount
                cur = a.amount
            events.append(ev)
        total = [committed[p] + bets[p] for p in range(n)]
        start = [gs.stacks[p] + total[p] for p in range(n)]
        for ev in events:
            ev["allin"] = (ev["kind"] == "raise"
                           and ev["prev"] + ev["amount"] >= start[ev["p"]])
        return {"events": events, "start": start, "sbp": sbp, "bbp": bbp}

    # --------------------------------------------------------- learning
    def _learn(self, gs):
        me = self.player_id
        opp = 1 - me
        d = self._st(gs.teams[opp])
        ctx = self._parse(gs)
        evs = ctx["events"]
        opp_pf = [e for e in evs if e["street"] == 0 and e["p"] == opp]
        cat = None  # the opponent's preflop story
        if ctx["sbp"] == opp and opp_pf:
            e = opp_pf[0]
            self._count(d, "sb", True)
            if e["kind"] == "fold":
                self._count_k(d, "sb_fold")
            elif e["kind"] == "call":
                self._count_k(d, "sb_limp")
                cat = "limp"
            elif e["allin"]:
                self._count_k(d, "sb_shove")
                cat = "shove"
            else:
                self._count_k(d, "sb_raise")
                cat = "raise"
        opp_raised_pf = False
        for i, e in enumerate(evs):
            if e["p"] != me:
                if e["street"] == 0 and e["kind"] == "raise":
                    opp_raised_pf = True
                    if cat is None:
                        cat = "shove" if e["allin"] else "raise"
                continue
            nxt = next((x for x in evs[i + 1:] if x["p"] == opp), None)
            if nxt is None or nxt["street"] != e["street"]:
                continue
            if e["kind"] == "raise" and e["street"] > 0 and e["facing"] > 0:
                self._count(d, "fr_fold", nxt["kind"] == "fold")
            elif e["kind"] == "raise" and e["allin"]:
                hit = nxt["kind"] == "call"
                if e["street"] == 0:
                    key = "rcallshove" if opp_raised_pf else "callshove"
                    self._count(d, key, hit)
                    if hit:
                        cat = "callshove"
                else:
                    self._count(d, "postshove", hit)
            elif e["kind"] == "raise" and e["street"] == 0:
                self._count(d, "vr", True)
                k = {"fold": "vr_fold", "call": "vr_call"}.get(
                    nxt["kind"], "vr_raise")
                self._count_k(d, k)
                if nxt["kind"] == "call" and cat is None:
                    cat = "callraise"
            elif e["kind"] == "raise":
                size = e["amount"] - e["cur"]
                small = size <= 0.6 * max(e["pot"] + e["cur"] - 0, 1)
                key = "fs_fold" if small else "fb_fold"
                if self._barrel(evs[:i], opp, e["street"]):
                    key += "_b"
                self._count(d, key, nxt["kind"] == "fold")
                if nxt["kind"] != "fold":
                    self._count(d, "fc_raise", nxt["kind"] == "raise")
            elif (e["kind"] == "call" and e["street"] == 0
                  and ctx["sbp"] == me and e["facing"] > 0
                  and e["cur"] == gs.big_blind):
                self._count(d, "bl_raise", nxt["kind"] == "raise")
        agg_post = False
        for e in evs:
            if e["p"] == opp and e["street"] > 0 and e["facing"] == 0:
                self._count(d, "po_bet", e["kind"] == "raise")
            if e["p"] == opp and e["street"] > 0 and e["kind"] == "raise":
                agg_post = True
        shown = [a.card for a in self.hist
                 if a.kind == "reveal_card" and a.player_id == opp]
        if len(shown) == 2:
            oc = [card_int(c) for c in shown]
            pct = self.pct[self.idx[_class_name(*oc)]]
            if cat:
                self._show(d, cat, pct)
            board = [card_int(c) for c in gs.board]
            if agg_post and len(board) == 5:
                self._show(d, "postagg", self._board_pct(oc, board))
            seen = set()
            for e in evs:
                act = self._act(e, opp)
                if act is None or (act, e["street"]) in seen:
                    continue
                seen.add((act, e["street"]))
                k = {1: 3, 2: 4, 3: 5}[e["street"]]
                if len(board) < k:
                    continue
                pct = self._board_pct(oc, board[:k])
                hist = d.setdefault("sd", {}).setdefault(act, [0] * 5)
                hist[min(4, int(pct * 5))] += 1

    @staticmethod
    def _act(e, opp):
        """Postflop action type of an opponent event, for strength models."""
        if e["p"] != opp or e["street"] == 0:
            return None
        if e["kind"] == "raise":
            return "bet" if e["facing"] == 0 else "raise"
        if e["kind"] == "call" and e["facing"] > 0:
            return "call"
        if e["kind"] == "call":
            return "check"
        return None

    def _dens(self, d, act, a, eps):
        """Posterior density of the opponent's hand-strength percentile
        (5 buckets, 0 = strongest) given it took act: a Dirichlet whose
        prior is 'top a, plus eps elsewhere', updated with showdowns."""
        shape = []
        for b in range(5):
            inside = _clip(a - b / 5.0, 0.0, 0.2) / 0.2
            shape.append(inside + (1 - inside) * eps)
        tot = sum(shape)
        n0 = self.params["sd_n0"]
        obs = d.get("sd", {}).get(act, [0] * 5)
        post = [n0 * shape[b] / tot + obs[b] for b in range(5)]
        z = sum(post)
        return [5.0 * x / z for x in post]

    def _barrel(self, evs, opp, street):
        """Has opp called a postflop bet on an earlier street?"""
        return any(x["p"] == opp and 0 < x["street"] < street
                   and x["kind"] == "call" and x["facing"] > 0 for x in evs)

    def _board_pct(self, hole, board):
        """Share of random hands that beat hole on this board (0 = nuts)."""
        dead = set(hole) | set(board)
        live = [c for c in DECK if c not in dead]
        mine = evaluate(hole + board)
        better = 0.0
        n = 80
        for _ in range(n):
            v = evaluate(self.rng.sample(live, 2) + board)
            better += 1 if v > mine else 0.5 if v == mine else 0
        return better / n

    # ------------------------------------------------------- ranges
    def _top(self, x, rho=1.0):
        """Class weights of a 'top x' range mixed with uniform."""
        w = []
        for j in range(169):
            t = 0.0 if x <= 0 else _clip((x - self.lo[j]) / self.frac[j],
                                         0.0, 1.0)
            w.append(rho * t + (1 - rho))
        return w

    def _eq(self, i, w):
        row = self.M[i]
        num = 0.0
        den = 0.0
        for j in range(169):
            cw = self.combos[j] * w[j]
            if cw:
                num += cw * row[j]
                den += cw
        return num / den if den else 0.5

    def _wfrac(self, w):
        return sum(self.frac[j] * w[j] for j in range(169))

    def _opp_range(self, d, ctx, opp, me):
        """Opponent's preflop range (class weights) given this hand."""
        evs = [e for e in ctx["events"] if e["street"] == 0]
        w = [1.0] * 169
        opp_acts = [e for e in evs if e["p"] == opp]
        if not opp_acts:
            return w
        first = opp_acts[0]
        if ctx["sbp"] == opp:
            sb = self._p_group(d, "sb", ["sb_fold", "sb_limp", "sb_raise",
                                         "sb_shove"])
            if first["kind"] == "call":
                x = sb["sb_limp"] + sb["sb_raise"] + sb["sb_shove"]
                w = self._top(x, self._rho(d, "limp", x))
            elif first["kind"] == "raise":
                if first["allin"]:
                    x = sb["sb_shove"] + 0.3 * sb["sb_raise"]
                    w = self._top(x, self._rho(d, "shove", x))
                else:
                    x = sb["sb_raise"] + sb["sb_shove"]
                    w = self._top(x, self._rho(d, "raise", x))
        else:
            if first["kind"] == "raise":
                if first["facing"] == 0:
                    x = self._p(d, "bl_raise")
                else:
                    vr = self._p_group(d, "vr", ["vr_fold", "vr_call",
                                                 "vr_raise"])
                    x = vr["vr_raise"]
                key = "shove" if first["allin"] else "raise"
                w = self._top(x, self._rho(d, key, x))
            elif first["kind"] == "call" and first["facing"] > 0:
                vr = self._p_group(d, "vr", ["vr_fold", "vr_call",
                                             "vr_raise"])
                x = vr["vr_call"] + vr["vr_raise"]
                w = self._top(x, self._rho(d, "callraise", x))
            elif first["kind"] == "call":
                q = self._p(d, "bl_raise")
                rho = self._rho(d, "raise", q)
                t = self._top(q, rho)
                w = [1.0 - 0.8 * (t[j] - (1 - rho)) for j in range(169)]
        raises = [e for e in opp_acts[1:] if e["kind"] == "raise"]
        if raises:
            x = 0.35 * self._wfrac(w)
            key = "shove" if raises[-1]["allin"] else "raise"
            t = self._top(x, self._rho(d, key, x))
            w = [w[j] * t[j] for j in range(169)]
        return w

    def _call_range(self, w, c, rho):
        """The part of range w that calls with probability c."""
        t = self._top(c * self._wfrac(w), rho)
        return [w[j] * t[j] for j in range(169)]

    # ---------------------------------------------------- decisions
    def _decide(self, gs):
        me = self.player_id
        if gs.num_players != 2 or not self.hole:
            return self._simple(gs)
        opp = 1 - me
        d = self._st(gs.teams[opp])
        ctx = self._parse(gs)
        if gs.street == "preflop":
            return self._preflop(gs, d, ctx, me, opp)
        return self._postflop(gs, d, ctx, me, opp)

    def _allin_ev(self, gs, me, opp, eq_called, p_fold):
        """EV (chips from now) of going all-in."""
        pot = gs.pot
        my_bet, opp_bet = gs.bets[me], gs.bets[opp]
        total = my_bet + gs.stacks[me]
        opp_add = max(0, min(total - opp_bet, gs.stacks[opp]))
        matched = opp_bet + opp_add
        my_add = max(0, matched - my_bet)
        final = pot + my_add + opp_add
        if total <= gs.current_bet:
            p_fold = 0.0
        return p_fold * pot + (1 - p_fold) * (eq_called * final - my_add)

    def _raise_to(self, gs, total):
        me = self.player_id
        if not gs.can_raise:
            return PlayerAction.call(me)
        lo, hi = gs.raise_range
        return PlayerAction.raise_to(me, int(_clip(total, lo, hi)))

    def _allin(self, gs):
        me = self.player_id
        if not gs.can_raise:
            return PlayerAction.call(me)
        return PlayerAction.raise_to(me, gs.max_raise_to)

    def _preflop(self, gs, d, ctx, me, opp):
        P = self.params
        i = self.cls
        call = gs.call_amount
        pot = gs.pot
        w = self._opp_range(d, ctx, opp, me)
        opp_raised = any(e["p"] == opp and e["kind"] == "raise"
                         for e in ctx["events"])
        opp_allin = gs.stacks[opp] == 0
        options = {"fold": 0.0 if call > 0 else -1e9}
        eq = self._eq(i, w)
        if opp_allin:
            options["call"] = self._allin_ev(gs, me, opp, eq, 0.0)
            return self._pick(gs, options)
        key = "rcallshove" if opp_raised else "callshove"
        c = self._p(d, key)
        wc = self._call_range(w, c, self._rho(d, "callshove",
                                              c * self._wfrac(w)))
        if gs.can_raise:
            options["shove"] = self._allin_ev(gs, me, opp, self._eq(i, wc),
                                              1 - c)
        r = P["r_ip"] if ctx["sbp"] == me else P["r_oop"]
        if call == 0:
            options["check"] = r * eq * pot
        else:
            v = r * eq * (pot + call) - call
            if ctx["sbp"] == me and not ctx["events"]:
                if not P["limp_ok"]:
                    v = -1e9
                else:
                    q = self._p(d, "bl_raise")
                    v = (1 - q) * v + q * (-call)
            options["call"] = v
        bb = gs.big_blind
        if (P["opens"] and gs.can_raise and not opp_raised
                and gs.current_bet == bb):
            vr = self._p_group(d, "vr", ["vr_fold", "vr_call", "vr_raise"])
            f, cc, r3 = vr["vr_fold"], vr["vr_call"], vr["vr_raise"]
            wfr = self._wfrac(w)
            w_call = self._call_range(w, cc + r3, self._rho(
                d, "callraise", (cc + r3) * wfr))
            w_3b = self._call_range(w, r3, self._rho(d, "raise", r3 * wfr))
            eq_call = self._eq(i, w_call)
            eq_3b = self._eq(i, w_3b)
            my_bet = gs.bets[me]
            eff = min(gs.stacks[me] + my_bet, gs.stacks[opp] + gs.bets[opp])
            lo, hi = gs.raise_range
            for mult in ((2.5, 3) if call > 0 else (3, 4)):
                T = int(_clip(mult * bb, lo, hi))
                if T >= hi:
                    continue
                add = T - my_bet
                v_call = r * eq_call * 2 * T - add
                v_3b = max(-add, eq_3b * 2 * eff - (eff - my_bet))
                options["to%d" % T] = f * pot + cc * v_call + r3 * v_3b
        return self._pick(gs, options)

    def _pick(self, gs, options):
        me = self.player_id
        best = max(options, key=lambda k: options[k])
        if best == "fold":
            return (PlayerAction.call(me) if gs.call_amount == 0
                    else PlayerAction.fold(me))
        if best == "shove":
            return self._allin(gs)
        if best.startswith("to"):
            return self._raise_to(gs, int(best[2:]))
        return PlayerAction.call(me)

    # ----------------------------------------------------- postflop
    def _sample_hand(self, cum, dead):
        rng = self.rng
        total = cum[-1]
        for _ in range(20):
            j = min(bisect_right(cum, rng.random() * total), 168)
            name = self.names[j]
            ra = RANK_CH.index(name[0])
            rb = RANK_CH.index(name[1])
            if len(name) == 3 and name[2] == "s":
                s1 = s2 = rng.randrange(4)
            else:
                s1, s2 = rng.sample(range(4), 2)
            a, b = ra * 4 + s1, rb * 4 + s2
            if a not in dead and b not in dead:
                return [a, b]
        return rng.sample([c for c in DECK if c not in dead], 2)

    def _postflop(self, gs, d, ctx, me, opp):
        P = self.params
        board = [card_int(c) for c in gs.board]
        hole = self.hole
        dead = set(hole) | set(board)
        w = self._opp_range(d, ctx, opp, me)
        cum = []
        s = 0.0
        for j in range(169):
            s += self.combos[j] * w[j]
            cum.append(s)
        if s <= 0:
            cum = [float(k + 1) for k in range(169)]
        street = {"flop": 1, "turn": 2, "river": 3}[gs.street]
        aggs = [e for e in ctx["events"]
                if e["p"] == opp and e["street"] > 0 and e["kind"] == "raise"]
        a_bet = self._p(d, "po_bet")
        a_raise = self._p(d, "fc_raise")
        lst = d["show"].get("postagg", [])
        m = (sum(lst) + P["agg_m"] * 3) / (len(lst) + 3)
        barrel = self._barrel(ctx["events"], opp, street)
        rng = self.rng
        mine_now = evaluate(hole + board)
        need = 5 - len(board)
        samples = []
        for _ in range(int(P["samples"])):
            oh = self._sample_hand(cum, dead)
            strength = evaluate(oh + board)
            if need:
                used = dead | set(oh)
                run = []
                while len(run) < need:
                    c = rng.randrange(52)
                    if c not in used:
                        used.add(c)
                        run.append(c)
                mv = evaluate(hole + board + run)
                ov = evaluate(oh + board + run)
            else:
                mv, ov = mine_now, strength
            samples.append([strength, 1.0,
                            1.0 if mv > ov else 0.5 if mv == ov else 0.0, oh])
        samples.sort(key=lambda t: -t[0])
        n = len(samples)
        if P["bayes"]:
            self._bayes_weights(samples, d, ctx, opp, board, hole, a_bet,
                                a_raise, m, rng)
            calls = aggs = []
        else:
            calls = [e for e in ctx["events"]
                     if e["p"] == opp and e["street"] > 0
                     and e["kind"] == "call" and e["facing"] > 0]
        for e in calls:
            f = self._p(d, "fs_fold")
            keep = _clip(1 - f, 0.2, 1.0)
            cut = keep * n
            for k in range(n):
                if k >= cut:
                    samples[k][1] *= P["call_eps"]
        for e in aggs:
            a = a_raise if e["facing"] > 0 else a_bet
            eps = _clip((m - a / 2) / (0.5 - a / 2), 0.05, 1.0)
            if e["facing"] > 0:
                eps *= P["raise_eps"]
            cut = a * n
            for k in range(n):
                if k >= cut:
                    samples[k][1] *= eps

        tw = sum(t[1] for t in samples)

        def equity(frac_top):
            lim = frac_top * tw
            acc = 0.0
            num = 0.0
            for t in samples:
                if acc >= lim:
                    break
                take = min(t[1], lim - acc)
                acc += take
                num += take * t[2]
            return num / acc if acc > 0 else 0.5

        eq = equity(1.0)
        pot = gs.pot
        call = gs.call_amount
        real = {1: P["r_flop"], 2: P["r_turn"], 3: 1.0}[street]
        options = {"fold": 0.0 if call > 0 else -1e9}
        opp_to_act = (gs.stacks[opp] > 0 and not any(
            e["p"] == opp and e["street"] == street for e in ctx["events"]))
        if call == 0 and P["look"] and opp_to_act:
            # one-step lookahead: after our check the opponent bets its
            # strongest a_bet (by weight) or checks behind; we then call
            # or fold, whichever is better.
            ab = _clip(a_bet, 0.05, 0.95)
            lst2 = d["show"].get("postagg", [])
            m2 = (sum(lst2) + P["agg_m"] * 3) / (len(lst2) + 3)
            eps = _clip((m2 - ab / 2) / (0.5 - ab / 2), 0.05, 1.0)
            # bet likelihood per sample: 1 in the top ab, eps below
            acc = 0.0
            wb = []
            for t in samples:
                wb.append(t[1] * (1.0 if acc < ab * tw else eps))
                acc += t[1]
            sb_ = sum(wb)
            sc_ = sum(t[1] - x for t, x in zip(samples, wb))
            pb = sb_ / tw if tw else 0.0
            eq_b = (sum(x * t[2] for t, x in zip(samples, wb)) / sb_
                    if sb_ else eq)
            eq_c = (sum((t[1] - x) * t[2] for t, x in zip(samples, wb)) / sc_
                    if sc_ else eq)
            B = min(P["opp_bet"] * pot, gs.stacks[opp], gs.stacks[me])
            v_bet = max(0.0, eq_b * (pot + 2 * B) * real - B)
            options["check"] = pb * v_bet + (1 - pb) * eq_c * pot * real
        elif call == 0:
            options["check"] = eq * pot * real
        else:
            options["call"] = (eq * (pot + call) * real - call
                               - P["margin"] * pot)
        if gs.can_raise:
            if call > 0:
                c = 1 - self._p(d, "fr_fold")
            else:
                c = self._p(d, "postshove")
            options["shove"] = self._allin_ev(gs, me, opp, equity(c), 1 - c)
            lo, hi = gs.raise_range
            for frac in (0.5, 0.75, 1.0):
                size = frac * (pot + call)
                total = int(_clip(gs.current_bet + size, lo, hi))
                if total >= hi:
                    continue
                add = total - gs.bets[me]
                opp_add = total - gs.bets[opp]
                small = (total - gs.current_bet) <= 0.6 * (pot + call)
                if call > 0:
                    f = self._p(d, "fr_fold")
                else:
                    f = self._p(d, ("fs_fold" if small else "fb_fold")
                                + ("_b" if barrel else ""))
                ec = equity(1 - f)
                options["to%d" % total] = (
                    f * pot + (1 - f) * (ec * (pot + add + opp_add) * real
                                         - add))
        return self._pick(gs, options)

    def _bayes_weights(self, samples, d, ctx, opp, board, hole, a_bet,
                       a_raise, m, rng):
        """Reweights sampled opponent hands by the likelihood of each
        action they took this hand, given the hand's strength on the board
        of that street (Bayes' rule with per-opponent Dirichlet models)."""
        P = self.params
        acts = []
        seen = set()
        for e in ctx["events"]:
            act = self._act(e, opp)
            if act in (None, "check") or (act, e["street"]) in seen:
                continue
            seen.add((act, e["street"]))
            if act == "bet":
                a = a_bet
                eps = _clip((m - a / 2) / (0.5 - a / 2), 0.05, 1.0)
            elif act == "raise":
                a = a_raise
                eps = _clip((m - a / 2) / (0.5 - a / 2), 0.05, 1.0)
                eps *= P["raise_eps"]
            else:
                a = _clip(1 - self._p(d, "fs_fold"), 0.2, 1.0)
                eps = P["call_eps"]
            acts.append((e["street"], self._dens(d, act, a, eps)))
        if not acts:
            return
        dead = set(hole) | set(board)
        live = [c for c in DECK if c not in dead]
        refs = {}
        for st in {x[0] for x in acts}:
            b = board[:{1: 3, 2: 4, 3: 5}[st]]
            refs[st] = (b, sorted(evaluate(rng.sample(live, 2) + b)
                                  for _ in range(int(P["ref"]))))
        for t in samples:
            oh = t[3]
            for st, dens in acts:
                b, ref = refs[st]
                v = evaluate(oh + b)
                lo = bisect_left(ref, v)
                hi = bisect_right(ref, v)
                pct = (len(ref) - hi + 0.5 * (hi - lo)) / len(ref)
                t[1] *= dens[min(4, int(pct * 5))]

    # ------------------------------------------------------ multiway
    def _simple(self, gs):
        me = self.player_id
        board = [card_int(c) for c in gs.board]
        hole = self.hole or [card_int(c) for c in self.hand]
        opps = max(1, len(gs.players_in_hand) - 1)
        dead = set(hole) | set(board)
        live = [c for c in DECK if c not in dead]
        need = 5 - len(board)
        won = 0.0
        n = 150
        for _ in range(n):
            cs = self.rng.sample(live, need + 2 * opps)
            fb = board + cs[:need]
            mv = evaluate(hole + fb)
            best = max(evaluate(cs[need + 2 * k:need + 2 * k + 2] + fb)
                       for k in range(opps))
            won += 1 if mv > best else 0.5 if mv == best else 0
        eq = won / n
        if eq > 1.6 / (opps + 1) and gs.can_raise:
            return self._raise_to(gs, gs.pot)
        if eq > gs.pot_odds:
            return PlayerAction.call(me)
        return (PlayerAction.call(me) if gs.call_amount == 0
                else PlayerAction.fold(me))
