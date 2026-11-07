"""Fast 5-7 card hand evaluator on integer cards (rank * 4 + suit).

rank 0..12 is 2..A. Returns an int: higher is better, equal ties.
"""

RANKS = "23456789TJQKA"
SUITS = "CDHS"


def _build_straights():
    table = [-1] * 8192
    for mask in range(8192):
        m = (mask << 1) | (1 if mask & 4096 else 0)  # bit 0 is a low ace
        for top in range(13, 3, -1):
            if (m >> (top - 4)) & 31 == 31:
                table[mask] = top - 1  # rank of the high card (5-high = 3)
                break
    return table


STRAIGHT = _build_straights()


def card_int(card):
    return RANKS.index(card.rank) * 4 + SUITS.index(card.suit)


def evaluate(cards):
    rc = [0] * 13
    sm = [0, 0, 0, 0]
    for c in cards:
        r = c >> 2
        rc[r] += 1
        sm[c & 3] |= 1 << r
    for m in sm:
        if m.bit_count() >= 5:
            s = STRAIGHT[m]
            if s >= 0:
                return (8 << 20) | s
            out = 0
            n = 0
            for r in range(12, -1, -1):
                if m >> r & 1:
                    out = (out << 4) | r
                    n += 1
                    if n == 5:
                        break
            return (5 << 20) | out
    quad = -1
    trips = []
    pairs = []
    singles = []
    mask = 0
    for r in range(12, -1, -1):
        n = rc[r]
        if n:
            mask |= 1 << r
            if n == 4:
                quad = r
            elif n == 3:
                trips.append(r)
            elif n == 2:
                pairs.append(r)
            else:
                singles.append(r)
    if quad >= 0:
        k = max([r for r in range(13) if rc[r] and r != quad])
        return (7 << 20) | (quad << 4) | k
    if trips and (len(trips) > 1 or pairs):
        p = trips[1] if len(trips) > 1 else -1
        if pairs and pairs[0] > p:
            p = pairs[0]
        return (6 << 20) | (trips[0] << 4) | p
    s = STRAIGHT[mask]
    if s >= 0:
        return (4 << 20) | s
    if trips:
        ks = sorted(pairs + singles, reverse=True)
        return (3 << 20) | (trips[0] << 8) | (ks[0] << 4) | ks[1]
    if len(pairs) >= 2:
        k = max(pairs[2:] + singles) if (pairs[2:] or singles) else 0
        return (2 << 20) | (pairs[0] << 8) | (pairs[1] << 4) | k
    if pairs:
        out = pairs[0]
        for r in singles[:3]:
            out = (out << 4) | r
        return (1 << 20) | (out << (4 * (3 - len(singles[:3]))))
    out = 0
    for r in singles[:5]:
        out = (out << 4) | r
    return out << (4 * (5 - len(singles[:5])))
