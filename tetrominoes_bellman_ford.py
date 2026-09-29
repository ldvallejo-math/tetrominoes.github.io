"""
Reimplementation of the MATLAB fringe-discovery algorithm using bitmasks
instead of w x 3 matrices. Same four patterns, same bottom-up target cell,
same double-shift "churn" logic -- just simpler to read.

Convention: bit 0 = bottom row, bit (w-1) = top row, matching the
algorithm's bottom-up fill order (so "lowest open bit" == "next unfilled
row scanning up from the bottom").
"""

def lowest_open(mask, w):
    """Row index (bit) of the next unfilled cell in the current column,
    scanning from the bottom. Everything below this bit is already filled."""
    r = 0
    while mask & (1 << r):
        r += 1
    return r

def legal_moves(c1, c2, c3, r, w):
    """Every way to cover the target cell (bit r of c1): the monomino,
    plus the 4 T-tetromino patterns that are geometrically possible given
    the bottom-up invariant (the other 12 named patterns in the original
    code become impossible once 'everything below r is filled' is fixed)."""
    moves = [0]  # 0 = monomino, always legal
    # vertical bar in c1 (rows r,r+1,r+2), stem into c2 at r+1
    if r + 2 < w and not (c1 & (1 << (r+1))) and not (c1 & (1 << (r+2))) and not (c2 & (1 << (r+1))):
        moves.append(4)
    # horizontal bar across c1,c2,c3 at row r, stem into c2 below (r-1)
    if r >= 1 and not (c2 & (1 << r)) and not (c3 & (1 << r)) and not (c2 & (1 << (r-1))):
        moves.append(5)
    # horizontal bar across c1,c2,c3 at row r, stem into c2 above (r+1)
    if r + 1 < w and not (c2 & (1 << r)) and not (c3 & (1 << r)) and not (c2 & (1 << (r+1))):
        moves.append(9)
    # vertical bar in c2 (rows r-1,r,r+1), stem reaching into c1 at r
    if r >= 1 and r + 1 < w and not (c2 & (1 << (r-1))) and not (c2 & (1 << r)) and not (c2 & (1 << (r+1))):
        moves.append(15)
    return moves

def apply_move(c1, c2, c3, r, move):
    if move == 0:
        return c1 | (1 << r), c2, c3
    if move == 4:
        return c1 | (1 << r) | (1 << (r+1)) | (1 << (r+2)), c2 | (1 << (r+1)), c3
    if move == 5:
        return c1 | (1 << r), c2 | (1 << r) | (1 << (r-1)), c3 | (1 << r)
    if move == 9:
        return c1 | (1 << r), c2 | (1 << r) | (1 << (r+1)), c3 | (1 << r)
    if move == 15:
        return c1 | (1 << r), c2 | (1 << (r-1)) | (1 << r) | (1 << (r+1)), c3

def churn(c1, c2, c3, w):
    """Shift by 0, 1, or 2 columns, whichever is needed to restore the
    invariant that the current column isn't already fully filled."""
    FULL = (1 << w) - 1
    if c1 == FULL and c2 == FULL and c3 == FULL:
        return 0, 0, 0
    if c1 == FULL and c2 == FULL:
        return c3, 0, 0
    if c1 == FULL:
        return c2, c3, 0
    return c1, c2, c3

def build_fringe_graph(w):
    """Returns:
      fringes: list of nodes (index -> (c1,c2,c3) state)
      edges:   list of (from_idx, to_idx, cost, move) directed edges,
               cost = 1 for a monomino move, 0 for a tetromino move
    """
    start = (0, 0, 0)
    fringes = [start]
    index = {start: 0}
    edges = []
    i = 0
    while i < len(fringes):
        c1, c2, c3 = fringes[i]
        r = lowest_open(c1, w)
        for move in legal_moves(c1, c2, c3, r, w):
            nc1, nc2, nc3 = apply_move(c1, c2, c3, r, move)
            nc1, nc2, nc3 = churn(nc1, nc2, nc3, w)
            key = (nc1, nc2, nc3)
            if key not in index:
                index[key] = len(fringes)
                fringes.append(key)
            cost = 1 if move == 0 else 0
            edges.append((i, index[key], cost, move))
        i += 1
    return fringes, edges

def weighted_edges(edges, m):
    """Hochberg's weighting: -1 on every T-edge (tetromino move),
    m on every M-edge (monomino move). move==0 is the monomino."""
    weighted = []
    for u, v, cost, move in edges:
        weight = m if move == 0 else -1
        weighted.append((u, v, weight))
    return weighted

def shortest_bounded_path(nodes, weighted_edge_list, source, target, max_length=40):
    """Shortest path from source to target using at most max_length edges
    (via the same layered-DP idea, but capped short enough that it can't
    sneak an extra lap around a negative cycle to cheat the answer)."""
    INF = float('inf')
    adj = {}
    for u, v, w in weighted_edge_list:
        adj.setdefault(u, []).append((v, w))
    d_prev = {v: INF for v in nodes}
    d_prev[source] = 0.0
    pred_layers = []
    best_k, best_val = None, d_prev.get(target, INF)
    if source == target:
        return [source], 0.0
    for k in range(1, max_length + 1):
        d_cur = {v: INF for v in nodes}
        pred_cur = {}
        for u, du in d_prev.items():
            if du == INF:
                continue
            for v, w_ in adj.get(u, []):
                nd = du + w_
                if nd < d_cur[v]:
                    d_cur[v] = nd
                    pred_cur[v] = u
        pred_layers.append(pred_cur)
        if d_cur[target] < best_val:
            best_val, best_k = d_cur[target], k
        d_prev = d_cur
    if best_k is None:
        return None, None
    path = [target]
    v = target
    for k in range(best_k, 0, -1):
        u = pred_layers[k-1][v]
        path.append(u)
        v = u
    path.reverse()
    return path, best_val

def all_shortest_bounded(nodes, weighted_edge_list, source, max_length=60):
    """Shortest (bounded-length) distance from source to EVERY node, in one
    pass -- same layered-DP as shortest_bounded_path, but keeping every
    node's best distance instead of throwing away all but one target."""
    INF = float('inf')
    adj = {}
    for u, v, w in weighted_edge_list:
        adj.setdefault(u, []).append((v, w))
    d_prev = {v: INF for v in nodes}
    d_prev[source] = 0.0
    best = dict(d_prev)
    for _ in range(max_length):
        d_cur = {v: INF for v in nodes}
        for u, du in d_prev.items():
            if du == INF:
                continue
            for v, w_ in adj.get(u, []):
                nd = du + w_
                if nd < d_cur[v]:
                    d_cur[v] = nd
        for v in nodes:
            if d_cur[v] < best[v]:
                best[v] = d_cur[v]
        d_prev = d_cur
    return best

# ---------------------------------------------------------------------------
# Bellman-Ford: smallest M-edge weight m that makes every cycle nonnegative
# ---------------------------------------------------------------------------
#
# With T-edges weighted -1 and M-edges weighted m, a cycle with tau T-edges
# and mu M-edges has total weight  m*mu - tau.  Every cycle is nonnegative
# exactly when  m >= tau/mu  for every cycle, so the answer is
#
#     m* = max over all cycles of tau/mu.
#
# We find it exactly (as a fraction) with this loop:
#   1. pick a candidate m = p/q (start at 0)
#   2. run Bellman-Ford with integer weights  q*m = p  on M-edges and  -q  on
#      T-edges (same signs as m and -1, but no floating point anywhere)
#   3. no negative cycle  -> m is the answer, stop
#      negative cycle C    -> tau_C/mu_C > m, so raise m to tau_C/mu_C, repeat
# m goes up strictly every round and there are finitely many cycles, so it
# terminates. When it stops, the last cycle found has weight exactly 0 at m,
# so no smaller m can work: m is the minimum, not just an upper bound.
#
# Speed: instead of always running the full N-1 passes, after each pass we
# look for a cycle in the predecessor graph. Any cycle there is guaranteed to
# be a negative cycle (standard Bellman-Ford lemma), so we can stop as soon as
# one appears. No pass cap or guesswork is involved: if nothing changes in a
# pass there is no negative cycle; if the predecessor graph has a cycle there
# is one; and within N passes one of the two always happens.

from fractions import Fraction


def _pred_cycle(pred_node, n):
    """Return the nodes of a cycle in the predecessor graph (each node has at
    most one predecessor), or None if it is acyclic. O(n)."""
    state = [0] * n          # 0 = unseen, otherwise id of the walk that saw it
    for s in range(n):
        if state[s]:
            continue
        walk_id = s + 1
        v = s
        while v != -1 and not state[v]:
            state[v] = walk_id
            v = pred_node[v]
        if v != -1 and state[v] == walk_id:     # came back into this walk
            cyc = [v]
            u = pred_node[v]
            while u != v:
                cyc.append(u)
                u = pred_node[u]
            return cyc
    return None


def bellman_ford_negative_cycle(n, src, dst, wt):
    """Bellman-Ford from a virtual source (dist 0 everywhere), so a negative
    cycle anywhere in the graph is found. Edges are parallel lists src/dst/wt
    with integer weights. Returns (edge indices of a negative cycle, in order)
    or None if there is no negative cycle."""
    dist = [0] * n
    pred_node = [-1] * n
    pred_edge = [-1] * n
    E = len(src)
    for _ in range(n):
        changed = False
        for i in range(E):
            u = src[i]; v = dst[i]
            nd = dist[u] + wt[i]
            if nd < dist[v]:
                dist[v] = nd
                pred_node[v] = u
                pred_edge[v] = i
                changed = True
        if not changed:
            return None
        cyc = _pred_cycle(pred_node, n)
        if cyc is not None:
            # cyc walks backwards along predecessors; reverse into edge order
            cyc.reverse()
            return [pred_edge[v] for v in cyc[1:] + cyc[:1]]
    cyc = _pred_cycle(pred_node, n)          # after n passes one must exist
    cyc.reverse()
    return [pred_edge[v] for v in cyc[1:] + cyc[:1]]


def min_nonnegative_m(fringes, edges, verbose=True):
    """Smallest m (exact Fraction) such that, with T-edges = -1 and
    M-edges = m, every cycle has nonnegative total weight.
    Returns (m, critical_cycle_edge_indices, tau, mu)."""
    n = len(fringes)
    src = [u for u, v, c, mv in edges]
    dst = [v for u, v, c, mv in edges]
    is_M = [mv == 0 for u, v, c, mv in edges]

    m = Fraction(0)
    critical = None
    while True:
        p, q = m.numerator, m.denominator
        wt = [p if M else -q for M in is_M]      # q * (m or -1)
        cyc = bellman_ford_negative_cycle(n, src, dst, wt)
        if cyc is None:
            if verbose:
                print(f"    m = {m}: no negative cycle  -> done")
            break
        tau = sum(1 for i in cyc if not is_M[i])
        mu = len(cyc) - tau
        assert sum(wt[i] for i in cyc) < 0      # sanity: it really is negative
        if mu == 0:
            raise ValueError("cycle made only of T-edges: it is negative for "
                             "every m, so no finite m works")
        if verbose:
            print(f"    m = {m}: negative cycle with {tau} T-edges, {mu} "
                  f"M-edges (weight {m}*{mu} - {tau} = {m*mu - tau})  "
                  f"-> raise m to {tau}/{mu} = {Fraction(tau, mu)}")
        m = Fraction(tau, mu)
        critical = (cyc, tau, mu)
    if critical is None:
        return m, None, 0, 0
    return m, critical[0], critical[1], critical[2]


def show_cycle_edges(fringes, edges, cyc, m):
    for i in cyc:
        u, v, c, mv = edges[i]
        name = "M (monomino)" if mv == 0 else f"T (tetromino p{mv})"
        weight = m if mv == 0 else -1
        print(f"      fringe {u} {fringes[u]} --[{name}, w={weight}]--> "
              f"fringe {v} {fringes[v]}")

if __name__ == "__main__":
    import time
    for w in (3, 5, 7, 9, 11):
        t0 = time.time()
        fringes, edges = build_fringe_graph(w)
        m_edges = sum(1 for u, v, cost, move in edges if move == 0)
        t_edges = sum(1 for u, v, cost, move in edges if move != 0)
        print(f"width {w}: {len(fringes)} fringes, {len(edges)} edges "
              f"({m_edges} M-edges, {t_edges} T-edges)")

        m, cyc, tau, mu = min_nonnegative_m(fringes, edges)
        print(f"  smallest m with every cycle >= 0:  m = {m}"
              + (f"  (~{float(m):.4f})" if m.denominator != 1 else ""))
        if cyc is not None:
            print(f"  critical cycle (weight exactly 0 at this m): "
                  f"{tau} T-edges, {mu} M-edges, {len(cyc)} edges total")
            if len(cyc) <= 12:
                show_cycle_edges(fringes, edges, cyc, m)
        print(f"  ({time.time() - t0:.1f}s)\n")
