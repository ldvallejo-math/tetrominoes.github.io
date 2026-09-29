"""
ILP for M(w, n): the minimum number of monominoes needed to tile a w x n
rectangle with T-tetrominoes and monominoes.

Formulation (exact cover):
    P       = set of all placements of a T-tetromino on the w x n grid
               (4 rotations x every translation that fits inside the grid)
    S_p     = the 4 cells covered by placement p in P
    x_p     in {0,1}  for each p in P   -- 1 if that T-tetromino is used
    y_(i,j) in {0,1}  for each cell     -- 1 if a monomino covers (i,j)

    minimize   sum_(i,j) y_(i,j)
    subject to sum_{p in P : (i,j) in S_p} x_p + y_(i,j) = 1   for every cell (i,j)
               x_p, y_(i,j) in {0,1}

Requires: pulp 2.x   (pip install "pulp<3")   -- uses the CBC solver bundled
with it. PuLP 4.0 changed the LpVariable API and this script fails there with
"TypeError: ... unexpected keyword argument 'cat'". Tested with PuLP 2.9.0.

Optimality check: by default, CBC (like most MIP solvers) stops branch-and-
bound as soon as the relative gap between the best integer solution found
and the best proven bound falls under a small tolerance (~1e-4 relative),
not necessarily at an exact 0.0 gap. That's a fine default for large
industrial models, but it means "Optimal" out of the box is "provably
within ~0.01% of optimal", not "provably exactly optimal". Here we force
gapRel=0 and gapAbs=0, so CBC will not report Optimal until the best bound
and the best integer solution coincide exactly -- an exact optimality
certificate, not an approximate one. (For an integer-valued objective like
this one, the default tolerance already can't hide a wrong answer -- but
setting the gap to 0 makes the guarantee explicit rather than incidental.)
"""

import os
import tempfile

import pulp


# The T-tetromino's 4 distinct orientations (rotations only; its reflections
# coincide with rotations, so there are exactly 4, not 8). Each is given as a
# set of (row, col) offsets within its own bounding box, box top-left = (0,0).
ORIENTATIONS = [
    {(0, 0), (0, 1), (0, 2), (1, 1)},   # point-down, 2 rows x 3 cols
    {(0, 1), (1, 0), (1, 1), (1, 2)},   # point-up,   2 rows x 3 cols
    {(0, 0), (1, 0), (2, 0), (1, 1)},   # point-right,3 rows x 2 cols
    {(1, 0), (0, 1), (1, 1), (2, 1)},   # point-left, 3 rows x 2 cols
]


def all_placements(w, n):
    """All valid T-tetromino placements on a w (rows) x n (cols) grid.

    Returns a list of frozensets of (row, col) cells, each of size 4.
    """
    placements = []
    for shape in ORIENTATIONS:
        H = max(r for r, c in shape) + 1
        W = max(c for r, c in shape) + 1
        for r0 in range(w - H + 1):
            for c0 in range(n - W + 1):
                cells = frozenset((r0 + dr, c0 + dc) for (dr, dc) in shape)
                placements.append(cells)
    return placements


def solve_tiling(w, n, msg=False, time_limit=None):
    """Solve the exact-cover ILP for a w x n rectangle, requiring an exact
    (zero-gap) optimality proof from the solver.

    Returns a dict with:
        status        : pulp status string ("Optimal", "Infeasible", ...)
        M             : optimal number of monominoes (None if not solved to optimality)
        t_placements  : list of the chosen T-tetromino placements (each a set of cells)
        monominoes    : list of (row, col) cells covered by a monomino
        truly_optimal : True only if BOTH pulp's sol_status and CBC's own log
                         say optimality was proven (see the comment in the body)
        hit_time_limit: True if CBC's log says it stopped on the time limit

    Note: if you pass a time_limit and CBC hits it before closing the gap to
    zero, status will NOT be "Optimal" (it reports the honest state instead
    of a solution it can't yet prove is optimal) -- M stays None in that
    case. That's the point of forcing the gap to 0: you never get an
    "Optimal" label on a solution that was only good enough by tolerance.
    """
    cells = [(i, j) for i in range(w) for j in range(n)]
    placements = all_placements(w, n)

    prob = pulp.LpProblem("T_tetromino_tiling", pulp.LpMinimize)

    x = {p: pulp.LpVariable(f"x_{idx}", cat="Binary") for idx, p in enumerate(placements)}
    y = {cell: pulp.LpVariable(f"y_{cell[0]}_{cell[1]}", cat="Binary") for cell in cells}

    # Objective: minimize number of monominoes
    prob += pulp.lpSum(y[c] for c in cells)

    # Exact cover: every cell is covered exactly once, by a tetromino or a monomino
    cover = {c: [] for c in cells}
    for p in placements:
        for c in p:
            cover[c].append(x[p])

    for c in cells:
        prob += pulp.lpSum(cover[c]) + y[c] == 1, f"cover_{c[0]}_{c[1]}"

    # gapRel=0, gapAbs=0: don't accept "close enough" -- force CBC to keep
    # branching until the best bound and best incumbent are exactly equal.
    #
    # CONFIRMED BUG (empirically, w=7 n=18): PuLP's own LpStatus mapping is
    # NOT reliable when a time_limit is given. If CBC is stopped by the time
    # limit while holding a feasible-but-unproven incumbent, PuLP can still
    # report LpStatusOptimal ("Optimal") even though CBC's own log literally
    # says "Result - Stopped on time limit" with a nonzero gap between the
    # incumbent and the best bound (seen: objective 6.0, lower bound 2.0,
    # gap 2.0 -- yet pulp.LpStatus[prob.status] == "Optimal"). So: never
    # trust pulp's status string alone for a time-limited solve.
    #
    # Two independent checks, and BOTH must pass:
    #   1. prob.sol_status == pulp.LpSolutionOptimal (1). PuLP does record the
    #      difference here: in the time-limit case above, LpStatus reads
    #      "Optimal" but sol_status is 2 (LpSolutionIntegerFeasible = found a
    #      solution, did not prove it optimal). Checked with PuLP 2.9.0.
    #   2. CBC's raw log contains "Optimal solution found". Capture
    # CBC's raw log via logPath (contextlib/redirect_stdout does NOT work --
    # PULP_CBC_CMD shells out to the cbc binary as a subprocess that writes
    # straight to the inherited fd 1, bypassing Python-level sys.stdout
    # redirection entirely) and require the exact phrase
    # "Optimal solution found".
    # Requiring both fails safe: if a future PuLP or CBC version changes the
    # log wording or the sol_status codes, results come back UNTRUSTED
    # instead of being silently accepted.
    log_fd, log_file = tempfile.mkstemp(suffix=".cbc.log")
    os.close(log_fd)
    try:
        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=0,
                                    gapAbs=0, logPath=log_file)
        prob.solve(solver)
        with open(log_file) as f:
            log_text = f.read()
    finally:
        try:
            os.remove(log_file)
        except OSError:
            pass

    pulp_status = pulp.LpStatus[prob.status]
    sol_proven = getattr(prob, "sol_status", None) == pulp.LpSolutionOptimal
    log_proven = "Optimal solution found" in log_text
    truly_optimal = sol_proven and log_proven
    hit_time_limit = "Stopped on time limit" in log_text

    result = {
        "status": pulp_status, "M": None, "t_placements": [], "monominoes": [],
        "truly_optimal": truly_optimal,
        "hit_time_limit": hit_time_limit,
    }

    if pulp_status == "Optimal" and not truly_optimal:
        # This is exactly the mislabeling bug described above -- do not
        # trust it. Report the honest state instead.
        result["status"] = "UNTRUSTED (pulp said Optimal, optimality not proven)"
        return result

    if truly_optimal and pulp_status != "Optimal":
        # Belt and suspenders: if both checks proved optimality but pulp's
        # status string disagrees, trust the checks.
        pulp_status = "Optimal"
        result["status"] = "Optimal"

    if pulp_status == "Optimal":
        result["M"] = int(round(pulp.value(prob.objective)))
        result["t_placements"] = [p for p in placements if pulp.value(x[p]) > 0.5]
        result["monominoes"] = [c for c in cells if pulp.value(y[c]) > 0.5]
        # Never hand back an "Optimal" result without checking the solver's own
        # output is a real tiling. This catches formulation bugs (wrong shape,
        # double-counted cell, etc.) that a zero-gap proof would NOT catch --
        # a wrong formulation can be "solved to exact optimality" and still
        # not correspond to a legal tiling.
        ok, reason = verify_solution(w, n, result)
        if not ok:
            raise RuntimeError(
                f"solve_tiling({w},{n}) returned status=Optimal but the solution "
                f"failed verification: {reason}. Do not trust this M value -- "
                f"there is a bug in the ILP formulation, not the solver's proof."
            )

    return result


def verify_solution(w, n, result):
    """Check a solve_tiling() result is an actual valid tiling: every
    T-tetromino placement is really a T-shape, no two pieces overlap, and
    every cell of the w x n grid is covered exactly once.

    This is a *feasibility* check, not an *optimality* check -- it can't
    tell you M is the minimum (that's what the solver's zero-gap "Optimal"
    status proves). It exists because optimality and legality are proved by
    two different things here: the solver proves the ILP's optimum is found;
    this proves the ILP you wrote is actually the tiling problem.

    Returns (True, "") if valid, else (False, "<description of the problem>").
    """
    covered = {}
    for p in result["t_placements"]:
        if len(p) != 4:
            return False, f"placement {p} does not have 4 cells"
        minr = min(r for r, c in p)
        minc = min(c for r, c in p)
        norm = frozenset((r - minr, c - minc) for r, c in p)
        if norm not in ORIENTATIONS:
            return False, f"placement {p} is not a valid T-tetromino orientation"
        for cell in p:
            if not (0 <= cell[0] < w and 0 <= cell[1] < n):
                return False, f"placement {p} has a cell outside the {w}x{n} grid"
            if cell in covered:
                return False, f"cell {cell} is covered twice ({covered[cell]} and this tetromino)"
            covered[cell] = "tetromino"
    for cell in result["monominoes"]:
        if cell in covered:
            return False, f"cell {cell} is covered twice ({covered[cell]} and a monomino)"
        covered[cell] = "monomino"
    all_cells = {(i, j) for i in range(w) for j in range(n)}
    missing = all_cells - set(covered)
    if missing:
        return False, f"cells never covered: {sorted(missing)}"
    if len(result["monominoes"]) != result["M"]:
        return False, "reported M does not match the actual number of monominoes in the solution"
    return True, ""


def print_tiling(w, n, result):
    """ASCII-render a solved tiling: each T-tetromino placement gets its own
    letter (cycling), monominoes are shown as '.'.
    """
    grid = [["?"] * n for _ in range(w)]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for idx, p in enumerate(result["t_placements"]):
        ch = letters[idx % len(letters)]
        for (r, c) in p:
            grid[r][c] = ch
    for (r, c) in result["monominoes"]:
        grid[r][c] = "."
    for row in grid:
        print("".join(row))


if __name__ == "__main__":
    # Quick sanity checks
    for (w, n) in [(4, 4), (4, 8), (5, 5), (5, 10), (7, 7)]:
        res = solve_tiling(w, n)
        print(f"M({w},{n}) = {res['M']}  [{res['status']}, gap forced to 0]")
        print_tiling(w, n, res)
        print()
