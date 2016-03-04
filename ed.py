def edsimple(a, b):
    a = a
    b = b
    substitution_penalty = 1
    inset_penalty = 1
    delete_penalty = 1
    # table = np.zeros((len(a) + 1, len(b) + 1), dtype=float)
    table = [[0 for j in xrange(0, len(b) + 1)] for i in xrange(0, len(a) + 1)]
    came_from = {}
    # table = np.ones((len(a) + 1, len(b) + 1))
    for i in range(len(a) + 1):
        table[i][0] = delete_penalty * i  # i
        came_from[i, 0] = (i - 1, 0), (a[i - 1], '<eps>')

    for j in range(len(b) + 1):
        table[0][j] = inset_penalty * j  # j
        came_from[0, j] = (0, j - 1), ('<eps>', b[j - 1])

    # print 'start'
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                diag = table[i - 1][j - 1] + 0.0
            else:
                diag = table[i - 1][j - 1] + substitution_penalty  # substitution cost
            top = table[i][j - 1] + inset_penalty  # insertion cost
            left = table[i - 1][j] + delete_penalty  # deletion cost

            best, prev, tok = min((diag, (i - 1, j - 1), (a[i - 1], b[j - 1])),
                                  (top, (i, j - 1), ('<eps>', b[j - 1])),
                                  (left, (i - 1, j), (a[i - 1], '<eps>')))

            table[i][j] = best
            came_from[i, j] = (prev, tok)
            # print 'current cell', table[i, j]
    # print table

    alignments = bt(came_from, a, b)
    return table[i][j], alignments


def bt(cf, a, b):
    i = len(a)
    j = len(b)
    alignments = []
    while not (i == 0 and j == 0):
        prev, tok = cf[i, j]
        alignments.append((tok[0], tok[1]))  # this is an alignment
        i = prev[0]
        j = prev[1]
    alignments.reverse()
    return alignments
