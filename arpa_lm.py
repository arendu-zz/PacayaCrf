import codecs


class BLM(object):
    def __init__(self, file_path):
        o = codecs.open(file_path, 'r', 'utf8').read().split('\\1-grams:\n')[1]
        b = o.split('\\2-grams:\n')[1]
        u = o.split('\\2-grams:\n')[0]
        self.backoff = {}
        self.prob = {}
        for ul in u.split('\n'):
            if ul.strip() != '' and len(ul.split('\t')) == 3:
                [p, uni, bow] = ul.split('\t')
                self.backoff[uni.strip()] = float(bow)
                self.prob[uni.strip()] = float(p)
        for bl in b.split('\n'):
            if bl.strip() != '' and len(bl.split('\t')) == 2:
                [p, bi] = bl.split('\t')
                self.prob[bi.strip()] = float(p)

    def get_prob(self, s):
        s = s.strip()
        if s in self.prob:
            return self.prob[s]
        else:
            [s1, s2] = s.split()
            return self.prob[s2.strip()] + self.backoff[s1.strip()]
