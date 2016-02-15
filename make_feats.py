__author__ = 'arenduchintala'

import codecs
import itertools
import json
import pdb
import sys
from optparse import OptionParser

import edu.jhu.hlt.optimize.AdaGradComidL2.AdaGradComidL2Prm as AdaGradComidL2Prm
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BeliefPropagationPrm as BeliefPropagationPrm
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpScheduleType as BpScheduleType
import edu.jhu.pacaya.gm.inf.BeliefPropagation.BpUpdateOrder as BpUpdateOrder
import edu.jhu.pacaya.gm.train.CrfTrainer.CrfTrainerPrm as CrfTrainerPrm
import edu.jhu.pacaya.util.semiring.LogSemiring as LogSemiring
from edu.jhu.hlt.optimize import AdaGradComidL2
from edu.jhu.pacaya.gm.data import LabeledFgExample, FgExampleDiskStore
from edu.jhu.pacaya.gm.feat import FeatureVector
from edu.jhu.pacaya.gm.model import FactorGraph, Var, VarSet, ExplicitExpFamFactor, VarConfig, \
    ClampFactor

from arpa_lm import BLM
from training_classes import TrainingInstance

global TT_FACTOR, TO_FACTOR, TC_FACTOR, factor_cell_to_features, feature_idxs, BLANK, CORRECT_BLANK
TT_FACTOR = 'tt_factor'
TO_FACTOR = 'to_factor'
TC_FACTOR = 'tc_factor'
BLANK = '__BLANK__'
CORRECT_BLANK = '__CORRECT_BLANK__'
factor_cell_to_features = {}
feature_idxs = {}
global obs2id, tag2id, id2obs, id2tag
obs2id = {}
tag2id = {}
id2obs = {}
id2tag = {}
'''
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'
'''


class ObservedFactor(ExplicitExpFamFactor):
    def __init__(self, varset, var_list, factor_type, observed_state):
        ExplicitExpFamFactor.__init__(self, varset)
        self.var_list = var_list
        self.factor_type = factor_type
        self.observed_state = observed_state

    def getFeatures(self, configuration_id):
        vs = self.getVars()
        configuration = vs.getVarConfig(configuration_id)
        state1 = configuration.getStateName(self.var_list[0])
        # print 'vs:' , vs.calcNumConfigs()
        # print 'config_id:', configuration_id, 'config:' , state1, self.factor_type
        # print 'vars:' , self.var_list[0].name, self.observed_state
        feats_fired = factor_cell_to_features[(self.factor_type, state1, self.observed_state)]
        feat_idxs = [(feature_idxs[f_label], f_val) for f_label, f_val in feats_fired]
        feats = zip(*feat_idxs)
        return FeatureVector(list(feats[0]), list(feats[1]))


class CRFFactor(ExplicitExpFamFactor):
    def __init__(self, varset, var_list, factor_type):
        ExplicitExpFamFactor.__init__(self, varset)
        self.var_list = var_list
        self.factor_type = factor_type

    def getFeatures(self, configuration_id):
        vs = self.getVars()
        configuration = vs.getVarConfig(configuration_id)
        state1 = configuration.getStateName(self.var_list[0])
        state2 = configuration.getStateName(self.var_list[1])
        # print 'vs:' , vs.calcNumConfigs()
        # print 'config_id:', configuration_id, 'config:' , state1, state2, self.factor_type
        # print 'vars:' , self.var_list[0].name, self.var_list[1].name
        feats_fired = factor_cell_to_features[(self.factor_type, state1, state2)]
        feat_idxs = [(feature_idxs[f_label], f_val) for f_label, f_val in feats_fired]
        feats = zip(*feat_idxs)
        return FeatureVector(list(feats[0]), list(feats[1]))


class Clamper(ClampFactor):
    def __init__(self, var, var_state):
        ClampFactor.__init__(self, var, var_state)


def get_trainer_prm():
    tr_prm = CrfTrainerPrm()
    ad_prm = AdaGradComidL2Prm()
    ad_prm.numPasses = 2  # Number of passes through the data
    ad_prm.l2Lambda = 1. / 2000.  # L2 regularizer weight
    tr_prm.batchOptimizer = AdaGradComidL2(ad_prm)
    # We can use brute force inference because the factor graph
    # consists of only a single variable and factor.
    # tr_prm.infFactory = BruteForceInferencerPrm(LogSemiring.getInstance())
    tr_prm.infFactory = BeliefPropagationPrm()
    tr_prm.infFactory.s = LogSemiring.getInstance()
    tr_prm.infFactory.schedule = BpScheduleType.TREE_LIKE
    tr_prm.infFactory.updateOrder = BpUpdateOrder.SEQUENTIAL
    tr_prm.infFactory.normalizeMessages = True
    tr_prm.infFactory.maxIterations = 1;
    tr_prm.infFactory.convergenceThreshold = 1e-3;
    tr_prm.infFactory.keepTape = True
    return tr_prm


def add_to_tags(t):
    global id2tag, tag2id
    tag2id[t] = tag2id.get(t, len(tag2id))
    id2tag[tag2id[t]] = t


def add_to_obs(o):
    global obs2id, id2obs
    obs2id[o] = obs2id.get(o, len(obs2id))
    id2obs[obs2id[o]] = o


def get_str(i):
    return '%.2f' % i


def populate_features():
    global feature_idxs, factor_cell_to_features, tag2id, id2tag, obs2id, id2obs
    for t1, t2 in itertools.product(tag2id.keys(), tag2id.keys()):
        feats = get_transition_features_labels(t1, t2)
        for f in feats:
            feature_idxs[f] = feature_idxs.get(f, len(feature_idxs))
        factor_cell_to_features[(TT_FACTOR, t1, t2)] = feats

    for t, o in itertools.product(tag2id.keys(), obs2id.keys()):
        feats = get_transition_features_labels(t, o)
        for f in feats:
            feature_idxs[f] = feature_idxs.get(f, len(feature_idxs))
        factor_cell_to_features[(TO_FACTOR, t, o)] = feats


def get_transition_features_labels(t1, t2):
    f_name = t1 + '|||' + t2
    return [f_name]


def get_emission_feature_labels(t, o):
    f_name = t + '|||' + o

    if t != BLANK:
        # f_ed = t + '|||' + o + '|||ed'
        f_pr = o + '|||prguess'

    f_pw = o + '|||pwguess'
    return [f_name, f_pr, f_pw]


def get_transition_features(t1, t2, blm):
    lm_prb = blm.get_prob(t1.strip() + ' ' + t2.strip())
    lm_prb = 1.0 / - lm_prb
    f_name = t1 + '|||' + t2
    f_val = get_str(lm_prb)
    return '\t'.join([f_name, f_val])


def get_emission_features(t, o, ed, pr_guesses, pw_guesses):
    features = []
    f_name = t + '|||' + o
    if t == BLANK:
        f_val = 0.0001
    else:
        f_val = get_str(ed.editdistance_simple(t, o)[0] / max(len(t), len(o)))

    features += [f_name, get_str(f_val)]
    if o in pr_guesses and t != BLANK:
        features += [o + '|||prguess', '1.0']
    if o in pw_guesses:
        features += [o + '|||pwguess', '1.0']
    return "\t".join(features)


def get_instance(ti_line):
    j = json.loads(ti_line)
    ti = TrainingInstance.from_dict(j)
    obs = [o.l2_word for o in ti.current_sent if o.lang == 'de' if o.l2_word.strip() != '']
    guesses = [g.guess for g in ti.current_guesses if g.guess.strip() != '']
    guesses += [o.l2_word for o in ti.current_sent if o.lang == 'en' if o.l2_word.strip() != '']  # wink! ;)
    guesses += [g.guess for g in ti.past_correct_guesses if g.guess.strip() != '']
    guesses += [g.guess for g in ti.past_guesses_for_current_sent if g.guess.strip() != '']
    return ti, obs, guesses


def make_fg_instances(training_instances, tag_ids, obs_ids):
    global tag2id, id2tag, obs2id, id2obs
    instances = FgExampleDiskStore()
    num = 0
    for ti in training_instances:
        print 'num:', num
        num += 1
        print 'new training instance:', ti.current_sent[0].sent_id
        print 'usename:', ti.user_id
        print 'experience:', len(ti.past_correct_guesses), len(ti.past_sentences_seen)
        print 'immediate experience:', len(ti.past_guesses_for_current_sent)
        factor_graph = FactorGraph()
        vc = VarConfig()
        curr_g = {}
        curr_s = {}
        past_g_sent = {}
        curr_rg = {}

        for crg in ti.current_revealed_guesses:
            curr_rg[tuple(crg.id)] = crg

        for psg in ti.past_guesses_for_current_sent:
            past_g_sent[tuple(psg.id)] = psg

        for cg in ti.current_guesses:
            curr_g[tuple(cg.id)] = cg

        curr_s_order = []
        for cs in ti.current_sent:
            curr_s[tuple(cs.id)] = cs
            curr_s_order.append((int(cs.position), cs))
        curr_s_order.sort()

        prev_hc = None
        for cso_pid, cso in curr_s_order:
            cso_id = tuple(cso.id)
            hc = Var(Var.VarType.PREDICTED, len(tag_ids), 'TAG_' + str(cso_pid), tag_ids)
            if cso.lang == 'en':
                # has no emission factor as hc is clamped, e_factor's messages will not affect belief
                hidden_state = cso.l2_word
                hidden_state = hidden_state if hidden_state.strip() != '' else CORRECT_BLANK
                c_factor = Clamper(hc, tag2id[hidden_state])
                factor_graph.addFactor(c_factor)
                vc.put(hc, tag2id[hidden_state])
                print hidden_state, 'is already in english i.e revealed'
            elif cso_id in curr_rg:
                # has been revealed so hidden var should be clamped to the correct guess
                hidden_state = curr_rg[cso_id].guess
                hidden_state = hidden_state if hidden_state.strip() != '' else CORRECT_BLANK
                c_factor = Clamper(hc, tag2id[hidden_state])
                factor_graph.addFactor(c_factor)
                vc.put(hc, tag2id[hidden_state])
                # e_factor = ObservedFactor(VarSet(hc), [hc], TO_FACTOR, cso.l2_word)
                # factor_graph.addFactor(e_factor)
                print hidden_state, 'is the revealed guess at position', cso_pid, 'sn id', cso_id
                # also has no need for emission factor since hc is clamped...
            else:
                if cso_id not in curr_g:
                    pdb.set_trace()
                # has not yet been revealed, this is a genuine guess
                hidden_state = curr_g[cso_id].guess
                hidden_state = hidden_state if hidden_state.strip() != '' else BLANK
                print hidden_state, 'is the current guess at position', cso_pid, 'sn id', cso_id
                vc.put(hc, tag2id[hidden_state])
                e_factor = ObservedFactor(VarSet(hc), [hc], TO_FACTOR, obs2id[cso.l2_word])
                factor_graph.addFactor(e_factor)

            if prev_hc:
                t_varset = VarSet(hc)
                t_varset.add(prev_hc)
                t_factor = CRFFactor(t_varset, [prev_hc, hc], TT_FACTOR)
                factor_graph.addFactor(t_factor)

            prev_hc = hc
        instances.add(LabeledFgExample(factor_graph, vc))
    return instances


if __name__ == '__main__':
    opt = OptionParser()
    # insert options here
    opt.add_option('--test', dest='test_file', default='')
    opt.add_option('--train', dest='train_file', default='')
    opt.add_option('--lm', dest='lm', default='')
    opt.add_option('--base', dest='base_vocab', default='')
    opt.add_option('--base_size', dest='base_size', default=10)
    (options, _) = opt.parse_args()
    if options.test_file.strip() == '' \
            or options.train_file.strip() == '' \
            or options.base_vocab.strip() == '' \
            or options.base_size == '':
        sys.stderr.write(
                "Usage: jython make_feats.py "
                "--train [train file] "
                "--test [test file] "
                "--base [base vocab file] "
                "--base_size [base size default=1000]\n")
        sys.exit(1)
    #blm = BLM(options.lm)
    obs_set = set([])
    base = [i.lower().strip() for i in
            codecs.open(options.base_vocab, 'r', 'utf8').readlines()[:int(options.base_size)]]
    tag2id = {}
    add_to_tags(BLANK)
    add_to_tags(CORRECT_BLANK)
    for b in base:
        add_to_tags(b)

    training_ti = []
    testing_ti = []
    for line in codecs.open(options.train_file, 'r', 'utf8').readlines():
        ti, obs, guess = get_instance(line)
        for g in guess:
            add_to_tags(g)
        for o in obs:
            add_to_obs(o)
        training_ti.append(ti)

    for line in open(options.test_file).readlines():
        ti, obs, guess = get_instance(line)
        for g in guess:
            add_to_tags(g)
        for o in obs:
            add_to_obs(o)
        testing_ti.append(ti)
    print 'len tags:', len(tag2id)
    print 'len obs:', len(obs2id)
    print 'populating features...'
    populate_features()
    make_fg_instances(training_ti, tag_ids=id2tag.keys(), obs_ids=id2obs.keys())
