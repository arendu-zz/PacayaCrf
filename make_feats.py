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
import edu.jhu.pacaya.gm.model.ExpFamFactor as ExFactor
import edu.jhu.pacaya.gm.train.CrfTrainer.CrfTrainerPrm as CrfTrainerPrm
import edu.jhu.pacaya.util.semiring.LogSemiring as LogSemiring
import java.io.FileWriter as FileWriter
from edu.jhu.hlt.optimize import AdaGradComidL2
from edu.jhu.pacaya.gm.data import LabeledFgExample, FgExampleList
from edu.jhu.pacaya.gm.feat import FeatureVector
from edu.jhu.pacaya.gm.model import FactorGraph, FgModel, Var, VarSet, VarConfig
from edu.jhu.pacaya.gm.train import CrfTrainer

from ed import edsimple
from training_classes import TrainingInstance

global TT_FACTOR, TO_FACTOR, TC_FACTOR, factor_cell_to_features, feature_label2id, BLANK, CORRECT_BLANK
TT_FACTOR = u'tt_factor'
TO_FACTOR = u'to_factor'
TC_FACTOR = u'tc_factor'
BLANK = u'__BLANK__'
UNK = u'__UNK__'
CORRECT_BLANK = u'__CORRECT_BLANK__'
factor_cell_to_features = {}
feature_label2id = {}
global obs2id, tag2id, id2obs, id2tag, tag_list, obs_list
obs2id = {}
obs2freq = {}
tag2id = {}
tag2freq = {}
id2obs = {}
id2tag = {}
tag_list = []
obs_list = []
'''
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdin = codecs.getreader('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
sys.stdout.encoding = 'utf-8'
'''


class UnCachedFgList(FgExampleList):
    def __init__(self, training_instanes, states_per_sent):
        FgExampleList.__init__(self)
        self.training_instances = training_instanes
        self.states_per_sentence = states_per_sent

    def size(self):
        return len(self.training_instances)

    def get_hs(self, cso, curr_rg, curr_g):
        cso_id = tuple(cso.id)
        clamp = False
        if cso.lang == 'en':
            # has no emission factor as hc is clamped, e_factor's messages will not affect belief
            hidden_state = cso.l2_word if cso.l2_word.strip() != '' else CORRECT_BLANK
            clamp = True
            # print hidden_state, 'is already in english i.e revealed'
        elif cso_id in curr_rg:
            # has been revealed so hidden var should be clamped to the correct guess
            # hidden_state = curr_rg[cso_id].guess if curr_rg[cso_id].guess.strip() != '' else CORRECT_BLANK
            hidden_state = cso.l1_parent.strip() if cso.l1_parent.strip() != '' else CORRECT_BLANK
            clamp = True
            # print hidden_state, 'is the revealed guess at position', cso_pid, 'sn id', cso_id
        else:
            if cso_id not in curr_g:
                pdb.set_trace()
            # has not yet been revealed, this is a genuine guess
            hidden_state = curr_g[cso_id].guess if curr_g[cso_id].guess.strip() != '' else BLANK
            # print hidden_state, 'is the current guess at position', cso_pid, 'sn id', cso_id

        if ' ' in hidden_state:  # a hackish way to remove a phrasal guess...
            hidden_state_len, hidden_state = sorted([(len(h), h) for h in hidden_state.split()]).pop()
        return clamp, hidden_state

    def get(self, i):
        global tag2id, id2tag, obs2id, id2obs, obs_list
        ti = self.training_instances[i]
        tag_subset = self.states_per_sentence[int(ti.current_sent[0].sent_id)]
        tag_subset.append(BLANK)
        tag_subset.append(CORRECT_BLANK)
        factor_graph = FactorGraph()
        vc = VarConfig()
        curr_g = {}
        curr_s = {}
        past_g_sent = {}
        curr_rg = {}
        curr_s_order = []

        for crg in ti.current_revealed_guesses:
            curr_rg[tuple(crg.id)] = crg

        for psg in ti.past_guesses_for_current_sent:
            past_g_sent[tuple(psg.id)] = psg

        for cg in ti.current_guesses:
            curr_g[tuple(cg.id)] = cg

        for cs in ti.current_sent:
            curr_s[tuple(cs.id)] = cs
            curr_s_order.append((int(cs.position), cs))
        curr_s_order.sort()
        var_map = {}
        fac_map = {}
        # fac_map_summary = {}
        for cso_id0, cso0 in curr_s_order:
            c, hs = self.get_hs(cso0, curr_rg, curr_g)
            # print c, hs
            if hs == BLANK:
                hc_var = Var(Var.VarType.LATENT, len(tag_subset), 'TAG_' + str(cso_id0), tag_subset)
                var_map[cso_id0] = (hc_var, hs, c)
            else:
                hc_var = Var(Var.VarType.PREDICTED, len(tag_subset), 'TAG_' + str(cso_id0), tag_subset)
                var_map[cso_id0] = (hc_var, hs, c)
            if not c:
                try:
                    vc.put(hc_var, hs)
                except:
                    print 'vc is broken...'
                    pdb.set_trace()
                assert cso0.id not in curr_rg
                e_factor = ObservedEFactor(VarSet(hc_var), hc_var, cso0.l2_word)
                fid = (cso_id0, 'er')
                fac_map[fid] = fac_map.get(fid, []) + [e_factor]
                # fac_map_summary[fid] = fac_map_summary.get(fid, []) + [cso0.l2_word + '->emission->' + str(cso_id0)]
            else:
                pass

        for cso_id1, cso1 in curr_s_order:
            for cso_id2, cso2 in curr_s_order:
                if cso_id1 != cso_id2 and cso_id1 - cso_id2 == 1:
                    (v1_hc, v1_hs, v1_c) = var_map[cso_id1]
                    (v2_hc, v2_hs, v2_c) = var_map[cso_id2]
                    if not v1_c and not v2_c:  # both vars are not revealed
                        fid = tuple(sorted([(cso_id1, 'h'), (cso_id2, 'h')]))
                        # there can be only 1 TT factor between 2 unobserved vars
                        if fid not in fac_map:
                            # print fid, 'both are hidden'
                            t_varset = VarSet(v1_hc)
                            t_varset.add(v2_hc)
                            t_factor = TTFactor(t_varset, var1=v1_hc,
                                                var1pos=cso_id1,
                                                var2=v2_hc,
                                                var2pos=cso_id2)

                            fac_map[fid] = fac_map.get(fid, []) + [t_factor]
                            # fac_map_summary[fid] = fac_map_summary.get(fid, []) + [
                            #    str(cso_id1) + '->trans->' + str(cso_id2)]
                    elif not v1_c and v2_c:  # var 2 is reveled i.e. observed
                        fid = tuple(sorted([(cso_id1, 'h'), (cso_id2, 'r')]))
                        if fid not in fac_map:
                            # print cso_id1, ' is hidden', cso_id2, 'is reveled'
                            t_factor = ObservedTFactor(VarSet(v1_hc),
                                                       var1=v1_hc,
                                                       var1pos=cso_id1,
                                                       var2=None,
                                                       var2pos=None,
                                                       observed_state=v2_hs)

                            # there can be multiple factors connected to a v1_hc
                            fac_map[fid] = fac_map.get(fid, []) + [t_factor]
                            # fac_map_summary[fid] = fac_map_summary.get(fid, []) + [
                            #    v2_hs + '->trans->' + str(cso_id1)]
                    elif v1_c and not v2_c:  # var 1 is reveled i.e. observed
                        fid = tuple(sorted([(cso_id1, 'r'), (cso_id2, 'h')]))
                        if fid not in fac_map:
                            # print cso_id2, 'is hidden', cso_id1, 'is reveled'
                            t_factor = ObservedTFactor(VarSet(v2_hc),
                                                       var1=None,
                                                       var1pos=None,
                                                       var2=v2_hc,
                                                       var2pos=cso_id2,
                                                       observed_state=v1_hs)

                            fac_map[fid] = fac_map.get(fid, []) + [t_factor]
                            # fac_map_summary[fid] = fac_map_summary.get(fid, []) + [v1_hs + '->trans->' + str(cso_id2)]
                    else:
                        # this means both v1_hc and v2_hc are reveled, so we dont do anything..
                        pass

        for fid, factors in fac_map.items():
            for _factor in factors:
                try:
                    factor_graph.addFactor(_factor)
                except:
                    print 'something broken when adding factor to factor_graph'
                    pdb.set_trace()
        sys.stderr.write('.')
        return LabeledFgExample(factor_graph, vc)


class ObservedEFactor(ExFactor):
    def __init__(self, varset, var, observed_state):
        global TO_FACTOR, obs_list
        ExFactor.__init__(self, varset)
        assert isinstance(observed_state, str) or isinstance(observed_state, unicode)
        assert observed_state in obs_list
        self.var = var
        self.factor_type = TO_FACTOR

        self.observed_state = observed_state

    def getFeatures(self, configuration_id):
        vs = self.getVars()
        configuration = vs.getVarConfig(configuration_id)
        state1 = configuration.getStateName(self.var)
        feats_fired = factor_cell_to_features[(self.factor_type, state1, self.observed_state)]
        feat_idxs = [(feature_label2id[f_label], 1, 0) for f_label in feats_fired]
        feats = zip(*feat_idxs)
        return FeatureVector(list(feats[0]), list(feats[1]))


class ObservedTFactor(ExFactor):
    def __init__(self, varset, var1, var1pos, var2, var2pos, observed_state):
        ExFactor.__init__(self, varset)
        global TT_FACTOR
        assert isinstance(observed_state, str) or isinstance(observed_state, unicode)
        assert observed_state in tag_list
        self.var1 = var1
        self.var2 = var2
        self.var1pos = var1pos
        self.var2pos = var2pos
        self.factor_type = TT_FACTOR
        self.observed_state = observed_state

    def getFeatures(self, configuration_id):
        vs = self.getVars()
        configuration = vs.getVarConfig(configuration_id)
        if self.var1 is not None:
            hstate = configuration.getStateName(self.var1)
            feats_fired = factor_cell_to_features[(self.factor_type, hstate, self.observed_state)]
        else:
            hstate = configuration.getStateName(self.var2)
            feats_fired = factor_cell_to_features[(self.factor_type, self.observed_state, hstate)]
        feat_idxs = [(feature_label2id[f_label], 1, 0) for f_label in feats_fired]
        feats = zip(*feat_idxs)
        return FeatureVector(list(feats[0]), list(feats[1]))


class TTFactor(ExFactor):
    def __init__(self, varset, var1, var1pos, var2, var2pos):
        ExFactor.__init__(self, varset)
        global TT_FACTOR
        self.var1 = var1
        self.var1pos = var1pos
        self.var2 = var2
        self.var2pos = var2pos
        self.factor_type = TT_FACTOR

    def getFeatures(self, configuration_id):
        vs = self.getVars()
        configuration = vs.getVarConfig(configuration_id)
        state1 = configuration.getStateName(self.var1)
        state2 = configuration.getStateName(self.var2)
        feats_fired = factor_cell_to_features[(self.factor_type, state1, state2)]
        feat_idxs = [(feature_label2id[f_label], 1.0) for f_label in feats_fired]
        feats = zip(*feat_idxs)
        return FeatureVector(list(feats[0]), list(feats[1]))


def get_trainer_prm():
    tr_prm = CrfTrainerPrm()
    ad_prm = AdaGradComidL2Prm()
    ad_prm.eta = 0.8
    ad_prm.batchSize = 50
    ad_prm.autoSelectLr = False
    ad_prm.numPasses = 1  # Number of passes through the data
    ad_prm.l2Lambda = 1. / 200.  # L2 regularizer weight
    tr_prm.batchOptimizer = AdaGradComidL2(ad_prm)
    # We can use brute force inference because the factor graph
    # consists of only a single variable and factor.
    # tr_prm.infFactory = BruteForceInferencerPrm(LogSemiring.getInstance())
    tr_prm.infFactory = BeliefPropagationPrm()
    tr_prm.infFactory.s = LogSemiring.getInstance()
    tr_prm.infFactory.schedule = BpScheduleType.TREE_LIKE
    tr_prm.infFactory.updateOrder = BpUpdateOrder.SEQUENTIAL
    tr_prm.infFactory.normalizeMessages = False
    tr_prm.infFactory.maxIterations = 2
    tr_prm.infFactory.convergenceThreshold = 1e-3
    tr_prm.infFactory.keepTape = False
    return tr_prm


def add_to_tags(t):
    assert ' ' not in t
    global id2tag, tag2id
    if t not in tag2id:
        l = len(tag2id)
        tag2id[t] = l
        id2tag[l] = t
        tag_list.append(t)


def add_to_obs(o):
    assert ' ' not in o
    global obs2id, id2obs
    if o not in obs2id:
        l = len(obs2id)
        obs2id[o] = l
        id2obs[l] = o
        obs_list.append(o)


def get_str(i):
    return '%.2f' % i


def populate_features():
    global feature_label2id, factor_cell_to_features, tag_list, obs_list
    for t1, t2 in itertools.product(tag_list, tag_list):
        feature_labels = get_transition_features_labels(t1, t2)
        for f in feature_labels:
            feature_label2id[f] = feature_label2id.get(f, len(feature_label2id))
        factor_cell_to_features[(TT_FACTOR, t1, t2)] = feature_labels

    for t, o in itertools.product(tag_list, obs_list):
        feature_labels = get_emission_feature_labels(t, o)
        for f in feature_labels:
            feature_label2id[f] = feature_label2id.get(f, len(feature_label2id))
        factor_cell_to_features[(TO_FACTOR, t, o)] = feature_labels


def get_transition_features_labels(t1, t2):
    assert isinstance(t1, str) or isinstance(t1, unicode)
    assert isinstance(t2, str) or isinstance(t2, unicode)
    global id2tag
    ts = sorted([t1, t2])  # The sorting of the tags are done to make the factor cells symmetric
    f_name = 'TRANS|'
    return [f_name, 'TRANS_BIAS']


def get_emission_feature_labels(t, o):
    assert isinstance(t, str) or isinstance(t, unicode)
    assert isinstance(o, str) or isinstance(o, unicode)
    global id2obs, id2tag, ed
    f_fired = []
    ed_Score, ed_alignments = edsimple(t, o)
    for ed_a in ed_alignments:
        f_fired.append('ED|' + str(ed_a))
    f_fired.append('EMISSION_BIAS')
    return f_fired


'''
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
'''


def get_instance(ti_line):
    j = json.loads(ti_line)
    ti = TrainingInstance.from_dict(j)
    obs = [o.l2_word for o in ti.current_sent if o.lang == 'de' if o.l2_word.strip() != '']
    guesses = [g.guess for g in ti.current_guesses if g.guess.strip() != '']
    guesses += [o.l2_word for o in ti.current_sent if o.lang == 'en' if o.l2_word.strip() != '']  # wink! ;)
    guesses += [g.guess for g in ti.past_correct_guesses if g.guess.strip() != '']
    guesses += [g.guess for g in ti.past_guesses_for_current_sent if g.guess.strip() != '']
    guesses = [g.split() for g in guesses]
    obs = [o.split() for o in obs]
    guesses_flat = sum(guesses, [])
    obs_flat = sum(obs, [])

    return ti, obs_flat, guesses_flat


if __name__ == '__main__':
    opt = OptionParser()
    # insert options here
    opt.add_option('--test', dest='test_file', default='')
    opt.add_option('--train', dest='train_file', default='')
    opt.add_option('--states', dest='states_file', default='')
    opt.add_option('--lm', dest='lm', default='')
    (options, _) = opt.parse_args()
    if options.test_file.strip() == '' or options.train_file.strip() == '' or options.states_file.strip() == '':
        sys.stderr.write(
                "Usage: jython make_feats.py "
                "--train [train file] "
                "--test [test file] "
                "--states [states file]\n")
        sys.exit(1)
    # blm = BLM(options.lm)
    states_per_sentence = {}
    all_states = set([])
    for line in codecs.open(options.states_file, 'r', 'utf8').readlines():
        items = line.split()
        sent_id = int(items[0])
        states_per_sentence[sent_id] = list(set(items[1:]))
        all_states.update(list(set(items[1:])))

    obs_set = set([])
    tag2id = {}
    add_to_tags(BLANK)
    add_to_tags(CORRECT_BLANK)
    add_to_tags(UNK)

    training_ti = []
    testing_ti = []
    for line in codecs.open(options.train_file, 'r', 'utf8').readlines():
        ti, obs, guess = get_instance(line)
        sid = int(ti.current_sent[0].sent_id)
        for s in states_per_sentence[sid]:
            add_to_tags(s)
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
    print 'len total tags:', len(tag2id)
    print 'len obs:', len(obs2id)
    print 'populating features...'
    populate_features()
    print 'num features', len(feature_label2id)
    # blm = BLM(options.lm)
    uc_training = UnCachedFgList(training_instanes=training_ti, states_per_sent=states_per_sentence)
    trainer = CrfTrainer(get_trainer_prm())
    feature_ids, feature_labels = zip(*sorted([(v, k) for k, v in feature_label2id.iteritems()]))
    # initialize weight for each feature
    factor_graph_model = FgModel(len(feature_label2id), list(feature_labels))
    for fid in list(feature_ids):
        factor_graph_model.add(fid, 0.0)

    trainer.train(factor_graph_model, uc_training)
    sw = FileWriter('feature.weights')
    factor_graph_model.printModel(sw)
    sw = codecs.open('feature.names', 'w', 'utf8')
    for k, i in feature_label2id.iteritems():
        sw.write(str(i) + '\t' + str(k) + '\n')
    sw.flush()
    sw.close()
