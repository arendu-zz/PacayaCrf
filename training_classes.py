__author__ = 'arenduchintala'
from editdistance import EditDistance
import enchant
import sys

class TrainingInstance(dict):
    def __init__(self,
            user_id,
            past_correct_guesses,
            past_sentences_seen,
            past_guesses_for_current_sent,
            current_sent,
            current_revealed_guesses,
            current_guesses):
        dict.__init__(self)
        self.__dict__ = self
        # these are the inputs
        self.user_id = user_id
        self.past_correct_guesses = past_correct_guesses
        self.past_sentences_seen = past_sentences_seen
        self.past_guesses_for_current_sent = past_guesses_for_current_sent
        self.current_sent = current_sent
        self.current_revealed_guesses = current_revealed_guesses
        # this is the output
        self.current_guesses = current_guesses

    @staticmethod
    def from_dict(_dict):
        ti = TrainingInstance(user_id=_dict['user_id'],
                past_correct_guesses=list(map(Guess.from_dict, _dict['past_correct_guesses'])),
                past_sentences_seen=_dict['past_sentences_seen'],
                past_guesses_for_current_sent=list(
                    map(Guess.from_dict, _dict['past_guesses_for_current_sent'])),
                current_sent=list(map(SimpleNode.from_dict, _dict['current_sent'])),
                current_revealed_guesses=list(map(Guess.from_dict, _dict['current_revealed_guesses'])),
                current_guesses=list(map(Guess.from_dict, _dict['current_guesses'])))
        return ti

def check_ignore_case(w, d):
    if d.check(w) or w in ['__UNK__', '__COPY__', '__BLANK__' , '*', '&quot;']:
        return True
    else:
        s_lower = [i.lower() for i in d.suggest(w)]
        if len(s_lower) == 0:
            return False
        else:
            if w in s_lower or w.lower() in s_lower:
                return True
            else:
                return False

def copy_or_not(w, l2,  english_d, ed):
    if w.lower() == l2.lower():
        return '__COPY__', l2 , 'de'
    else:
        pass
    en_single_word = [s.lower() for s in english_d.suggest(w) if len(s.split()) == 1 and len(s.split('-')) == 1]
    if w.isdigit():
        en_single_word = []
    sys.stderr.write('ed for ' + w + '\n')
    if len(en_single_word) == 0:
        sys.stderr.write('german ed was lower, guess:' + w + ',en correction:[]' + ', german context:'+ l2 + '\n')
        return '__COPY__', l2 , 'de'
    else:
        ed2_en = ed.editdistance_simple(w, en_single_word[0])[0]
        ed2_de = ed.editdistance_simple(w, l2)[0]
        if ed2_de >= ed2_en:
            sys.stderr.write('english ed was lower, guess:' + w + ', correction:' + en_single_word[0] + ', german context:' +l2+ '\n')
            return en_single_word[0] , en_single_word[0] , 'en'
        else:
            sys.stderr.write('german ed was lower, guess:' + w + ',en correction:' + en_single_word[0] + ', german context:'+ l2 + '\n')
            return '__COPY__', l2 , 'de'

class Guess(dict):
    def __init__(self, id, guess, revealed, l2_word):
        dict.__init__(self)
        self.__dict__ = self
        Guess.spell = enchant.DictWithPWL("en_US", "../data/vocab/en.vocab")
        Guess.spell_german = enchant.request_pwl_dict('../data/de/nachrichtenleicht-filtered/nachrichtenleicht.de.vocab')
        Guess.ed = EditDistance(None)
        tmp_l2_lower = l2_word.lower()
        if guess.strip() == '':
            self.guess = '__BLANK__'
        elif guess.strip() == '__BLANK__' or guess.strip() == '__UNK__' or guess.strip() == '__COPY__':
            self.guess = guess.replace("'","")

        else:
            guess = sorted([(len(g),g) for g in guess.split()])[-1][1]
            guess = guess[:-1] if guess[-1] == '*' and len(guess) > 1 else guess
            if check_ignore_case(guess, Guess.spell):
                self.guess = guess.replace("'","")
            else:
                result = copy_or_not(guess,tmp_l2_lower, Guess.spell, Guess.ed)
                self.guess = result[0].replace("'","")
            '''
            elif tmp_l2_lower in [s.lower() for s in Guess.spell_german.suggest(guess)] or tmp_l2_lower == guess.lower():
                sys.stderr.write('not in en dict:' + guess + '\n')
                sys.stderr.write('changing:' + guess + ' to __COPY__ , very similar to l2_word:'  + tmp_l2_lower + '\n')
                self.guess = '__COPY__'
            else:
                sys.stderr.write('not in en dict:' + guess + ' and not copy of:' + tmp_l2_lower +'\n')
                suggest = Guess.spell.suggest(guess)
                single_word = [s.lower() for s in suggest if len(s.split()) == 1 and len(s.split('-')) == 1]
                if len(single_word) > 0:
                    if guess.isdigit() and not single_word[0].isdigit():
                        sys.stderr.write('no good suggestion for:' + guess + '\n')
                        self.guess = guess
                    else:
                        sys.stderr.write('spell correcting:' + guess + ' to:' + single_word[0] +'\n')
                        self.guess = single_word[0]
                else:
                    sys.stderr.write('setting UNK:' + guess + '\n')
                    self.guess = '__UNK__'
                '''
        self.l2_word = l2_word
        self.id = id
        self.revealed = revealed

    def copy(self, new_id=None):
        if new_id:
            g = Guess(id=new_id, guess=self.guess, revealed=self.revealed, l2_word=self.l2_word)
        else:
            g = Guess(id=self.id, guess=self.guess, revealed=self.revealed, l2_word=self.l2_word)
        return g

    def __str__(self):
        return ','.join([str(self.id), str(self.guess), str(self.revealed)])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __cmp__(self, other):
        if str(self) == str(other):
            return 0
        elif str(self) > str(other):
            return 1
        else:
            return -1
    @staticmethod
    def from_dict(_dict):
        g = Guess(id=tuple(_dict['id']),
                guess=_dict['guess'],
                revealed=_dict['revealed'],
                l2_word=_dict['l2_word'])
        return g


class SimpleNode(dict):
    def __init__(self, sent_id, id, l2_word, position, lang, l1_parent):
        dict.__init__(self)
        self.__dict__ = self
        self.sent_id = sent_id
        self.id = id
        self.l2_word = l2_word
        self.l1_parent = l1_parent
        self.position = int(position)
        self.lang = lang

    def __cmp__(self, other):
        if self.position == other.position:
            return 0
        elif self.position > other.position:
            return 1
        else:
            return -1

    @staticmethod
    def from_dict(_dict):
        s = SimpleNode(sent_id=_dict['sent_id'],
                id=tuple(_dict['id']),
                l2_word=_dict['l2_word'],
                l1_parent=_dict['l1_parent'],
                position=_dict['position'],
                lang=_dict['lang'])
        return s
