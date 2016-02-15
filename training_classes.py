__author__ = 'arenduchintala'


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


class Guess(dict):
    def __init__(self, id, guess, revealed, l2_word):
        dict.__init__(self)
        self.__dict__ = self
        self.guess = guess
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
    def from_dict(_dict, make_lowercase=True):
        g = Guess(id=tuple(_dict['id']),
                  guess=_dict['guess'].strip() if not make_lowercase else _dict['guess'].lower().strip(),
                  revealed=_dict['revealed'],
                  l2_word=_dict['l2_word'].strip() if not make_lowercase else _dict['l2_word'].lower().strip())
        return g


class SimpleNode(dict):
    def __init__(self, sent_id, id, l2_word, position, lang):
        dict.__init__(self)
        self.__dict__ = self
        self.sent_id = sent_id
        self.id = id
        self.l2_word = l2_word
        self.position = int(position)
        self.lang = lang

    def __cmp__(self, other):
        if self.position == other.position:
            return 0
        elif self.position > other.position:
            return 1
        else:
            return -1

    def __str__(self):
        return ','.join([str(self.position), str(self.id), self.l2_word, self.lang, str(self.sent_id)])

    @staticmethod
    def from_dict(_dict, make_lowercase=True):
        s = SimpleNode(sent_id=_dict['sent_id'],
                       id=tuple(_dict['id']),
                       l2_word=_dict['l2_word'].strip() if not make_lowercase else _dict['l2_word'].lower().strip(),
                       position=_dict['position'],
                       lang=_dict['lang'])
        return s
