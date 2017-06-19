#!/usr/bin/python3
import numpy as np
from collections import defaultdict, deque

class GramLearner(object):
    def __init__(self,path_to_wordlist = None , step_max = 4):
        """ constructor for GramLearner object
        """
        self.cnt_ = 1
        self.step_max_ = step_max
        self.word_to_idx_ = {}
        self.idx_to_word_ = {}
        self.proba_vect_ = []
        self.count_vect_ = [] 
        self.word_to_idx_[""] = 0
        self.idx_to_word_[0] = ""
        for i in range( 1, self.step_max_ + 2 ):
            self.proba_vect_.append({})
            self.count_vect_.append(defaultdict(lambda:0)) # to remain concise
        if path_to_wordlist is not None:
            self.initWordList(path_to_wordlist)
    
    def initWordList(self, wordlist):
        """ @wordlist: path to file containing a 
                       vocabulary list of the langage.
                       each line of the file contains one unique word.
            this function initialize @idx_to_word_ and @word_to_idx
        """
        with open(wordlist,"r") as f:
            for line in f:
                if not line in self.word_to_idx_:
                    filtered_word = line.replace("\n","").replace("\r","")
                    self.word_to_idx_[filtered_word] = self.cnt_
                    self.idx_to_word_[self.cnt_] = filtered_word
                    self.cnt_ += 1
        print( self.cnt_ )

    @staticmethod
    def wordGen(textpath):
        """ generator of words reading directly from @textpath 
        """
        with open(textpath,"r") as f:
            for line in f:
                for word in line.replace("\n","").replace(",","").replace("."," .").replace(". . .","...").split(" "):
                    yield word

    def getIdx(self,word):
        """ str word -> int idx
        """
        if word in self.word_to_idx_:
            return self.word_to_idx_[word]
        else:
            self.word_to_idx_[word] = self.cnt_
            self.idx_to_word_[self.cnt_] = word
            self.cnt_ += 1
            return self.word_to_idx_[word]

    def getWord(self,idx):
        """ int idx -> str word
        """
        if idx in self.idx_to_word_ :
            return self.idx_to_word_[idx]
        else:
            return None

    def __getitem__(self, key):
        """ Element accessor. return a count.
            @key : either a string or a container of string
        """
        if type(key) is str :
            return self.count_vect_[0].get(tuple([self.word_to_idx_[key]]),0) 
            # get prevent the key to be recorded in count_vect_
        elif len(key) > self.step_max_ + 1 :
            raise Exception('item must be of len <= sep_max')
        else :
            length = len(key)
            idx_tuple = tuple([self.getIdx(word) for word in list(key)])
            return self.count_vect_[length-1].get(idx_tuple,0)

                
    def updateCount(self, path):
        """ update @count_vect_ reading a text file pointed by @path
             @path : path to text file
        """
        self.line_count_ = 0
        self.wc_ = 0
        pattern = []
        for i in range(self.step_max_+ 1 ):
            pattern.append(deque( np.zeros( i+1 , dtype=np.uint32) , maxlen = i+1))
        reader = self.wordGen(path)
        for word in reader:
            idx = self.getIdx(word)
            self.wc_ += 1
            for i in range(self.step_max_ + 1):
                pattern[i].append(idx)
                current_pattern = tuple(pattern[i])
                if self.wc_ > i :
                    self.count_vect_[i][current_pattern] += 1

    def computeProba(self):
        """ Compute transition probabilities from count_vect_
             @proba_vect_[i]: key: i+1 tuple (isized) -> 1element   
        """
        sum_counts = sum(self.count_vect_[1].values())
        self.proba_vect_[1] = { k: v/sum_counts for k,v in self.count_vect_[1].items() }
        for i in range( 2 , self.step_max_ + 1 ):
            self.proba_vect_[i] = { key_full: val/self.count_vect_[i-1][key_full[:-1]] for key_full, val in self.count_vect_[i].items() }
    
    def predictNextIdx(self,t_idx):
        """ predict next idx, given a tuple of idx
        """
        assert isinstance(t_idx, tuple) or isinstance (t_idx,list) , "must be a tuple or a list"
        assert len(t_idx)<= self.step_max_, "you cannot have such long priori (len <= %i)" % self.step_max_
        t_idx = tuple(t_idx)
        step = len(t_idx)
        proba_dict = {k[-1]:v for k,v in self.proba_vect_[step].items() if k[:-1]== t_idx}
        print(proba_dict)
        max_idx = max(list(proba_dict), key=(lambda key:proba_dict[key]))
        return max_idx

    def predictNextIdxRandom(self,t_idx):
        """ Compute next idx given the previous ones
        """
        assert isinstance(t_idx, tuple) or isinstance (t_idx,list) , "must be a tuple or a list"
        assert len(t_idx)<= self.step_max_, "you cannot have such long priori (len <= %i)" % self.step_max_
        t_idx = tuple(t_idx)
        step = len(t_idx)
        exp_proba_dict = { k[-1]: np.exp(v) for k,v in self.proba_vect_[step].items() if k[:-1]== t_idx }
        # print( exp_proba_dict ) ## DEBUG
        rand_idx = list(exp_proba_dict)[np.random.choice( len(exp_proba_dict), p = np.array(list(exp_proba_dict.values()))/ sum(exp_proba_dict.values()))]
        return rand_idx

    def generateNIdx(self,t_idx,N = 30):
        """ generate N next idx by stepwise greedy optimisation
            @t_idx: a tuple of idx
            @N : number of idx to generate
        """
        seed = t_idx
        nexts = []
        window_idx = deque( seed, maxlen = self.step_max_ )
        for i in range(N):
            new = self.predictNextIdx(tuple(window_idx))
            window_idx.append(new)
            nexts.append(new)
        return nexts

    def generateNIdxRandom(self,t_idx,N = 30):
        """ generate N next idx by stepwise softmax selection
            @t_idx: a tuple of idx
            @N : number of idx to generate
        """
        seed = t_idx
        nexts = []
        window_idx = deque( seed, maxlen = self.step_max_ )
        for i in range(N):
            new = self.predictNextIdxRandom(tuple(window_idx))
            window_idx.append(new)
            nexts.append(new)
        return nexts

    def generateNWords(self, seedword, N = 30):
        """ generate N words after seedwords according to the model
            choice of words are made by greedy maximization
             @seedword : a list of words
             @N : number of words to generate
             return : list of words generated
        """
        seed = [self.word_to_idx_[word] for word in seedword]
        nexts = self.generateNIdx(seed,N)
        return [self.idx_to_word_[idx] for idx in nexts]

    def generateNWordsRandom(self, seedword, N = 30):
        """ generate n word after seedwords according to the model
            choice of words are made by softmax selection.
             @seedword : a list of words
             @N : number of words to generate
             return : list of words generated
        """
        seed = [self.word_to_idx_[word] for word in seedword]
        nexts = self.generateNIdxRandom(seed,N)
        return [self.idx_to_word_[idx] for idx in nexts]


if __name__ == "__main__":
    # test GramLearner methods.
    gram = GramLearner("wordlist.txt")
    reader = gram.wordGen("test3.txt")
    gram.updateCount("test3.txt")
    gram.computeProba()
    print(gram.proba_vect_[4])
    print("now we generate")
    u = list(gram.count_vect_[3])[10]
    print(u)
    print(gram.predictNextIdx(t_idx = u[:-1]))
    print(gram.predictNextIdxRandom(u[:-1]))
    print(gram.predictNextIdxRandom(u[:-1]))
    print(gram.predictNextIdxRandom(u[:-1]))
    list_words = [next(reader) for i in range(4)]
    print(list_words)
    print(gram.generateNWordsRandom(list_words))


