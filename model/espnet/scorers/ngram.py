from abc import ABC
import kenlm
import torch
from espnet.scorer_interface import BatchScorerInterface, PartialScorerInterface

class NgramBase(ABC):
    """Base class for KenLM n-gram scorers."""

    def __init__(self, ngram_model, token_list):
        """Initialize NgramBase with KenLM model and token list."""
        # Map ESPnet tokens to KenLM tokens, convert <eos> to </s>
        self.chardict = [x if x != "<eos>" else "</s>" for x in token_list]
        self.charlen = len(self.chardict)
        # Load KenLM binary model
        self.lm = kenlm.LanguageModel(ngram_model)
        # Temporary state for KenLM BaseScore
        self.tmpkenlmstate = kenlm.State()

    def init_state(self, x):
        """Initialize KenLM state for new sequences."""
        state = kenlm.State()
        self.lm.NullContextWrite(state)
        return state

    def score_partial_(self, y, next_token, state, x):
        """Compute KenLM scores for given next_token IDs."""
        # Advance state with last token in prefix
        out_state = kenlm.State()
        prev_token = self.chardict[y[-1]] if y.shape[0] > 1 else "<s>"
        self.lm.BaseScore(state, prev_token, out_state)
        # Prepare output scores tensor
        scores = torch.empty_like(next_token, dtype=x.dtype, device=y.device)
        # Score each candidate token
        for i, j in enumerate(next_token):
            idx = int(j)
            scores[i] = self.lm.BaseScore(out_state, self.chardict[idx], self.tmpkenlmstate)
        return scores, out_state

class NgramFullScorer(NgramBase, BatchScorerInterface):
    """Full scorer for KenLM n-gram, scoring entire vocabulary."""
    def score(self, y, state, x):
        """Score all tokens in vocabulary."""
        ids = torch.arange(self.charlen, device=x.device)
        return self.score_partial_(y, ids, state, x)

class NgramPartScorer(NgramBase, PartialScorerInterface):
    """Partial scorer for KenLM n-gram, scoring subset of tokens."""
    def score_partial(self, y, next_token, state, x):
        """Score provided next_token subset."""
        return self.score_partial_(y, next_token, state, x)

    def select_state(self, state, i, new_id=None):
        """No-op state selection for KenLM scorer."""
        return state 