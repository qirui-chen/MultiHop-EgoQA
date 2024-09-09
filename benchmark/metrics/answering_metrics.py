import numpy as np
import sentence_transformers
from pycocoevalcap.bleu.bleu import Bleu as Bleuold
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


class Bleu(Bleuold):

    def compute_score(self, gts, res):

        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)
            bleu_scorer += (hypo[0], ref)
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)

        return score, scores


class SentenceTransformerSimilarity:

    def __init__(self):
        self.model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to('cuda')

    def compute_score(self, gts, res):
        scores = []
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            pred_emb = self.model.encode(hypo)
            gts_emb = self.model.encode(ref)
            score = sentence_transformers.util.dot_score(pred_emb, gts_emb)[0, 0].cpu()
            # print(sentence_transformers.util.dot_score(pred_emb, gts_emb).shape)
            scores.append(float(score))

        score = np.mean(scores)

        return score, scores


if __name__ == '__main__':

    hypo = {
        "1": ["A child is playing with a ball"],
        "2": ["A group of people are dancing"],
        "3": ["A dog is barking at a cat"]
    }

    ref = {
        "1": ["A child is kicking a soccer ball.", "Kids are playing in the park.", "A boy is throwing a ball"],
        "2": [
            "People are enjoying a dance party.", "A group is dancing at a festival.",
            "Several individuals are performing a dance routine"
        ],
        "3": ["A dog is barking loudly.", "The dog is chasing a cat.", "A cat is being barked at by a dog"]
    }

    # bleu4_scorer = Bleu(4)
    # bleu4_score, bleu4_scores = bleu4_scorer.compute_score(ref, hypo)
    # print(bleu4_score[3], bleu4_scores[3])

    # cider_scorer = Cider()
    # cider_score, cider_scores = cider_scorer.compute_score(ref, hypo)
    # print(cider_score, cider_scores)

    # meteor_scorer = Meteor()
    # meteor_score, scores = meteor_scorer.compute_score(ref, hypo)

    # print(meteor_score, scores)

    sim_scorer = SentenceTransformerSimilarity()
    score, scores = sim_scorer.compute_score(ref, hypo)
    print(score, scores)
