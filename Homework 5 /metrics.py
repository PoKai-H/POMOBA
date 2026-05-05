import numpy as np
import random
class Metrics:
    def __init__(self):
        self.ability_error = []
        self.win_prob_error = []
        self.king_error = []

    def track_ability(self, chickens):
        errors = []
        for c in chickens:
            for coop in range(c.n_coops):
                belief_mean = np.dot(
                    c.belief[coop],
                    np.arange(1, c.n_coops+1)
                )
                true_val = c.ability[coop]
                errors.append(abs(belief_mean - true_val))
        self.ability_error.append(np.mean(errors))

    def track_win_prob(self, chickens, win_prob_func):
        errors = []

        # sample ONLY 50 random pairs instead of all
        all_pairs = [(i, j) for i in range(len(chickens)) for j in range(i+1, len(chickens))]
        sampled_pairs = random.sample(all_pairs, min(20, len(all_pairs)))

        for i, j in sampled_pairs:
            c1, c2 = chickens[i], chickens[j]

            for coop in range(c1.n_coops):
                pred = 0
                for xi in range(1, c1.n_coops+1):
                    for xj in range(1, c1.n_coops+1):
                        pred += (
                            win_prob_func(xi, xj)
                            * c1.belief[coop][xi-1]
                            * c2.belief[coop][xj-1]
                        )

                true = 1 if c1.ability[coop] > c2.ability[coop] else 0.5
                errors.append(abs(pred - true))

        self.win_prob_error.append(np.mean(errors))

    def track_king(self, chickens):
        # simple proxy: crown vs expected
        crowns = [c.sack for c in chickens]
        avg = np.mean(crowns)
        errors = [abs(c - avg) for c in crowns]
        self.king_error.append(np.mean(errors))