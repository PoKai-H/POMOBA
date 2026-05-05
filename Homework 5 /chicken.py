import numpy as np

class Chicken:
    def __init__(self, cid, n_coops):
        self.id = cid
        self.n_coops = n_coops

        # Ability vector (hidden truth)
        self.ability = self.sample_ability()

        # Belief over abilities (uniform initially)
        self.belief = np.ones((n_coops, n_coops)) / n_coops

        # Location
        self.coop = np.random.randint(n_coops)
        self.zone = "battle"  # battle, spectator, transit

        # History
        self.history = []

        # Loss tracking per coop
        self.losses = np.zeros(n_coops)

        # HW5 additions
        self.sack = 0
        self.owners = set([self.id])

    def sample_ability(self):
        N = self.n_coops
        probs = np.exp(-np.arange(1, N+1)/(N/3))
        probs /= probs.sum()
        return np.random.choice(np.arange(1, N+1), size=N, p=probs)

    def update_belief(self, opponent_belief, result, coop, win_prob_func):
        """
        Proper Bayesian update using FULL likelihood
        """
        N = self.n_coops
        new_belief = np.zeros(N)

        for xi in range(1, N+1):
            likelihood = 0
            for xj in range(1, N+1):
                p = win_prob_func(xi, xj)
                if result == "win":
                    likelihood += p * opponent_belief[coop][xj-1]
                else:
                    likelihood += (1 - p) * opponent_belief[coop][xj-1]

            new_belief[xi-1] = likelihood * self.belief[coop][xi-1]

        # normalize
        new_belief /= new_belief.sum()
        self.belief[coop] = new_belief

    def add_observation(self, obs):
        self.history.append(obs)

    def reset_for_tournament(self):
        self.losses[:] = 0