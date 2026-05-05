import numpy as np
from utils import win_prob
from chicken import Chicken


class Environment:
    def __init__(self, n_coops=4, n_chickens=64, k=3):
        self.n_coops = n_coops
        self.n_chickens = n_chickens
        self.k = k

        self.agents = {}  # IMPORTANT: used for HW5 hooks
        self.chickens = [Chicken(i, n_coops) for i in range(n_chickens)]

        self.fought = set()

    # -----------------------------
    # PAIRING LOGIC (FIXED SAFE)
    # -----------------------------
    def valid_pair(self, c1, c2):
        if c1.id == c2.id:
            return False

        if (c1.id, c2.id, c1.coop) in self.fought:
            return False

        if c1.losses[c1.coop] >= self.k:
            return False

        if c2.losses[c2.coop] >= self.k:
            return False

        return True

    def assign_cages(self):
        cages = []
        coop_groups = {i: [] for i in range(self.n_coops)}

        # group by coop
        for c in self.chickens:
            if c.zone == "battle":
                coop_groups[c.coop].append(c)

        # pair safely
        for coop, group in coop_groups.items():
            np.random.shuffle(group)

            used = set()

            for i in range(len(group)):
                if group[i] in used:
                    continue

                for j in range(i + 1, len(group)):
                    if group[j] in used:
                        continue

                    c1, c2 = group[i], group[j]

                    if self.valid_pair(c1, c2):
                        cages.append((c1, c2, coop))
                        used.add(c1)
                        used.add(c2)
                        break

        return cages

    # -----------------------------
    # BATTLES
    # -----------------------------
    def resolve_battles(self, cages):
        results = []

        for c1, c2, coop in cages:
            x1 = c1.ability[coop]
            x2 = c2.ability[coop]

            p = win_prob(x1, x2)

            if np.random.rand() < p:
                winner, loser = c1, c2
            else:
                winner, loser = c2, c1

            loser.losses[coop] += 1

            self.fought.add((c1.id, c2.id, coop))
            self.fought.add((c2.id, c1.id, coop))

            results.append((winner, loser, coop))

        return results

    # -----------------------------
    # OBSERVATIONS
    # -----------------------------
    def observations(self, results):
        for winner, loser, coop in results:
            obs = (winner.id, loser.id, coop)

            for c in self.chickens:
                if c.coop == coop and c.zone in ["battle", "spectator"]:
                    c.add_observation(obs)

    # -----------------------------
    # MAIN STEP
    # -----------------------------
    def step(self, agents):
        # IMPORTANT: bind agents for HW5
        self.agents = agents

        # 1. assign cages
        cages = self.assign_cages()

        # 2. agent actions (spectators only)
        actions = {}
        for c in self.chickens:
            if c.zone == "spectator":
                actions[c.id] = agents[c.id].choose_action(c, self)

        # 3. resolve battles
        results = self.resolve_battles(cages)

        # 4. observations
        self.observations(results)

        # 5. belief updates
        for winner, loser, coop in results:
            winner.update_belief(loser.belief, "win", coop, win_prob)
            loser.update_belief(winner.belief, "loss", coop, win_prob)

        # 6. HW5 post-step hooks (sharing / sacks / transfers)
        for c in self.chickens:
            agent = self.agents.get(c.id, None)
            if agent and hasattr(agent, "post_step"):
                agent.post_step(c, self)

        return results

    # -----------------------------
    # TOURNAMENT RESET
    # -----------------------------
    def reset_tournament(self):
        coop_groups = {i: [] for i in range(self.n_coops)}

        for c in self.chickens:
            coop_groups[c.coop].append(c)

        # assign crowns
        for coop, group in coop_groups.items():
            undefeated = [c for c in group if c.losses[coop] == 0]

            if undefeated:
                for c in undefeated:
                    c.sack += 1
            else:
                wins = {}

                for c in group:
                    wins[c.id] = sum(
                        1 for w, l, r in self.fought
                        if w == c.id and r == coop
                    )

                max_w = max(wins.values())

                for c in group:
                    if wins[c.id] == max_w:
                        c.sack += 1

        # reset state cleanly
        self.fought.clear()

        for c in self.chickens:
            c.reset_for_tournament()
            c.zone = "battle"