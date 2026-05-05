import numpy as np
import copy

class BaseAgent:
    def choose_action(self, chicken, env):
        raise NotImplementedError


class NPCAgent:
    def choose_action(self, chicken, env):
        # move to coop where expected ability is highest
        expected = []

        for coop in range(chicken.n_coops):
            exp = np.dot(
                chicken.belief[coop],
                np.arange(1, chicken.n_coops + 1)
            )
            expected.append(exp)

        best_coop = int(np.argmax(expected))

        return {
            "move": best_coop,
            "watch": best_coop
        }

class RandomAgent(BaseAgent):
    def choose_action(self, chicken, env):
        return {
            "move": np.random.randint(env.n_coops),
            "watch": np.random.randint(env.n_coops)
        }

class SimpleMDPAgent:
    def choose_action(self, chicken, env):
        #exploration 
        if np.random.rand() < 0.1:
            return {
                "move": np.random.randint(chicken.n_coops),
                "watch": np.random.randint(chicken.n_coops)
            }

        # normal MDP logic below
        best_action = None
        best_value = -1

        for coop in range(chicken.n_coops):
            expected_win = 0

            for xi in range(1, chicken.n_coops + 1):
                for xj in range(1, chicken.n_coops + 1):
                    p_x = chicken.belief[coop][xi - 1]
                    p_y = chicken.belief[coop][xj - 1]

                    expected_win += (
                        p_x * p_y * env.win_prob(xi, xj)
                    )

            if expected_win > best_value:
                best_value = expected_win
                best_action = coop

        return {
            "move": best_action,
            "watch": best_action
        }
class MCTSAgent:
    def __init__(self, simulations=10):
        self.simulations = simulations

    def rollout(self, chicken, coop, env):
        score = 0

        for _ in range(self.simulations):
            p = chicken.belief[coop] ** 2
            p = p / np.sum(p)   # IMPORTANT: renormalize

            xi = np.random.choice(
                np.arange(1, chicken.n_coops + 1),
                p=p
            )
            xj = np.random.choice(
                np.arange(1, chicken.n_coops + 1)
            )

            score += env.win_prob(xi, xj)

        return score / self.simulations

    def choose_action(self, chicken, env):
        best_coop = 0
        best_score = -1

        for coop in range(chicken.n_coops):
            score = self.rollout(chicken, coop, env)

            if score > best_score:
                best_score = score
                best_coop = coop

        return {
            "move": best_coop,
            "watch": best_coop
        }
class CooperativeMDPAgent(SimpleMDPAgent):
    def post_step(self, chicken, env):
        # Identify teammates of the same class type[cite: 1, 2]
        teammates = [c for c in env.chickens if isinstance(env.agents[c.id], CooperativeMDPAgent) and c.id != chicken.id]
        
        if not teammates:
            return

        # 1. ShareObservations
        for teammate in teammates:
            for obs in chicken.history:
                if obs not in teammate.history:
                    teammate.history.append(obs)

        # 2. ExtendSackOwnership
        for teammate in teammates:
            chicken.owners.add(teammate.id)

        # 3. TransferCrowns[cite: 1]
        strongest = max(teammates, key=lambda c: c.sack)
        if chicken.sack > 0:
            chicken.sack -= 1
            strongest.sack += 1

class CooperativeMCTSAgent(MCTSAgent):
    def post_step(self, chicken, env):
        # Same logic as above, but checking for CooperativeMCTSAgent 
        teammates = [c for c in env.chickens if isinstance(env.agents[c.id], CooperativeMCTSAgent) and c.id != chicken.id]
        
        if not teammates:
            return

        for teammate in teammates:
            # Share Observations[cite: 1]
            new_obs = [o for o in chicken.history if o not in teammate.history]
            teammate.history.extend(new_obs)
            # Extend Ownership[cite: 1]
            chicken.owners.add(teammate.id)
class CooperativeAgent:
    def choose_action(self, chicken, env):
        expected = chicken.belief @ np.arange(1, chicken.n_coops+1)
        best = np.argmax(expected)

        return {
            "move": best,
            "watch": best
        }

    def post_step(self, chicken, env):
        """
        Implements HW5 actions: ShareObservations, TransferCrowns, ExtendSackOwnership[cite: 1, 2]
        Called in environment.py after battles and initial observations
        """
        # Identify teammates (other agents of the same class)
        teammates = [c for c in env.chickens if isinstance(env.agents[c.id], CooperativeAgent) and c.id != chicken.id]
        
        if not teammates:
            return

        # 1. ShareObservations(Action=IdOfChickenToShareWith)
        # We use a set-based update to prevent the history list from growing exponentially
        current_history_ids = {id(obs) for obs in chicken.history}
        for teammate in teammates:
            for obs in chicken.history:
                # Only add if it's not already in their history to save memory/time
                if obs not in teammate.history:
                    teammate.history.append(obs)

        # 2. ExtendSackOwnership(Action=IdOfChickenToTransfer)
        # Adding teammates as owners divides the total crowns at end of game
        # Original owner (chicken.id) must be the one to extend ownership
        for teammate in teammates:
            chicken.owners.add(teammate.id)

        # 3. TransferCrowns(Action=(IdOfChickenToTransfer, NumberofCrowns))
        # Strategy: Move crowns to the teammate with the highest current sack 
        # to consolidate "wealth" for the set[cite: 1, 2]
        strongest_teammate = max(teammates, key=lambda c: c.sack)
        
        if chicken.sack > 0 and strongest_teammate.id != chicken.id:
            # Transfer logic: move 1 crown per step if available[cite: 1, 2]
            transfer_amount = 1 
            chicken.sack -= transfer_amount
            strongest_teammate.sack += transfer_amount