from environment import Environment
from agent import NPCAgent, SimpleMDPAgent, MCTSAgent,CooperativeAgent,CooperativeMCTSAgent, CooperativeMDPAgent
from metrics import Metrics
from utils import win_prob
import matplotlib.pyplot as plt

metrics = Metrics()

env = Environment()

agents = {}

#  NPCs
for i in range(56):
    agents[i] = NPCAgent()
#cooperative agent 
for i in range(56, 60):
    agents[i] = CooperativeAgent()
# 4 MDP agents
for i in range(60, 64):
    agents[i] = CooperativeMDPAgent()

# IMPORTANT: connect agents to env
env.agents = agents

T = 256

for t in range(T):
    for _ in range(6):
        results = env.step(agents)

        if t % 10 == 0:   # only every 5 tournaments
            metrics.track_ability(env.chickens)
            metrics.track_win_prob(env.chickens, win_prob)

    print(f"Tournament {t}: total crowns = {sum(c.sack for c in env.chickens)}")

    env.reset_tournament()
    metrics.track_king(env.chickens)

# plots
plt.figure()
plt.plot(metrics.ability_error)
plt.title("Ability Belief Convergence")

plt.figure()
plt.plot(metrics.win_prob_error)
plt.title("Win Probability Convergence")

plt.figure()
plt.plot(metrics.king_error)
plt.title("King Prediction Convergence")

plt.show()

# final scores
scores = {}
for c in env.chickens:
    share = c.sack / len(c.owners)
    for o in c.owners:
        scores[o] = scores.get(o, 0) + share

print(sorted(scores.items(), key=lambda x: -x[1]))