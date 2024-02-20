import utils
import flappy_bird_gym
import random
import time

CONSTANT = 0.999973
MIN_REWARD = -100
MAX_REWARD = 100
DISCOUNT = 0.000006
class SmartFlappyBird:
    def __init__(self, iterations):
        self.Qvalues = utils.Counter()
        self.landa = 1
        self.epsilon = 0.3  # change to proper value
        self.alpha = 0.3  # change to proper value
        self.iterations = iterations
        self.move = 0
        self.policy_move = 0
        self.random_move = 0

    def policy(self, state):
        return self.max_arg(state) # for each state, action related the max value will be the policy for that state

    @staticmethod
    def get_all_actions():
        return [0, 1] # up or down

    @staticmethod
    def convert_continuous_to_discrete(state):
        r1 = round(state[0] % 0.5, 1)
        r2 = round(state[1], 1)
        return [r1, r2]

    def compute_reward(self, prev_info, new_info, done, observation, this_state):
        if done:
            return MIN_REWARD
        return MAX_REWARD if new_info['score'] > prev_info['score'] else 0

    def get_action(self, state):
        if utils.flip_coin(self.epsilon):
            self.random_move += 1
            return random.choice(self.get_all_actions())
        else:
            self.policy_move += 1
            return self.policy(state)

    def maxQ(self, state):
        discrete_state = SmartFlappyBird.convert_continuous_to_discrete(state)
        q_0 = (discrete_state[0], discrete_state[1], 0)
        q_1 = (discrete_state[0], discrete_state[1], 1)
        return max(q_1, q_0, key=lambda x: self.Qvalues[x])

    def max_arg(self, state):
        return self.maxQ(state)[2]

    def update(self, reward, state, action, next_state):
        discrete_state = SmartFlappyBird.convert_continuous_to_discrete(state)
        q = (discrete_state[0], discrete_state[1], action)
        max_next_state_q = self.maxQ(next_state)
        self.Qvalues[q] += self.alpha * (reward + self.landa * self.Qvalues[max_next_state_q] - self.Qvalues[q])

    def update_epsilon_alpha(self):
        self.epsilon = CONSTANT ** self.move
        alph = self.alpha - DISCOUNT
        next_alpha = max(0.01, alph)
        self.alpha = next_alpha

    def run_with_policy(self, landa):
        self.landa = landa
        env = flappy_bird_gym.make("FlappyBird-v0")
        observation = env.reset()
        info = {'score': 0}
        for _ in range(self.iterations):
            while True:
                action = self.get_action(observation)
                this_state = observation
                prev_info = info
                observation, reward, done, info = env.step(action)
                self.move += 1
                reward = self.compute_reward(prev_info, info, done, observation, this_state)
                self.update(reward, this_state, action, observation)
                self.update_epsilon_alpha()
                if done:
                    observation = env.reset()
                    break
        print(f"policy_moves = {self.policy_move}")
        print(f"random_moves = {self.random_move}")
        env.close()

    def run_with_no_policy(self, landa):
        total_score = 0
        for _ in range(10):
            self.landa = landa
            self.epsilon = -1
            env = flappy_bird_gym.make("FlappyBird-v0")
            observation = env.reset()
            info = {'score': 0}
            while True:
                action = self.get_action(observation)
                this_state = observation
                prev_info = info
                observation, reward, done, info = env.step(action)
                reward = self.compute_reward(prev_info, info, done, observation, this_state)
                env.render()
                time.sleep(1 / 120)  # FPS
                if done:
                    total_score += info['score']
                    break
            env.close()
        print(f"average = {total_score / 10}")

    def run(self):
        self.run_with_policy(1)
        self.run_with_no_policy(1)

program = SmartFlappyBird(iterations=2000)
program.run()
