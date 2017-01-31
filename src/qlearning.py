#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

## Joseph McGill
## Fall 2016
## Grid world using Q-Learning
##
## This is the grid world implementation where the agent uses Q-Learning
## with an epsilon greedy strategy to navigate the world and reach the goal.
## The agent trains for 10,000 episodes, though it typically converges to
## the optimal solution around 2,000 episodes.


# Grid world implementation using Q-Learning
class GridWorld:

    # Constructor
    def __init__(self, size, success_prob, alpha):

        # size of the grid world
        self.size = size

        # success rate of a move (1 - epsilon)
        self.success_prob = success_prob

        # start and goal points of the agent
        self.agent = [self.size - 1, 0]
        self.start_pos = [self.size - 1, 0]
        self.goal_pos = [0, self.size - 1]
        self.move_counts = []
        self.move_count = 0

        # moves the agent can take
        self.moves = ['up', 'right', 'down', 'left']

        # grid the agent will move on
        self.grid = np.zeros((size, size), dtype=np.int64)

        # dictionary for the expected value of actions (value)
        # at a given state (key)
        self.expected_values = {}
        self.gamma = 0.95
        self.alpha = alpha

        # insert labels in to the grid (1 to size*size)
        count = 1
        for i in range(size):
            for j in range(size):
                self.grid[i, j] = count
                self.expected_values[count] = [0, 0, 0, 0]
                count += 1


    # Function to move the agent
    def move_agent(self, direction):

        old = list(self.agent)

        # Perform the specified action with a probability of 1 - epsilon
        if np.random.binomial(1, self.success_prob) != 1:
            direction = np.random.choice(self.moves)

        # make the movement
        direction = direction.lower()
        if direction == 'up':
            if self.agent[0] > 0:
                self.agent[0] -= 1
                self.move_count += 1

        elif direction == 'right':
            if self.agent[1] < self.size - 1:
                self.agent[1] += 1
                self.move_count += 1

        elif direction == 'down':
            if self.agent[0] < self.size - 1:
                self.agent[0] += 1
                self.move_count += 1

        elif direction == 'left':
            if self.agent[1] > 0:
                self.agent[1] -= 1
                self.move_count += 1

        else:
            print("Not a valid movement")

        # update the expected value of the previous state
        self.update(old, direction, self.agent)


    # Function to update the expected value of the previous state (Q learning)
    def update(self, curr_state, action, next_state):

        # expected values of the current and next states given the action
        pos = self.expected_values[self.grid[curr_state[0], curr_state[1]]]
        next_pos = self.expected_values[self.grid[next_state[0], next_state[1]]]
        move = self.moves.index(action)

        # reward for completing the action
        reward = 0
        if next_state[0] == self.goal_pos[0] and next_state[1] == self.goal_pos[1]:
            reward = 1

        # find the max q of the next state
        max_q = max(next_pos)

        # update the expected value
        pos[move] += self.alpha * (reward + self.gamma * max_q - pos[move])

    # Function to find the best action
    def best_action(self):

        # get the q values for the current state
        expected_vals = self.expected_values[self.grid[self.agent[0], self.agent[1]]]

        # find the best action to take based on the maximum Q value
        max_q = max(expected_vals)
        index = expected_vals.index(max_q)

        # if the max is 0 (no expected values), take a random action
        if max_q == 0:
            index = np.random.choice(len(expected_vals))

        return self.moves[index]

    # Function to train the agent
    def train(self):

        min_move_count = np.inf
        min_move_point = np.inf

        # train for 10000 episodes
        for i in range(10000):

            # train until the goal state is reached
            while (not (self.agent[0] == self.goal_pos[0] and
                   self.agent[1] == self.goal_pos[1])):

                self.move_agent(self.best_action())

            # update the min move count
            if self.move_count < min_move_count:
                min_move_count = self.move_count
                min_move_point = i

            self.agent = list(self.start_pos)
            self.move_counts.append(self.move_count)
            self.move_count = 0


        # return the optimal solution found
        return min_move_point, self.move_counts


## Grid world using Q-Learning
if __name__=="__main__":

    # create a grid world with an agent using Q-Learning
    grid = GridWorld(100, 0.9, alpha = 0.1)
    min_point, moves = grid.train()

    # plot the number of moves taken per episode
    plt.plot(moves, 'b', lw=1, label="Q-Learning")
    plt.xlabel("Episodes")
    plt.ylabel("Number of Moves")
    plt.ylim([0, 150000])
    plt.title("Number of Moves taken per episode")
    plt.legend(loc="upper right")
    plt.show()
