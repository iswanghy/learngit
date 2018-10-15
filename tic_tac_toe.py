#!/usr/bin/python
# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import time
import os
from IPython import display


class TicTacToe(object):
    __board = np.zeros(9)
    __next = 1
    __move_cnt = 0

    def __init__(self):
        self.__board = np.zeros(9)
        self.__next, self.__move_cnt = 1, 0

    def reset(self):
        self.__board = np.zeros(9)
        self.__next, self.__move_cnt = 1, 0

    def show_board(self):
        print self.__board[0:3]
        print self.__board[3:6]
        print self.__board[6:9]

    def move(self, pos):
        if self.__move_cnt == 9:
            print 'Game is Over. Cannot move anymore.'
            return -1
        if self.__board[pos] != 0:
            print 'Position is already occupied. Try again.'
            return -2
        self.__board[pos] = self.__next
        self.__move_cnt += 1
        self.__next *= -1
        return 1

    def get_available_pos(self):
        ret = []
        for i in range(9):
            if self.__board[i] == 0:
                ret.append(i)
        return ret

    def get_next(self):
        return self.__next

    def win_check(self):
        goal_list = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                     [0, 3, 6], [1, 4, 7], [2, 5, 8],
                     [0, 4, 8], [2, 4, 6]]
        for goal in goal_list:
            if sum(self.__board[goal]) == 3:
                print 'Player A won the game!'
                # self.show_board()
                return 1
            if sum(self.__board[goal]) == -3:
                print 'Player B won the game!'
                # self.show_board()
                return -1
        if self.__move_cnt == 9:
            print 'Draw!'
            # self.show_board()
            return 2
        return 0

    @staticmethod
    def board2int(board):
        temp = [i+1 for i in board]
        ret = 0
        for i in range(9):
            ret += temp[i] * (3 ** i)
        return int(ret)

    def get_state(self):
        return self.__board.copy(), self.__next, self.__move_cnt, TicTacToe.board2int(self.__board)


class QLearning(object):
    __game = TicTacToe()
    __actions = np.array(range(9))
    __q_table = pd.DataFrame(np.zeros((3 ** 9, 9)), columns=np.array(range(9)))  # 收益表 Q-table
    __gamma = 0.9  # 折损因子
    __lr = 0.8  # 学习率

    def __init__(self, lr, gamma):
        self.__game = TicTacToe()
        self.__actions = np.array(range(9))
        self.__q_table = pd.DataFrame(np.zeros((3 ** 9, 9)), columns=np.array(range(9)))
        self.__gamma = gamma
        self.__lr = lr

    def act_choose(self, epsilon):
        state = self.__game.get_state()
        state_id = TicTacToe.board2int(state[0])
        state_act = self.__q_table.iloc[state_id, :]
        available_act = self.__game.get_available_pos()
        if np.random.uniform() > epsilon or state_act.all() == 0:
            action = np.random.choice(available_act)
        else:
            action = state_act.idxmax()
        return action

    def env_feedback(self, action):
        self.__game.move(action)
        res = self.__game.win_check()
        end = 0
        if res == 0:
            reward = -2
        elif res == 2:
            reward, end = 0, 1
        elif res + self.__game.get_next() == 0:
            reward, end = 50, 1
        else:
            reward, end = -50, 1
        return reward, end, res

    def __update_q_table(self, old_id, new_id, action, end, reward):
        q_original = self.__q_table.loc[old_id, action]
        if not end:
            tmp = reward + self.__gamma * self.__q_table.iloc[new_id].max()
        else:
            tmp = reward
        self.__q_table.loc[old_id, action] = (1 - self.__lr) * q_original + self.__lr * tmp
        return


    def __update_q_table_list(self, id_list, act_list, reward):
        state = self.__game.get_state()
        reward -= state[2] * 2
        id_list = id_list[::-1]
        act_list = act_list[::-1]
        for i in range(1, len(id_list)):
            end = (i == 1)
            self.__update_q_table(id_list[i], id_list[i-1], act_list[i], end, reward * ((-0.8)**(i-1)))


    def __show(self):
        os.system('cls')
        self.__game.show_board()
        time.sleep(1)


    def run(self, max_episodes, epsilon):
        df = pd.read_csv('D:/WORK_PLAYGROUND/RL_test/aha.csv')
        self.__q_table = df.copy()
        self.__q_table.columns = range(9)
        episodes = 0
        # while episodes <= max_episodes:
        while 1:
            end = 0
            self.__game.reset()
            id_list, act_list = [], []
            if episodes % 1000 == 0:
                pass
                self.__show()
            while not end:
                act = self.act_choose(epsilon)
                state = self.__game.get_state()
                old_id = state[3]
                id_list.append(old_id)
                act_list.append(act)
                reward, end, res = self.env_feedback(act)
                # self.__update_q_table(old_id, self.__game.get_state()[3], act, end, reward)
                if episodes % 1000 == 0:
                    pass
                    self.__show()
            if end:
                self.__update_q_table_list(id_list, act_list, reward)
                episodes += 1
            if episodes % 10000 == 0:
                self.__q_table.to_csv('D:/WORK_PLAYGROUND/RL_test/aha.csv', index=False)
        return self.__q_table.copy()


if __name__ == '__main__':
    ql = QLearning(0.3, 0.7)
    x = ql.run(max_episodes=10000, epsilon=0.9)
    # x.to_csv('D:/WORK_PLAYGROUND/RL_test/aha.csv', index=False)