#!/usr/bin/python
# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import time
import os
from IPython import display


class TicTacToe(object):
    """实现一个简单的井字过三关"""
    __board = np.zeros(9)  # 棋盘
    __next = 1  # 当前等待落子的玩家，{1:A, -1:B}
    __move_cnt = 0  # 当前步数

    def __init__(self):
        self.__board = np.zeros(9)
        self.__next, self.__move_cnt = 1, 0

    def reset(self):
        """重置对局"""
        self.__board = np.zeros(9)
        self.__next, self.__move_cnt = 1, 0

    def show_board(self):
        """输出当前棋盘"""
        print self.__board[0:3]
        print self.__board[3:6]
        print self.__board[6:9]

    def move(self, pos):
        """落子"""
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
        """获取当前为0的位置，即可以落子的位置"""
        ret = []
        for i in range(9):
            if self.__board[i] == 0:
                ret.append(i)
        return ret

    def get_next(self):
        return self.__next

    def win_check(self):
        """检测胜负关系"""
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
        """
        将当前棋盘映射为一个整数id，用以标记当前局面。
        先将[-1,0,1]变为[0,1,2]并将其视为三进制数字，而后转为十进制数字
        """
        temp = [i+1 for i in board]
        ret = 0
        for i in range(9):
            ret += temp[i] * (3 ** i)
        return int(ret)

    def get_state(self):
        """返回当前局面信息"""
        return self.__board.copy(), self.__next, self.__move_cnt, TicTacToe.board2int(self.__board)


class QLearning(object):
    """简易的基于Q-learning算法的RL"""
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
        """
        利用 ε-贪心法进行行为选择，以ε的概率选择当前最优的行为，以1-ε的概率随机选择一个行为；
        训练初期应将ε设置为一个较小的值，使得可以尝试更多不同的对局，否则容易陷入局部最优值而无法获取全局最优值。
        ε=0 为仅探索模式， ε=1 为仅利用模式
        :param epsilon: 0 <= ε <= 1
        """
        state = self.__game.get_state()
        # print state
        state_id = TicTacToe.board2int(state[0])
        # print state_id
        state_act = self.__q_table.iloc[state_id, :]
        # print state_act
        available_act = self.__game.get_available_pos()
        # print available_act
        # if np.random.uniform() > epsilon or state_act[available_act].all() != 0:
        if np.random.uniform() > epsilon:
            action = np.random.choice(available_act)
        else:
            action = state_act[available_act].idxmax()
        # print action
        return action

    def env_feedback(self, action):
        """计算每一步的收益反馈，已弃用"""
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

    def env_feedback_v2(self, action):
        """
        计算每一步的收益reward
        获胜50，失败-50，平局0
        未结束时，则基于当前局面双方有多少可获胜三元组以及对方是否存在 "将获胜" 三元组决定收益
        """
        self.__game.move(action)
        state = self.__game.get_state()
        res = self.__game.win_check()
        end = 0
        goal_list = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                     [0, 3, 6], [1, 4, 7], [2, 5, 8],
                     [0, 4, 8], [2, 4, 6]]
        if res == 0:
            count_a, count_b, crit_a, crit_b = 0, 0, 0, 0
            for goal in goal_list:
                tmp = state[0][goal]
                if sum(tmp) >= 2:
                    count_a += 1
                    crit_a += 1
                elif sum(tmp) <= -2:
                    count_b += 1
                    crit_b += 1
                elif sum(tmp) == 1 and tmp.prod() == 0:
                    count_a += 1
                elif sum(tmp) == -1 and tmp.prod() == 0:
                    count_b += 1
                elif sum(tmp) == 0 and sum([i**2 for i in tmp]) == 0:
                    count_a += 1
                    count_b += 1
                else:
                    pass
            if state[1] == 1 and crit_a > 0:
                reward = -50
            elif state[1] == -1 and crit_b > 0:
                reward = -50
            else:
                rate = ((count_a + 0.01) ** (-state[1])) * ((count_b + 0.01) ** state[1])
                reward = (rate - 1.5) * 10
                reward = min(reward, 50)
                reward = max(reward, -50)
        elif res == 2:
            reward, end = 0, 1
        elif res + self.__game.get_next() == 0:
            reward, end = 50, 1
        else:
            reward, end = -50, 1
        return reward, end, res

    def __update_q_table(self, old_id, new_id, action, end, reward):
        """
        更新上一步的Q-table数值
        会将当前状态下（对手）的Q-table最大值，乘以折损因子，作为扣除项
        lr为学习率
        :param old_id: 上一个状态的id
        :param new_id: 当前状态的id
        :param action: 上一步所选行为
        :param end: 是否终局
        :param reward: 收益
        """
        q_original = self.__q_table.loc[old_id, action]
        if not end:
            tmp = (1 - self.__gamma) * reward - self.__gamma * self.__q_table.iloc[new_id].max()
        else:
            tmp = reward
        self.__q_table.loc[old_id, action] = (1 - self.__lr) * q_original + self.__lr * tmp
        return

    def __show(self):
        os.system('cls')
        self.__game.show_board()
        time.sleep(1.3)

    def run(self, max_episodes, epsilon):
        df = pd.read_csv('D:/WORK_PLAYGROUND/RL_test/aha.csv')
        self.__q_table = df.copy()
        self.__q_table.columns = range(9)
        episodes = 0
        while episodes <= max_episodes:
            end = 0
            self.__game.reset()
            if episodes % 1000 == 0:
                pass
                self.__show()
            while not end:
                act = self.act_choose(epsilon)
                state = self.__game.get_state()
                old_id = state[3]
                reward, end, res = self.env_feedback_v2(act)
                self.__update_q_table(old_id, self.__game.get_state()[3], act, end, reward)
                if episodes % 1000 == 0:
                    pass
                    self.__show()
            if end:
                episodes += 1
            if episodes % 10000 == 0:
                self.__q_table.to_csv('D:/WORK_PLAYGROUND/RL_test/aha.csv', index=False)
                print('Q-table saved.')
                print time.ctime()
                time.sleep(1)
        return self.__q_table.copy()


if __name__ == '__main__':
    ql = QLearning(0.5, 0.3)
    x = ql.run(max_episodes=20000, epsilon=1)
    x.to_csv('D:/WORK_PLAYGROUND/RL_test/aha.csv', index=False)
