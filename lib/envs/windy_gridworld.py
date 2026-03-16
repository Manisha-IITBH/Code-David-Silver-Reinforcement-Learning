import numpy as np
import sys
from io import StringIO

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class WindyGridworldEnv:

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord


    def _calculate_transition_prob(self, current, delta, winds):

        new_position = np.array(current) + np.array(delta) + np.array([-1,0]) * winds[tuple(current)]

        new_position = self._limit_coordinates(new_position).astype(int)

        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        is_done = tuple(new_position) == (3,7)

        return [(1.0, new_state, -1.0, is_done)]


    def __init__(self):

        self.shape = (7,10)

        nS = np.prod(self.shape)
        nA = 4

        winds = np.zeros(self.shape)
        winds[:,[3,4,5,8]] = 1
        winds[:,[6,7]] = 2

        P = {}

        for s in range(nS):

            position = np.unravel_index(s, self.shape)

            P[s] = {a:[] for a in range(nA)}

            P[s][UP] = self._calculate_transition_prob(position,[-1,0],winds)
            P[s][RIGHT] = self._calculate_transition_prob(position,[0,1],winds)
            P[s][DOWN] = self._calculate_transition_prob(position,[1,0],winds)
            P[s][LEFT] = self._calculate_transition_prob(position,[0,-1],winds)


        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0),self.shape)] = 1.0


        self.nS = nS
        self.nA = nA
        self.P = P
        self.isd = isd

        self.s = np.random.choice(range(nS),p=isd)


    def reset(self):

        self.s = np.random.choice(range(self.nS),p=self.isd)

        return self.s,{}


    def step(self,action):

        transitions = self.P[self.s][action]

        prob,next_state,reward,done = transitions[0]

        self.s = next_state

        return next_state,reward,done,False,{}


    def render(self):

        self._render()


    def _render(self,mode='human',close=False):

        if close:
            return

        outfile = StringIO() if mode=='ansi' else sys.stdout

        for s in range(self.nS):

            position = np.unravel_index(s,self.shape)

            if self.s == s:
                output = " x "

            elif position == (3,7):
                output = " T "

            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()

            if position[1] == self.shape[1]-1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)

        outfile.write("\n")
