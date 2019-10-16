import numpy as np
import matplotlib.pyplot as plt


class Tetris:

    def __init__(self, width=10, height=20, action_per_frame=2, render=True):
        # parameters
        self.render = render
        self.width = width
        self.height = height
        self.actions_per_frame = action_per_frame
        self.actions = ["left", "right", "up", "down", "fall", "none"]
        self.num_actions = len(self.actions)
        self.done = False

        # init board
        self.board = np.zeros([self.height, self.width])
        self.moving = np.zeros([self.height, self.width])
        self.moving_w = -1
        self.moving_h = -1
        self.moving_piece_id = -1
        self.moving_piece_rotate_id = -1
        if render:
            self.color_map = [[0,0,0], [0,1,1], [0,0,1], [1,1,0], [1,0.5,0], [0, 1, 0], [0.5,0,1], [1,0,0]]
            self.color_codes = np.zeros([self.height, self.width])
            self.color = np.zeros([self.height, self.width, 3])
            self.moving_color_code = 0
        self.time = 0
        self.stuck_count = 0
        self.score = 0

        # init pieces
        self.piecies = self.__init_pieces__()
        self.spawn_positions = self.__init_spawn_positions__() # w & vacant h

    def print_pieces(self):
        H, W = 0, 0
        for piece in self.piecies:
            H += piece[0].shape[0] + 1
            curW = 0
            for rotated in piece:
                curW += rotated.shape[1] + 1
            W = max(W, curW)
        img = np.zeros([H, W])
        h = 0
        for piece in self.piecies:
            w = 0
            for rotated in piece:
                img[h:h+rotated.shape[0], w:w+rotated.shape[1]] = rotated
                w += rotated.shape[1] + 1
            h += piece[0].shape[0] + 1
        plt.imshow(img)
        plt.show()

    def get_printed_state(self):
        if self.render:
            for i in range(len(self.color_map)):
                self.color[self.color_codes == i] = self.color_map[i]
                self.color[self.moving == 1] = self.color_map[self.moving_color_code]
            return self.color
        else:
            return np.maximum(self.board, self.moving)

    def print_state(self):
        if self.render:
            for i in range(len(self.color_map)):
                self.color[self.color_codes == i] = self.color_map[i]
                self.color[self.moving == 1] = self.color_map[self.moving_color_code]
            plt.imshow(self.color)
        else:
            plt.imshow(self.board + self.moving * 2)
        plt.show()

    def start(self):
        self.board = np.zeros([self.height, self.width])
        self.moving = np.zeros([self.height, self.width])
        if self.render:
            self.color_codes = np.zeros([self.height, self.width])
            self.color = np.zeros([self.height, self.width, 3])
        self.time = 0
        self.score = 0
        self.__spawn_piece__()
        self.stuck_count = 0
        self.done = False

    def step(self, action_id):
        # return new_state, reward, done
        if self.done:
            print("Game finished")
            return self.moving, self.board, 0, self.done
        self.time += 1
        self.__try_move__(action_id)
        reward = 1
        done = False
        if self.time % self.actions_per_frame == 0:
            try_result = self.__try_move__(3)
            if not try_result:
                self.stuck_count += 1
                if self.stuck_count == self.actions_per_frame:
                    self.board = np.maximum(self.board, self.moving)
                    if self.render:
                        self.color_codes[self.moving > 0] = self.moving_color_code
                    self.moving = np.zeros(self.moving.shape)
                    reward += self.__remove_rows__()
                    if not self.__spawn_piece__():
                        done = True
                        reward = -100
                    self.stuck_count = 0
                    self.done = done
        return self.get_state(), reward, done

    def get_state(self):
        return [self.moving, self.board]

    def __remove_rows__(self):
        new_board = np.zeros(self.board.shape)
        new_color_codes = np.zeros(self.board.shape)
        count = 0
        full = np.all(self.board >= 1, axis=1)
        cur_row = self.height - 1
        for i in range(self.height):
            if not full[self.height - 1 - i]:
                new_board[cur_row] = self.board[self.height - 1 - i]
                new_color_codes[cur_row] = self.color_codes[self.height - 1 - i]
                cur_row -= 1
            else:
                count += 1
        self.board = new_board
        self.color_codes = new_color_codes
        if count != 0:
            print("Removed %d lines!" % count)
        return count * (count + 1) * 5


    def __spawn_piece__(self):
        self.moving_piece_id = np.random.randint(len(self.piecies))
        self.moving_piece_rotate_id = np.random.randint(len(self.piecies[self.moving_piece_id]))
        rotated = self.piecies[self.moving_piece_id][self.moving_piece_rotate_id]
        w, dh = self.spawn_positions[self.moving_piece_id][self.moving_piece_rotate_id]
        self.moving[0:rotated.shape[0]-dh, w:w+rotated.shape[1]] = rotated[dh:, :]
        self.moving_h = -dh
        self.moving_w = w
        if self.render:
            self.moving_color_code = self.moving_piece_id + 1
        if np.any(np.logical_and(self.moving > 0, self.board > 0)):
            return False
        return True

    def __try_move__(self, action_id):
        if action_id == 0:  # left
            if np.sum(self.moving[:, 0]) > 0:
                return False
            new_moving = np.zeros(self.moving.shape)
            new_moving[:, :-1] = self.moving[:, 1:]
            if np.any(np.logical_and(new_moving > 0, self.board > 0)):
                return False
            self.moving = new_moving
            self.moving_w -= 1
            return True
        if action_id == 1:  # right
            if np.sum(self.moving[:, -1]) > 0:
                return False
            new_moving = np.zeros(self.moving.shape)
            new_moving[:, 1:] = self.moving[:, :-1]
            if np.any(np.logical_and(new_moving > 0, self.board > 0)):
                return False
            self.moving = new_moving
            self.moving_w += 1
            return True
        if action_id == 2:  # up
            new_rotate_id = (self.moving_piece_rotate_id + 1) % len(self.piecies[self.moving_piece_id])
            if self.moving_h < 0 and self.moving_h + self.spawn_positions[self.moving_piece_id][new_rotate_id][1] < 0:
                return False
            new_moving = np.zeros(self.moving.shape)
            rotated = self.piecies[self.moving_piece_id][new_rotate_id]
            if self.moving_w + rotated.shape[1] > self.moving.shape[1]:
                return False
            if self.moving_w < 0:
                return False
            if self.moving_h + rotated.shape[0] > self.moving.shape[0]:
                return False
            new_moving[self.moving_h:self.moving_h + rotated.shape[0], self.moving_w:self.moving_w + rotated.shape[1]] = rotated
            if np.any(np.logical_and(new_moving > 0, self.board > 0)):
                return False
            self.moving_piece_rotate_id = new_rotate_id
            self.moving = new_moving
            return True
        if action_id == 3:  # down
            if np.sum(self.moving[-1, :]) > 0:
                return False
            new_moving = np.zeros(self.moving.shape)
            new_moving[1:, :] = self.moving[:-1, :]
            if np.any(np.logical_and(new_moving > 0, self.board > 0)):
                return False
            self.moving = new_moving
            self.moving_h += 1
            return True
        if action_id == 4:  # fall
            count = 0
            while self.__try_move__(3):
                count += 1
            if count == 0:
                return False
            return True
        return True

    def __init_spawn_positions__(self):
        spawns = []
        for piece in self.piecies:
            piece_spawn = []
            for rotated in piece:
                vacant_rows = np.min(np.where(np.sum(rotated, axis=1) >= 1))
                spawn_w = int((self.width - rotated.shape[1]) / 2)
                piece_spawn.append((spawn_w, vacant_rows))
            spawns.append(piece_spawn)
        return spawns

    def __init_pieces__(self):
        I_piece = [
            np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0]
            ]),
            np.array([
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
            ])
        ]
        O_piece = [
            np.array([
                [1, 1],
                [1, 1]
            ])
        ]
        J_piece = [
            np.array([
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 1]
            ]),
            np.array([
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 0]
            ]),
            np.array([
                [1, 0, 0],
                [1, 1, 1],
                [0, 0, 0]
            ]),
            np.array([
                [0, 1, 1],
                [0, 1, 0],
                [0, 1, 0]
            ])
        ]
        L_piece = [
            np.array([
                [0, 0, 0],
                [1, 1, 1],
                [1, 0, 0]
            ]),
            np.array([
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ]),
            np.array([
                [0, 0, 1],
                [1, 1, 1],
                [0, 0, 0]
            ]),
            np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 1]
            ])
        ]
        S_piece = [
            np.array([
                [0,0,0],
                [0,1,1],
                [1,1,0]
            ]),
            np.array([
                [0,1,0],
                [0,1,1],
                [0,0,1]
            ])
        ]
        Z_piece = [
            np.array([
                [0,0,0],
                [1,1,0],
                [0,1,1]
            ]),
            np.array([
                [0,0,1],
                [0,1,1],
                [0,1,0]
            ])
        ]
        T_piece = [
            np.array([
                [0,0,0],
                [1,1,1],
                [0,1,0]
            ]),
            np.array([
                [0,1,0],
                [1,1,0],
                [0,1,0]
            ]),
            np.array([
                [0,1,0],
                [1,1,1],
                [0,0,0]
            ]),
            np.array([
                [0,1,0],
                [0,1,1],
                [0,1,0]
            ])
        ]
        return [I_piece, J_piece, L_piece, O_piece, S_piece, T_piece, Z_piece]
