import random
import numpy as np


class GAME:
    def __init__(self, grid_size=12):
        self.grid_size = grid_size
        self.grid = [[2 if (i == 0 or i == grid_size - 1 or j == 0 or j == grid_size - 1) else 0
                       for i in range(grid_size)] for j in range(grid_size)]

        self.snake = []
        self.free_space = []
        self.current_dir = "up"
        self.alive = True
        self.cherry_pos = []

        self.set_env()

        self.step_useless = 0

    def reset(self):
        self.__init__(self.grid_size)
        self.create_cherry()
        return self.get_state()

    def get_state(self):
        state_grid = [[2 if (i == 0 or i == self.grid_size - 1 or j == 0 or j == self.grid_size - 1) else 0
                        for i in range(self.grid_size)] for j in range(self.grid_size)]

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if [j, i] in self.snake:
                    state_grid[i][j] = 1 if self.snake.index([j, i]) == 0 else 0.5
                elif [j, i] == self.cherry_pos:
                    state_grid[i][j] = -1

        return np.array(state_grid, dtype=np.float32).flatten()

    def step(self, action_idx):
        dirs = ["up", "down", "left", "right"]
        move_str = dirs[action_idx]

        reward = self.make_move(move_str)
        next_state = self.get_state()
        done = not self.alive

        return next_state, reward, done

    def set_env(self):
        self.snake = []
        self.free_space = []

        x = random.randint(1, self.grid_size - 2)
        y = random.randint(1, self.grid_size - 2)

        self.grid[y][x] = 1

        possible_moves = [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]
        while True:
            idx = random.randint(0, len(possible_moves) - 1)
            next_pos = possible_moves[idx]
            if self.grid[next_pos[1]][next_pos[0]] == 0:
                break
            else:
                possible_moves.remove(next_pos)

        self.snake.append([x, y])
        self.snake.append(next_pos)

        if next_pos == [x - 1, y]: self.current_dir = "right"
        elif next_pos == [x + 1, y]: self.current_dir = "left"
        elif next_pos == [x, y - 1]: self.current_dir = "down"
        elif next_pos == [x, y + 1]: self.current_dir = "up"

        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                self.free_space.append([j, i])

        self.free_space.remove(next_pos)
        self.free_space.remove([x, y])

    def create_cherry(self):
        pos = random.choice(self.free_space)
        self.cherry_pos = [pos[0], pos[1]]

    def make_move(self, move):
        score = 0
        contra = {"right": "left", "left": "right", "up": "down", "down": "up"}
        dirs = ["right", "left", "down", "up"]
        moves = [[1, 0], [-1, 0], [0, 1], [0, -1]]

        if move in dirs and contra[move] != self.current_dir:
            idx = dirs.index(move)
            self.current_dir = move
        else:
            idx = dirs.index(self.current_dir)

        dx, dy = moves[idx]
        head = self.snake[0]
        
        # Calculate distance BEFORE moving
        old_dist = abs(head[0] - self.cherry_pos[0]) + abs(head[1] - self.cherry_pos[1])
        
        new_head = [head[0] + dx, head[1] + dy]

        if (new_head[0] > 0 and new_head[1] > 0) and \
           (new_head[0] < self.grid_size - 1 and new_head[1] < self.grid_size - 1) and \
           new_head not in self.snake:

            if new_head in self.free_space:
                self.free_space.remove(new_head)

            self.snake.insert(0, new_head)

            if new_head == self.cherry_pos:
                self.step_useless=-0.5
                score += 10
                self.create_cherry()
            else:
                self.step_useless+=1
                tail = self.snake.pop()
                self.free_space.append(tail)
                
                # Calculate distance AFTER moving
                new_dist = abs(new_head[0] - self.cherry_pos[0]) + abs(new_head[1] - self.cherry_pos[1])
                
                # Reward getting closer, penalize moving away
                if new_dist < old_dist:
                    score += 0.1
                else:
                    score -= 0.05
                    
                
                score -= 0.01*self.step_useless
        else:
            score -= 10
            self.alive = False

        return score
    def draw(self):
        new_image = [[2 if (i == 0 or i == self.grid_size - 1 or j == 0 or j == self.grid_size - 1) else 0
                       for i in range(self.grid_size)] for j in range(self.grid_size)]

        for i in range(self.grid_size):
            print("-" * (self.grid_size * 5))
            row_str = ""
            for j in range(self.grid_size):
                if [j, i] in self.snake:
                    if self.snake.index([j, i]) == 0:
                        row_str += "| 1  "
                        new_image[i][j] = 1
                    else:
                        row_str += "|0.5 "
                        new_image[i][j] = 0.5
                elif [j, i] == self.cherry_pos:
                    row_str += "| -1 "
                    new_image[i][j] = -1
                else:
                    row_str += f"| {self.grid[j][i]}  "
            print(row_str + "|")
        print("-" * (self.grid_size * 5))

        return new_image

    def game(self):
        self.create_cherry()
        while self.alive:
            image = self.draw()
            print(image)
            move = input(">>> ")
            if move == "rst": self.reset()
            score = self.make_move(move)
            print(score)


if __name__ == "__main__":
    g = GAME()
    g.game()