import torch
import numpy as np
import pygame
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet

# Initialize pygame for user input
pygame.init()

# Constants for key presses
KEY_MAPPING = {
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_UP: Direction.UP,
    pygame.K_DOWN: Direction.DOWN
}

class Agent:
    def __init__(self):
        self.model = Linear_QNet(11, 256, 3)
        self.model.load_state_dict(torch.load("./model/model.pth"))
        self.model.eval()

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

def play_game():
    agent = Agent()
    game = SnakeGameAI()

    clock = pygame.time.Clock()

    # Alternating between AI and user moves
    is_user_turn = True

    while True:
        if is_user_turn:
            # Handle user input
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in KEY_MAPPING:
                        direction = KEY_MAPPING[event.key]
                        game.direction = direction  # Update the game direction
                        is_user_turn = False  # Switch turn to AI

            # Perform a user-controlled step in the game
            reward, done, score = game.play_step_user()

        else:
            # AI makes a move
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            is_user_turn = True  # Switch turn to the user

        if done:
            game.reset()
            print('Game Over. Score:', score)

        # Limit the frame rate
        clock.tick(10)  # Adjust the value for the speed of the game

if __name__ == '__main__':
    play_game()
