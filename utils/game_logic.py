from utils.strategies import RandomStrategy


class Game:
    """Base class for games."""

    def __init__(self):
        """Initialize the game object properties."""
        self.round_idx = 1
        self.scores = {
            "player": 0,
            "computer": 0,
            "draw": 0,
        }

    def round(self, *args, **kwargs) -> dict:
        """Play a round of the game."""
        return {"error": "Not implemented"}


class RockPaperScissors(Game):
    """Class for Rock-Paper-Scissors game."""

    def __init__(self, computer_strategy="random"):
        """Initialize the Rock-Paper-Scissors game object properties.

        Args:
            computer_strategy (str, optional): Strategy for computer player. Defaults to 'random'.
        """
        super().__init__()

        if computer_strategy == "random":
            self.strategy = RandomStrategy()
        else:
            raise ValueError("Invalid computer strategy")

        self.sign2move = {
            "fist": "rock",
            "stop": "paper",
            "palm": "paper",
            "two up": "scissors",
            "two up inverted": "scissors",
            "peace": "scissors",
            "peace inverted": "scissors",
        }

        self.allowed_signs = list(self.sign2move.keys())
        self.choices = list(set(self.sign2move.values()))

    def __decide_who_wins(self, player, computer):
        if player == "rock":
            if computer == "paper":
                return "computer"
            elif computer == "scissors":
                return "player"
            return "draw"

        elif player == "paper":
            if computer == "scissors":
                return "computer"
            elif computer == "rock":
                return "player"
            return "draw"

        else:  # player == 'scissors'
            if computer == "rock":
                return "computer"
            elif computer == "paper":
                return "player"
            return "draw"

    def round(self, player_sign: str) -> dict:
        """Play a round of the Rock-Paper-Scissors game.

        Args:
            player_sign (str): Sign chosen by the player, which is translated to a move.

        Returns:
            dict: Dictionary containing the player's choice, computer's choice, and the winner.
        """
        if player_sign not in self.allowed_signs:
            return {"error": "Invalid sign"}

        player_choice = self.sign2move[player_sign]
        computer_choice = self.strategy()
        who_wins = self.__decide_who_wins(player_choice, computer_choice)
        self.scores[who_wins] += 1
        self.round_idx += 1
        return {
            "player_choice": player_choice,
            "computer_choice": computer_choice,
            "who_wins": who_wins,
        }


if __name__ == "__main__":
    import numpy as np

    game = RockPaperScissors()

    for i in range(10_000):
        player_sign = np.random.choice(game.allowed_signs)
        pack = game.round(player_sign)

        text = f"Player choice: {pack['player_choice']}\n"
        text += f"Computer choice: {pack['computer_choice']}\n"
        text += f"Winner is {pack['who_wins']}"
        text += "\n\n"
        print(text)

    print(game.scores)
