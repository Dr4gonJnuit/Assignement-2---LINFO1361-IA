from agent import Agent
import random

class AI(Agent):
    """An agent that plays following your algorithm.

    This agent extends the base Agent class, providing an implementation your agent.

    Attributes:
        player (int): The player id this agent represents.
        game (ShobuGame): The game the agent is playing.
    """
    def __init__(self, player, game):
        """Initializes an AlphaBetaAgent instance with a specified player, game, and maximum search depth.

        Args:
            player (int): The player ID this agent represents (0 or 1).
            game (ShobuGame): The Shobu game instance the agent will play on.
            max_depth (int): The maximum depth of the search tree.
        """
        super().__init__(player, game)
        self.max_depth = 2 # 50 A game rarely goes beyond 50 moves -> but 50 is too much
        self.C = 2

    def play(self, state, remaining_time):
        """Determines the next action to take in the given state.

        Args:
            state (ShobuState): The current state of the game.
            remaining_time (float): The remaining time in seconds that the agent has to make a decision.

        Returns:
            ShobuAction: The chosen action.
        """
        return self.alpha_beta_search(state)

    def is_cutoff(self, state, depth):
        """Determines if the search should be cut off at the current depth.

        Args:
            state (ShobuState): The current state of the game.
            depth (int): The current depth in the search tree.

        Returns:
            bool: True if the search should be cut off, False otherwise.
        """
        if depth >= self.max_depth or self.game.is_terminal(state):  # That's mean the search tree is too deep
            return True

        return False

    def eval(self, state): # Not strong enough
        """Evaluates the given state and returns a score from the perspective of the agent's player.

        Args:
            state (ShobuState): The game state to evaluate.

        Returns:
            float: The evaluated score of the state.
        """
        # TODO: Implement a better evaluation function
        pieces_player, pieces_opponent = float("inf"), float("inf")

        board = state.board
        for home_board_id, home_board in enumerate(board):
            pieces_player = min(len(home_board[self.player]), pieces_player)
            pieces_opponent = min(len(home_board[1 - self.player]), pieces_opponent)

        return self.C * pieces_player - pieces_opponent

    def alpha_beta_search(self, state):
        """Implements the alpha-beta pruning algorithm to find the best action.

        Args:
            state (ShobuState): The current game state.

        Returns:
            ShobuAction: The best action as determined by the alpha-beta algorithm.
        """
        _, action = self.max_value(state, -float("inf"), float("inf"), 0)
        return action

    def max_value(self, state, alpha, beta, depth):
        """Computes the maximum achievable value for the current player at a given state using the alpha-beta pruning.

        This method recursively explores all possible actions from the current state to find the one that maximizes
        the player's score, pruning branches that cannot possibly affect the final decision.

        See: page 200 of the book "Artificial Intelligence: A Modern Approach" by Stuart Russell and Peter Norvig
             for the pseudo-code of the MAX-VALUE.

        Args:
            state (ShobuState): The current state of the game.
            alpha (float): The current alpha value, representing the minimum score that the maximizing player is assured of.
            beta (float): The current beta value, representing the maximum score that the minimizing player is assured of.
            depth (int): The current depth in the search tree.

        Returns:
            tuple: A tuple containing the best value achievable from this state and the action that leads to this value.
                If the state is a terminal state or the depth limit is reached, the action will be None.
        """
        """
        if depth == 0:
            a = list(filter(lambda action: action.passive_board_id == 0 and action.passive_stone_id == 0 and action.active_board_id == 3 and action.active_stone_id == 0 and action.direction == 5 and action.length == 2, state.actions))
            return (float("inf"), a[0] if len(a) > 0 else None)
        """

        if self.is_cutoff(state, depth):
            score = self.eval(state)
            score += random.random() * 0.01 * score
            return (score, None)

        v, move = -float("inf"), None
        for action in state.actions:
            v2, a2 = self.min_value(self.game.result(state, action), alpha, beta, depth + 1)
            if v2 > v:
                v, move = v2, action
                alpha = max(alpha, v)
            if v >= beta:
                return (v, move)

        return (v, move)

    def min_value(self, state, alpha, beta, depth):
        """Computes the minimum achievable value for the opposing player at a given state using the alpha-beta pruning.

        Similar to max_value, this method recursively explores all possible actions from the current state to find
        the one that minimizes the opponent's score, again using alpha-beta pruning to cut off branches that won't
        affect the outcome.

        See: page 200 of the book "Artificial Intelligence: A Modern Approach" by Stuart Russell and Peter Norvig
             for the pseudo-code of the MIN-VALUE.

        Args:
            state (ShobuState): The current state of the game.
            alpha (float): The current alpha value, representing the minimum score that the maximizing player is assured of.
            beta (float): The current beta value, representing the maximum score that the minimizing player is assured of.
            depth (int): The current depth in the search tree.

        Returns:
            tuple: A tuple containing the best value achievable from this state for the opponent and the action that leads to this value.
                If the state is a terminal state or the depth limit is reached, the action will be None.
        """
        if self.is_cutoff(state, depth):
            score = self.eval(state)
            score += random.random() * 0.01 * score
            return (score, None)

        v, move = float("inf"), None
        for action in state.actions:
            v2, a2 = self.max_value(self.game.result(state, action), alpha, beta, depth + 1)
            if v2 < v:
                v, move = v2, action
                beta = min(beta, v)
            if v <= alpha:
                return (v, move)

        return (v, move)
