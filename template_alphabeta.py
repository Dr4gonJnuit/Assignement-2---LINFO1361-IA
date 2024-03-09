from agent import Agent

class AlphaBetaAgent(Agent):
    """An agent that uses the alpha-beta pruning algorithm to determine the best move.

    This agent extends the base Agent class, providing an implementation of the play
    method that utilizes the alpha-beta pruning technique to make decisions more efficiently.

    Attributes:
        max_depth (int): The maximum depth the search algorithm will explore.
    """

    def __init__(self, player, game, max_depth):
        """Initializes an AlphaBetaAgent instance with a specified player, game, and maximum search depth.

        Args:
            player (int): The player ID this agent represents (0 or 1).
            game (ShobuGame): The Shobu game instance the agent will play on.
            max_depth (int): The maximum depth of the search tree.
        """
        super().__init__(player, game)
        self.max_depth = max_depth

    def play(self, state, remaining_time):
        """Determines the best action by applying the alpha-beta pruning algorithm.

        Overrides the play method in the base class.

        Args:
            state (ShobuState): The current state of the game.
            remaining_time (float): The remaining time in seconds that the agent has to make a decision.

        Returns:
            ShobuAction: The action determined to be the best by the alpha-beta algorithm.
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
        if self.max_depth <= depth: # That's mean the search tree is too deep
            return True
        
        if self.game.is_terminal(state): # That's mean the game is over
            return True
        
        return False
    
    def eval(self, state):
        """Evaluates the given state and returns a score from the perspective of the agent's player.

        Args:
            state (ShobuState): The game state to evaluate.

        Returns:
            float: The evaluated score of the state.
        """
        agent_id = self.player
        player_to_play = state.to_move # Can be 0 or 1 (Not very useful but it's just for better understanding) -> Can be placed in the if statement below
        
        pieces_player, pieces_opponent = float("inf"), float("inf")
        
        board = state.board
        for _, home_board in enumerate(board): # _ is the home_board id
            pieces_player += min(len(home_board[0]), pieces_player)
            pieces_opponent += min(len(home_board[1]), pieces_opponent)
        
        # If the next player is the agent, that mean the opponent is the agent. Since we want to calculate the score
        # from the perspective of the agent, we need to substract the number of pieces of the opponent that is the agent
        if player_to_play == agent_id:
            return pieces_opponent - pieces_player
        else:
            return pieces_player - pieces_opponent

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
        if self.is_cutoff(state, depth):
            return (self.eval(state), None)
        
        v, move = -float("inf"), None
        for action in state.actions:
            v2, a2 = self.min_value(self.game.result(state, action), alpha, beta, depth + 1)
            if v2 > v:
                v, move = v2, action
                aplha = max(alpha, v)
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
            return (self.eval(state), None)
        
        v, move = float("inf"), None
        for action in state.actions:
            v2, a2 = self.max_value(self.game.result(state, action), alpha, beta, depth + 1)
            if v2 < v:
                v, move = v2, action
                beta = min(beta, v)
            if v <= alpha:
                return (v, move)
        
        return (v, move)