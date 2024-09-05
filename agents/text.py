pacman_pos = currentGameState.getPacmanPosition()
    
    # get the state of the pacman position on the maze, (help to determine weather it is a dead end or tunnel or corner)
    pos_maze_state: MazeStateInstance = maze_info[pacman_pos[0]][pacman_pos[1]]
    maze_state_score = 0    

    # check is pacman in a dead end path
    if pos_maze_state == MazeState.DEAD_END:

        # check if the dead end path has food
        if pos_maze_state.path_info.total_food > 0:
            eat_move = (pos_maze_state.index + 1) + (pos_maze_state.path_info.length - (pos_maze_state.index + 1)) * 2
            if nearest_ghost_dist > eat_move or total_scared_time > 0:  # if there is enough move to eat, then process to eat in dead end
                maze_state_score += (nearest_ghost_dist - eat_move) * 50
            else:
                # otherwise, escape from the dead end to avoid beign trap
                maze_state_score -= (eat_move - nearest_ghost_dist) * 50
        else:
            print("Harsh Penaty")
            # apply a harsh penalty to avoid pacman from entering the dead end, the deeper the end, the higher the penalty
            maze_state_score -= 200 * (pos_maze_state.index + 1)

    # check is pacman in a tunnel
    elif pos_maze_state == MazeState.TUNNEL:

        # calculate the distance to exit from both end 
        exit_1 = pos_maze_state.index + 1
        exit_2 = pos_maze_state.path_info.length - pos_maze_state.index

        # calculate the nearest ghost near from both ending point of the tunnel 
        ghost_near_exit1 = min([manhattanDistance(pos_maze_state.path_info.start, ghost.getPosition()) for ghost in ghost_state])
        ghost_near_exit2 = min([manhattanDistance(pos_maze_state.path_info.end, ghost.getPosition()) for ghost in ghost_state])

        #Adjust score based on tunnel characteristics
        if pos_maze_state.path_info.total_food > 0:  # Encourage clearing tunnels with food
            maze_state_score += 50
            if nearest_ghost_dist > min(ghost_near_exit1, ghost_near_exit2):  # Favor clearing if ghosts are far
                maze_state_score += 100 / (exit_1 + exit_2)
            else:  # Penalize staying in the tunnel if ghosts are near
                maze_state_score -= 50 / (exit_1 + exit_2)
        else:  # Penalize unnecessary tunnel visits
            maze_state_score -= 50

    elif pos_maze_state == MazeState.CORNER:
        pass
    # if the position is not any dangerous spot, add some rewards
    else:
        maze_state_score += 10

    if total_scared_time > 0:
        # If ghosts are scared, focus on chasing them
        score += total_scared_time - sum(ghost_dist)
    else:
        # Otherwise, focus on avoiding them
        score += sum(ghost_dist)
    
    score += maze_state_score
    
    tile_visits = visit_freq[pacman_pos[0]][pacman_pos[1]]

    # Penalize for revisiting tiles frequently
    if tile_visits > 0:
        # Apply a penalty based on the frequency of visits
        maze_state_score -= 10 * tile_visits  # Adjust the multiplier to control the strength of the penalty

        # If Pacman has visited the tile too many times, apply a larger penalty
        if tile_visits > 5:  # Threshold can be adjusted
            maze_state_score -= 50  # Additional penalty for excessive revisits
    else:
        # Small reward for visiting a new tile
        maze_state_score += 20  # Encourages exploration of new areas