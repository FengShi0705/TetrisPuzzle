from copy import deepcopy
import utils
import matplotlib.pyplot as plt

class Solution:

    # At the initializer, we declare all the data structures that we will be using
    def __init__(self, target, fig, ax):
        self.originaltarget = target
        self.target = deepcopy(target)  # the target shape, which we will update with every piece we put
        self.height = len(target)  # number of rows of the target
        self.width = len(target[0])  # number of columns of the target
        self.M = []  # the solution matrix which we will fill with every piece we put
        self.neighbour_map = []  # stores a list of neighbours (adjacent squares) for each square of the target
        self.ind_score_map = []  # stores the individual score for each square of the target
        self.score_map = []  # stores the score of the best 4-block path containing each square of the target
        self.path_map = []  # stores the best 4-block path containing each square of the target
        self.second_phase = False  # indicates if the second phase (filling gaps between loose squares) has started
        self.score_list = [set() for _ in range(0, 49)]  # a list to quickly access to the lowest scoring path
        self.best_score = None  # a variable that stores the best (lowest) score of the list at every moment
        self.piece_count = 0  # a counter that will be incremented by one with each piece we put
        self.fig = fig
        self.ax = ax

        # We fill the maps/lists
        for r in range(0, self.height):

            self.M.append([])
            self.neighbour_map.append([])
            self.ind_score_map.append([])
            self.score_map.append([])
            self.path_map.append([])

            for c in range(0, self.width):

                self.M[r].append((0, 0))
                self.neighbour_map[r].append(None)
                self.ind_score_map[r].append(None)
                self.score_map[r].append(None)
                self.path_map[r].append(None)

                if target[r][c] == 1:  # when we find a target square (==1), we add its coordinates to the node list
                    neighbours, score = self.fill_node(r, c)
                    self.neighbour_map[r][c] = neighbours
                    self.ind_score_map[r][c] = score

    # Fills the neighbour map, the individual score map, the score list, the score map and the path map. This is used
    # only twice: once at the beginning of the program, and once before starting the second phase
    def fill_map(self):
        if self.second_phase:  # for the first phase, this was already done at the initialization so no need to repeat
            for r in range(0, self.height):
                for c in range(0, self.width):
                    if self.target[r][c]:
                        neighbours, score = self.fill_node(r, c)
                        self.neighbour_map[r][c] = neighbours
                        self.ind_score_map[r][c] = score
        for r in range(0, self.height):
            for c in range(0, self.width):
                if self.target[r][c]:
                    score, path = self.get_score_and_path(r, c)
                    if score:  # if score = None, no valid path has been found
                        self.score_list[score].add((r, c))
                        self.score_map[r][c] = score
                        self.path_map[r][c] = path
                        if not self.best_score or score < self.best_score:  # if this is the best score so far, store it
                            self.best_score = score

    # Updates the neighbour map and individual score map for a specified square/node with coordinates (r, c)
    def fill_node(self, r, c):
        score = 0  # here we will store the individual score of the square/node
        neighbours = []  # here we will append all the neighbours that we find

        north = south = east = west = False  # these are used below to make the code more readable

        # First, we check if the square is at the edge of the matrix, to avoid going out of bounds
        north_edge = r == 0
        south_edge = r == self.height - 1
        west_edge = c == 0
        east_edge = c == self.width - 1

        if not self.second_phase:  # first phase
            if not north_edge and self.target[r - 1][c] == 1:  # check neighbours north
                neighbours.append((r - 1, c))
                score += 1
                north = True
                if r > 1 and self.target[r - 2][c] == 1:  # 2nd order (2 squares away) neighbour
                    score += 1
            if not south_edge and self.target[r + 1][c] == 1:  # check neighbours south
                neighbours.append((r + 1, c))
                score += 1
                south = True
                if r < self.height - 2 and self.target[r + 2][c] == 1:
                    score += 1
            if not west_edge and self.target[r][c - 1] == 1:  # check neighbours west
                neighbours.append((r, c - 1))
                score += 1
                west = True
                if c > 1 and self.target[r][c - 2] == 1:
                    score += 1
            if not east_edge and self.target[r][c + 1] == 1:  # check neighbours east
                neighbours.append((r, c + 1))
                score += 1
                east = True
                if c < self.width - 2 and self.target[r][c + 2] == 1:
                    score += 1
            if (north or west) and not north_edge and not west_edge and self.target[r - 1][c - 1] == 1:  # check NW
                score += 1
            if (north or east) and not north_edge and not east_edge and self.target[r - 1][c + 1] == 1:  # check NE
                score += 1
            if (south or west) and not south_edge and not west_edge and self.target[r + 1][c - 1] == 1:  # check SW
                score += 1
            if (south or east) and not south_edge and not east_edge and self.target[r + 1][c + 1] == 1:  # check SE
                score += 1

        else:  # second phase
            if not north_edge:  # check neighbours north
                if self.target[r - 1][c] == 1:
                    neighbours.append((r - 1, c))
                    score += 1
                if r > 1 and self.target[r - 2][c] == 1:  # 2nd order (2 squares away) neighbour
                    neighbours.append((r - 2, c))
                    score += 1
            if not south_edge:  # check neighbours south
                if self.target[r + 1][c] == 1:
                    neighbours.append((r + 1, c))
                    score += 1
                if r < self.height - 2 and self.target[r + 2][c] == 1:
                    neighbours.append((r + 2, c))
                    score += 1
            if not west_edge:  # check neighbours west
                if self.target[r][c - 1] == 1:
                    neighbours.append((r, c - 1))
                    score += 1
                if c > 1 and self.target[r][c - 2] == 1:
                    neighbours.append((r, c - 2))
                    score += 1
            if not east_edge:   # check neighbours east
                if self.target[r][c + 1] == 1:
                    neighbours.append((r, c + 1))
                    score += 1
                if c < self.width - 2 and self.target[r][c + 2] == 1:
                    neighbours.append((r, c + 2))
                    score += 1
            if not north_edge and not west_edge and self.target[r - 1][c - 1] == 1:  # check northwest
                neighbours.append((r - 1, c - 1))
                score += 1
            if not north_edge and not east_edge and self.target[r - 1][c + 1] == 1:  # check northeast
                neighbours.append((r - 1, c + 1))
                score += 1
            if not south_edge and not west_edge and self.target[r + 1][c - 1] == 1:  # check southwest
                neighbours.append((r + 1, c - 1))
                score += 1
            if not south_edge and not east_edge and self.target[r + 1][c + 1] == 1:  # check southeast
                neighbours.append((r + 1, c + 1))
                score += 1

        return neighbours, score

    # Clears the score list. This is used between the first and the second phase
    def clear_score_list(self):
        self.score_list = [set() for _ in range(0, 48)]
        self.best_score = None

    # Returns the best possible 4-block path (and its score) from a specified square with coordinates (r, c). The lower
    # the score of the path, the better it is.
    def get_score_and_path(self, r, c):
        total_score = self.ind_score_map[r][c]
        path = {(r, c)}
        path_length = len(path)

        # In the first phase, this is more straight-forward as we don't deal with bridges or incomplete paths.
        if not self.second_phase:
            while path_length < 4:  # we keep looking for neighbours until we reach a length of 4 blocks
                best_neighbour, best_score = self.get_best_neighbour(path)  # look for the lowest-scoring neighbour
                if not best_neighbour:  # if no neighbours, stop looking
                    break
                else:
                    total_score += best_score  # update the score
                    path.add(best_neighbour)  # add the selected neighbour to the path
                    path_length += 1

        # In the second phase, we have to consider bridges between loose squares and also we may have to manually
        # complete paths of 3 squares
        else:
            piece_includes_bridge = False
            while path_length < 4:
                best_neighbour, best_score, bridge = self.get_best_neighbour(path)
                if not best_neighbour:  # if no neighbours, stop looking
                    break
                else:
                    total_score += best_score
                    path.add(best_neighbour)
                    path_length += 1
                    if bridge:
                        path.add(bridge)
                        path_length += 1
                        total_score += 13  # a bridge has a higher (worse) score than any other block (max is 12, so 13)
                        piece_includes_bridge = True

            if path_length == 3 and not piece_includes_bridge:  # if length of path = 3, we complete it with a free square
                for (r1, c1) in path:
                    if r1 > 0 and self.M[r1 - 1][c1] == (0, 0) and (r1 - 1, c1) not in path:
                        path.add((r1 - 1, c1))
                        path_length += 1
                        total_score += 13
                        break
                    if r1 < self.height - 1 and self.M[r1 + 1][c1] == (0, 0) and (r1 + 1, c1) not in path:
                        path.add((r1 + 1, c1))
                        path_length += 1
                        total_score += 13
                        break
                    if c1 > 0 and self.M[r1][c1 - 1] == (0, 0) and (r1, c1 - 1) not in path:
                        path.add((r1, c1 - 1))
                        path_length += 1
                        total_score += 13
                        break
                    if c1 < self.width - 1 and self.M[r1][c1 + 1] == (0, 0) and (r1, c1 + 1) not in path:
                        path.add((r1, c1 + 1))
                        path_length += 1
                        total_score += 13
                        break

        if path_length < 4:
            total_score = None
            path = None

        return total_score, path

    # Returns the lowest-score neighbour of a group (path) of 1-3 squares
    def get_best_neighbour(self, path):
        best_neighbour = None
        best_score = None

        # At the first phase, this is straight-forward: just iterate through the neighbours and pick the lowest score
        if not self.second_phase:
            for (r, c) in path:
                neighbours = self.neighbour_map[r][c]
                for (nr, nc) in neighbours:
                    if (nr, nc) not in path:
                        current_score = self.ind_score_map[nr][nc]
                        if not best_score or current_score < best_score:
                            best_neighbour = (nr, nc)
                            best_score = current_score
            return best_neighbour, best_score

        # At the second phase, we consider a new type of neighbours that can be 2 squares away, with a gap in the
        # middle. That gap has to be covered by a 'bridge'. We will have to decide whether it is possible to use
        # a bridge in each situation and what is the best place to put it.
        else:
            best_bridge = None
            for (r, c) in path:
                if self.originaltarget[r][c]:  # if the block is not in the target, then it is a bridge/gap
                    neighbours = self.neighbour_map[r][c]
                else:  # if it is a bridge, we have to manually check its neighbours
                    neighbours = []
                    if r > 0 and self.target[r - 1][c] == 1 and self.M[r - 1][c] == (0, 0):  # north
                        neighbours.append((r - 1, c))
                    if r < self.height - 1 and self.target[r + 1][c] == 1 and self.M[r + 1][c] == (0, 0):  # south
                        neighbours.append((r + 1, c))
                    if c > 0 and self.target[r][c - 1] == 1 and self.M[r][c - 1] == (0, 0):  # west
                        neighbours.append((r, c - 1))
                    if c < self.width - 1 and self.target[r][c + 1] == 1 and self.M[r][c + 1] == (0, 0):  # east
                        neighbours.append((r, c + 1))
                for (nr, nc) in neighbours:
                    if (nr, nc) not in path:
                        valid_neighbour = False
                        bridge = None
                        distance = abs(nr - r) + abs(nc - c)
                        if distance == 1:  # we have to check the distance between the block and the neighbour
                            valid_neighbour = True  # for distance = 1 there is no problem
                        elif distance == 2:  # for distance = 2 we need to check if the resulting path would be valid
                            if len(path) <= 2:  # first we check that the path is not too long (we are adding 2 squares)
                                # Now, we need to check if there is a valid bridge between the block and the neighbour
                                # To do so, we compare coordinates to know the relative position of the neighbour
                                if nr == r:  # neighbour is in the same row
                                    if nc > c and self.M[r][c + 1] == (0, 0) and (r, c + 1) not in path:  # check east
                                        bridge = (r, c + 1)  # if it is not, then we go ahead
                                        valid_neighbour = True
                                    elif nc < c and self.M[r][c - 1] == (0, 0) and (r, c - 1) not in path:  # check west
                                        bridge = (r, c - 1)
                                        valid_neighbour = True
                                elif nc == c:  # neighbour is in the same column
                                    if nr > r and self.M[r + 1][c] == (0, 0) and (r + 1, c) not in path:  # check south
                                        bridge = (r + 1, c)
                                        valid_neighbour = True
                                    elif nr < r and self.M[r - 1][c] == (0, 0) and (r - 1, c) not in path:  # check nrth
                                        bridge = (r - 1, c)
                                        valid_neighbour = True
                                else:  # neighbour is in a different row and column
                                    if nr > r and nc > c:  # neighbour is southeast
                                        if self.M[r + 1][c] == (0, 0) and (r + 1, c) not in path:  # check south
                                            bridge = (r + 1, c)
                                            valid_neighbour = True
                                        elif self.M[r][c + 1] == (0, 0) and (r, c + 1) not in path:  # check east
                                            bridge = (r, c + 1)
                                            valid_neighbour = True
                                    elif nr > r and nc < c:  # neighbour is southwest
                                        if self.M[r + 1][c] == (0, 0) and (r + 1, c) not in path:  # check south
                                            bridge = (r + 1, c)
                                            valid_neighbour = True
                                        elif self.M[r][c - 1] == (0, 0) and (r, c - 1) not in path:  # check west
                                            bridge = (r, c - 1)
                                            valid_neighbour = True
                                    elif nr < r and nc > c:  # neighbour is northeast
                                        if self.M[r - 1][c] == (0, 0) and (r - 1, c) not in path:  # check north
                                            bridge = (r - 1, c)
                                            valid_neighbour = True
                                        elif self.M[r][c + 1] == (0, 0) and (r, c + 1) not in path:  # check east
                                            bridge = (r, c + 1)
                                            valid_neighbour = True
                                    elif nr < r and nc < c:  # neighbour is northwest
                                        if self.M[r - 1][c] == (0, 0) and (r - 1, c) not in path:  # check north
                                            bridge = (r - 1, c)
                                            valid_neighbour = True
                                        elif self.M[r][c - 1] == (0, 0) and (r, c - 1) not in path:  # check west
                                            bridge = (r, c - 1)
                                            valid_neighbour = True
                        else:  # we should never reach this point
                            print("Error: distance between loose neighbours should be either 1 or 2")

                        if valid_neighbour:
                            current_score = self.ind_score_map[nr][nc]
                            if not best_score or current_score < best_score:
                                best_neighbour = (nr, nc)
                                best_score = current_score
                                best_bridge = bridge
            return best_neighbour, best_score, best_bridge

    # Checks the score list and returns the best possible 4-block piece to put next. The criteria is lower score first,
    # where score reflects the level of isolation of the block (more isolated = lower score)
    def get_next_piece(self):
        piece = None
        piece_found = False
        if self.best_score:
            while not piece_found and self.best_score < 48:
                if self.score_list[self.best_score]:
                    #block = next(iter(self.score_list[self.best_score]))  # return the first element of the score list
                    block = self.score_list[self.best_score].pop()
                    self.score_map[block[0]][block[1]] = None
                    piece = self.path_map[block[0]][block[1]]
                    piece_found = True
                else:
                    self.best_score += 1
        return piece

    # Updates the score of all the blocks near the latest piece (deleted_blocks)
    def update_map(self, deleted_blocks):

        update_list1, update_list2 = self.get_update_lists(deleted_blocks)  # identify affected blocks

        # First, we will remove all trace of the deleted blocks
        for (r, c) in deleted_blocks:
            if not self.second_phase or self.originaltarget[r][c]:  # condition is met unless (r, c) is a gap/bridge
                # The above condition is always met unless the node is a bridge. The 'not self.second_phase' part
                # is to avoid checking it if we are still in the first phase, in which gaps are not yet filling gaps
                score = self.score_map[r][c]
                if score:  # and (r, c) in self.score_list[score]:
                    self.score_list[score].remove((r, c))  # delete from score list
                    self.score_map[r][c] = None  # delete from score map
                self.path_map[r][c] = None  # delete from path map
                self.neighbour_map[r][c] = None  # delete from neighbour map
                self.ind_score_map[r][c] = None  # delete from individual score map

        # Then, we update the affected blocks' list of neighbours and individual scores
        for (r, c) in update_list1:
            neighbours, score = self.fill_node(r, c)
            self.neighbour_map[r][c] = neighbours
            self.ind_score_map[r][c] = score

        # Finally, we update the score and path for the affected blocks
        for (r, c) in update_list2:
            score = self.score_map[r][c]
            if score:
                self.score_list[score].remove((r, c))  # first, remove the old score from the score list

            new_score, path = self.get_score_and_path(r, c)  # then, get the new score and path
            if new_score:  # finally, update the score and path
                self.score_list[new_score].add((r, c))
                self.score_map[r][c] = new_score
                self.path_map[r][c] = path
                if new_score < self.best_score:
                    self.best_score = new_score
            else:  # if new_score == None, no valid path has been found
                self.score_map[r][c] = None
                self.path_map[r][c] = None

    # Gets all the blocks whose score could be affected when deleting other blocks from the target
    def get_update_lists(self, deleted_blocks):
        list_1 = []  # here we put the nodes whose neighbour list or individual score needs to be updated
        list_2 = []  # here we put the nodes whose path and total score needs to be updated
        if not self.second_phase:
            for (r, c) in deleted_blocks:
                if self.neighbour_map[r][c]:  # if it is not in the nghbr_map, then it is a bridge node and we ignore it
                    list_of_neighbours1 = self.neighbour_map[r][c]
                    for n1 in list_of_neighbours1:
                        if n1 not in list_1 and n1 not in deleted_blocks:
                            list_1.append(n1)
                            if n1 not in list_2:
                                list_2.append(n1)
                        list_of_neighbours2 = self.neighbour_map[n1[0]][n1[1]]
                        for n2 in list_of_neighbours2:
                            if n2 not in list_1 and n2 not in deleted_blocks:
                                list_1.append(n2)
                                if n2 not in list_2:
                                    list_2.append(n2)
                            list_of_neighbours3 = self.neighbour_map[n2[0]][n2[1]]
                            for n3 in list_of_neighbours3:
                                if n3 not in list_2 and n3 not in deleted_blocks:
                                    list_2.append(n3)
        else:
            for (r, c) in deleted_blocks:
                if self.neighbour_map[r][c]:  # if it is not in the nghbr_map, then it is a bridge node and we ignore it
                    list_of_neighbours1 = self.neighbour_map[r][c]
                    for n1 in list_of_neighbours1:
                        if n1 not in list_1 and n1 not in deleted_blocks:
                            list_1.append(n1)
                            if n1 not in list_2:
                                list_2.append(n1)
                        list_of_neighbours2 = self.neighbour_map[n1[0]][n1[1]]
                        for n2 in list_of_neighbours2:
                            if n2 not in list_2 and n2 not in deleted_blocks and get_distance(r, c, n2[0], n2[1]) < 4:
                                list_2.append(n2)
        return list_1, list_2

    # Chooses the right tetromino to fill a region, and updates the target and the solution matrix
    def put_piece(self, piece):
        self.piece_count += 1
        piece_id = choose_tetromino(piece)
        for (r, c) in piece:
            self.M[r][c] = (piece_id, self.piece_count)
            self.target[r][c] = 0

    # Main function, which is the only one that we will call from outside
    def run(self):



        # First phase
        self.fill_map()
        piece = self.get_next_piece()


        while piece:  # if None, exit loop
            self.put_piece(piece)
            self.update_map(piece)

            # Update plot
            ally, allx = zip(*list(piece))
            plotpiece = list(zip(allx, ally))
            utils.update_ax(self.fig, plotpiece, self.ax, self.piece_count)

            piece = self.get_next_piece()

        # Second phase (covering the loose blocks)
        self.second_phase = True
        self.clear_score_list()
        self.fill_map()
        piece = self.get_next_piece()
        while piece:  # if None, exit loop
            self.put_piece(piece)
            self.update_map(piece)

            # Update plot
            ally, allx = zip(*list(piece))
            plotpiece = list(zip(allx, ally))
            utils.update_ax(self.fig, plotpiece, self.ax, self.piece_count)

            piece = self.get_next_piece()



        return self.M


# Used in version 6e. Not used currently
# Calculates distance between two blocks
def get_distance(r1, c1, r2, c2):
    return abs(r2 - r1) + abs(c2 - c1)


# Chooses the appropriate tetromino shape (piece_id) for a given group of 4 blocks
def choose_tetromino(path):
    sorted_path = sorted(path)  # sorts blocks in path by y coordinate (top to bottom, left to right)
    start = sorted_path[0]  # initial block
    y_offset = start[0]  # offset with respect to the initial block
    x_offset = start[1]
    shape = []
    for i in range(1, 4):
        block = sorted_path[i]
        y = block[0] - y_offset  # we compare each block's coordinates with the initial one's
        x = block[1] - x_offset
        shape.append((y, x))

    # Select the piece_id from all 19 possibilities
    if shape == [(0, 1), (1, 0), (1, 1)]:
        piece_id = 1
    elif shape == [(1, 0), (2, 0), (3, 0)]:
        piece_id = 2
    elif shape == [(0, 1), (0, 2), (0, 3)]:
        piece_id = 3
    elif shape == [(1, 0), (2, 0), (2, 1)]:
        piece_id = 4
    elif shape == [(1, -2), (1, -1), (1, 0)]:
        piece_id = 5
    elif shape == [(0, 1), (1, 1), (2, 1)]:
        piece_id = 6
    elif shape == [(0, 1), (0, 2), (1, 0)]:
        piece_id = 7
    elif shape == [(1, 0), (2, -1), (2, 0)]:
        piece_id = 8
    elif shape == [(0, 1), (0, 2), (1, 2)]:
        piece_id = 9
    elif shape == [(0, 1), (1, 0), (2, 0)]:
        piece_id = 10
    elif shape == [(1, 0), (1, 1), (1, 2)]:
        piece_id = 11
    elif shape == [(1, 0), (1, 1), (2, 0)]:
        piece_id = 12
    elif shape == [(1, -1), (1, 0), (1, 1)]:
        piece_id = 13
    elif shape == [(1, -1), (1, 0), (2, 0)]:
        piece_id = 14
    elif shape == [(0, 1), (0, 2), (1, 1)]:
        piece_id = 15
    elif shape == [(0, 1), (1, -1), (1, 0)]:
        piece_id = 16
    elif shape == [(1, 0), (1, 1), (2, 1)]:
        piece_id = 17
    elif shape == [(0, 1), (1, 1), (1, 2)]:
        piece_id = 18
    elif shape == [(1, -1), (1, 0), (2, -1)]:
        piece_id = 19
    else:
        piece_id = None

    return piece_id


# This is the function that we call from outside and which makes the program run
def solution(target,fig,ax):

    sol = Solution(target,fig,ax)
    M = sol.run()

    return M
