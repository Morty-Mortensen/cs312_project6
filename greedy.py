#!/usr/bin/python3
from copy import deepcopy

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
# elif PYQT_VER == 'PYQT4':
#	from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)#perm is a list of int, path instead of perm
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])#give the route to the TSP solution,use the original city list at the end
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''


    #The overall time complexity is O(n^3) since when we are looking around the neigbors to find the closest neighbor, we need to
    # look n cities that are around the current city, so this takes O(n). Then, we repeat this process n times in order
    # to return to the starting city, so O(n^2). Then, we will repeat this whole process for every single cities(n cities,) so I say
    #it's O(n^3)
    #The overall space complexity is O(n^2) since potential_bssf_cost array and potential_bssf_path array contains n lists and in each list,
    #there will be n cities stored
    def greedy(self, time_allowance=60.0):
        #Time and space complexiies for each initialization is O(1)(except line 97) since it just assigning one
        # value to the variable which takes constant time and needs one space to store a value
        start_time = time.time()
        results = {}
        potential_bssf_cost = []
        potentioal_bbsf_path = []

        #Time and space complexites are O(n=number of city objests) since it needs n spaces to hold n city objects in a list
        #so generating this list takes O(n) times
        cities = self._scenario.getCities()

        self.num_cities = len(cities)

        #Time complexity is O(n^3) please see the comment on line81
        #Space complexity is O(n^2) please see the comment on line85
        while time.time() - start_time < time_allowance:
            for city_index in range(len(cities)):
                #For initialization, time and space complexities are O(1) except when we are deepcopying n cities. (In this case,
                # time and spcace complexities are O(n) since we need n spaces to store those cities and generating this array cost
                # O(n).)
                sum_cost = 0
                starting_city = cities[city_index]
                last_city = None
                cities_to_visit = deepcopy(cities)
                visited_cities = []
                visited_cities.append(starting_city)
                cur_city = starting_city
                del cities_to_visit[city_index]

                #Time complexity is O(n^2) since it going to loop through n times till no more city is left to visit, and for each
                #citiy, we need to look n neibours to find the closest neighbor
                #Space complecity is also O(n^2) since potential_bssf_cost array and potential_bssf_path
                # array contains n lists and in each list, there will be n cities stored
                while len(cities_to_visit) > 0:
                    #For time and space complexities for get_neighbors_costs function, see the description in this function below
                    neighbors_costs = self.get_neighbors_costs(cur_city, cities_to_visit)
                    if min(neighbors_costs) == float("inf"):
                        break
                    else:
                        closest_neighbor_index = neighbors_costs.index(min(neighbors_costs))
                        sum_cost += min(neighbors_costs)
                        closest_neighbor = cities_to_visit[closest_neighbor_index]
                        visited_cities.append(closest_neighbor)
                        cur_city = closest_neighbor
                        if len(cities_to_visit) == 1:
                            last_city = cities_to_visit[0]
                        del cities_to_visit[closest_neighbor_index]

                if len(visited_cities) != len(cities):
                    continue

                else:
                    cost_from_last_city_to_first_city = last_city.costTo(starting_city)
                    if cost_from_last_city_to_first_city == float("inf"):
                        continue
                    else:
                        sum_cost += last_city.costTo(starting_city)
                        potential_bssf_cost.append(sum_cost)
                        potentioal_bbsf_path.append(visited_cities)
            break
        bssf_cost = min(potential_bssf_cost)
        bssf_city_path = potentioal_bbsf_path[potential_bssf_cost.index(min(potential_bssf_cost))]
        end_time = time.time()

        #Time and space complexities are both O(1) since it assigning a value is assigned to each variable and it just need one
        #space to store that value except for line 159
        results['cost'] = bssf_cost
        results['time'] = end_time - start_time
        results['count'] = len(potential_bssf_cost)
        #Time complexity is O(n) since it's only looking at the cost per one city and space complexity is also O(n)
        #since list that holds n cities is necessary
        results['soln'] = TSPSolution(bssf_city_path)
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    #Time complexity is O(n) since it needs to loop through the array that has n cities
    #Space complecity is O(n) since the cost array needs space to store n costs(n is the same number as the number of cities since these
    # two things(city and cost) correspond each other)
    def get_neighbors_costs(self, cur_city, cities_to_visit):
        cost = []
        for city_to_visit in cities_to_visit:
            cost.append(cur_city.costTo(city_to_visit))
        return cost

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
    #Time complexity is O(n^2 * n!): n represents the numebrs of states and in worst case, heap will contain  n! of
    # nodes and each node(state) contains 2D matrix that is n * n array.
	#Space complexity is O(n^2 * n!) since in worst case, heap needs a space to store n! nodes and each node(state)
    # needs a space to store 2D matrix that is n * n array.
    def branchAndBound(self, time_allowance=60.0):
        #Time and space complexities are O(1) since it takes constant time to assign a value to a variable, and each variable
        #needs one space to hold value
        start_time = time.time()
        results = {}
        heap = []
        path = []
        self.original_cities = self._scenario.getCities()
        original_city_num = len(self._scenario.getCities())
        num_solutions = 0
        total_staets_created = 1
        num_pruned_states = 0
        city_list = self.original_cities

        #For time and space complexities, please see the destcription in greedy function above.
        bssf = self.greedy(time_allowance=time_allowance)['soln']

        #The time and space complexities for get_initial_reduced_matrix function is O(n^2). For more details,
        # please see the comment in this function below.
        reduced_matrix, lower_bound = self.get_initial_reduced_matrix(city_list)
        path.append(0)
        starting_city = State(reduced_matrix, lower_bound, path)

        #Time and space complexitirs for heapq.heappush() is O(1) since there is no element inside the heap at the beginning,
        #and we just inserting the starting_city which takes constant time and only one space needed in the heap
        heapq.heappush(heap, starting_city)
        max_queue_size = 1

        # Time complexity is O(n^2 * n!) please see the comment on line 182
        # Space complexity is O(n^2 * n!) please see the comment on line 184
        while len(heap) != 0 and time.time() - start_time < time_allowance:
            if len(heap) > max_queue_size:
                max_queue_size = len(heap)

            # Time complexity for heapop is O(logn!) since reorganizing the heap usually is O(logn) but
            # we have n! nodes, so it will be O(logn!)
            # Space complexity for heappop is O(n^2 * n!) since each state has 2D matrix which consist of n rows * n columns,
            # and we have n! nodes in the heap
            parent = heapq.heappop(heap)
            if parent.lb < bssf.cost:
                #For time and space complexities for expand() function, please see the comment in this function.
                children_list = self.expand(parent)
                total_staets_created += len(children_list)
                for child in children_list:
                    if original_city_num == len(child.path):
                        if child.lb < bssf.cost:
                            route = []
                            original_cities = self.original_cities
                            for i in child.path:
                                route.append(original_cities[i])
                            bssf = TSPSolution(route)
                            num_solutions += 1
                    else:
                        if child.lb < bssf.cost:
                            #Time complexity for heappush is O(logn!) since reorganizing the heap usually is O(logn) but
                            #we have n! nodes, so it will be O(logn!)
                            #Space complexity for heappush is O(n^2 * n!) since each state has 2D matrix which consist of n rows * n columns,
                            #and we have n! nodes in the heap
                            heapq.heappush(heap, child)
                        else:
                            num_pruned_states += 1
            else:
                num_pruned_states += 1
                continue

        end_time = time.time()
        #Time and space complexities are both O(1) since it takes constant time to assign value to variable and each variable
        #needs one space to store one element
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = num_solutions
        results['soln'] = bssf
        results['max'] = max_queue_size
        results['total'] = total_staets_created
        results['pruned'] = num_pruned_states
        return results

    #The overall time complexity is  O(n^3) since each child state has 2D array which takes O(n^2) to generate (since
    # this 2D matrix consist of n rows and n columns,) and we have n numbers of these states.
    #The overall space complexity is O(n^3) Each staet has 2D matrix which consist of n rows and n columns and we
    # need enough space to store all the elements to this matrix. Then, children_list consist of n numbers of these states, so
    # the overall space complexity will be O(n^3)
    def expand(self, parent):
        #Time and space complexities of initialization of each array is O(1) since it only takes constant time and just need one
        # space to hold empty array
        children_list = []
        unvisited_cities = []

        #Time and space complexities are both O(n) since it loop through each cities n(numbers of cities) times and
        # unvisited_cities array needs n spaces to store n numbers of integers which represent the index of each city
        for i in range(1, len(self.original_cities)):
            if parent.matrix[parent.path[-1]][i] != float("inf"):
                unvisited_cities.append(i)

        #Time complexity is O(n^2) since we have nested for loops here.
        # For the outer for loop, it loop through each element in unvisited_cities array and for inner for loop, it looping
        # through each row/column of reduce matrix.
        # Space complexity is O(n^2) since each child state needs 2D matrix which needs n rows * n columns.
        for dest_city_index in unvisited_cities:
            # Time and Space complexity for deep copying path array is O(n) since it needs
            # space to hold n(numbers of cities) cities and it cost O(n) time to generate this path array since it has n elements
            #to put into the path array
            path = deepcopy(parent.path)
            path.append(dest_city_index)

            #For time and space complexity, please see above.
            reduced_matrix = deepcopy(parent.matrix)

            #Time and space complexities are O(1) since it take constant time to assign value to lb, and update the lb,
            #and lb need one space to store the value
            lb = parent.lb
            lb += reduced_matrix[parent.path[-1]][dest_city_index]

            #Time complexity is O(n) since it looping through each row/column index in reduce matrix which takes n
            # (the same numbers as numbers of cities) times
            #Space complexity is O(1) since it is just updating the reduce matrix that already exist one position at a time
            for row_index in range(len(reduced_matrix)):
                reduced_matrix[row_index][dest_city_index] = float("inf")
            for col_index in range(len(reduced_matrix)):
                reduced_matrix[parent.path[-1]][col_index] = float("inf")
            reduced_matrix[dest_city_index][parent.path[-1]] = float("inf")

            #For time and space complexities for update_reduce_matrix_and_lb function, see the details below in this function
            lb = self.update_reduce_matrix_and_lb(reduced_matrix, lb)

            child_city = State(reduced_matrix, lb, path)
            children_list.append(child_city)
        return children_list

    #Time complexity is O(n) since we need to loop through n rows/columns. n is the same numbers as the numbers of cities we were given at the
    #beginning
    #Spcae complexity is O(1) since we are assigning min value to min_cost_row/min_cost_col which takes constant space, and also
    #we are just updating lower_bound and reduce_matrix which already exist and these also takes constant space
    def update_reduce_matrix_and_lb(self, reduced_matrix, lb):
        for row_index in range(0, len(self.original_cities)):
            min_cost_row = min(reduced_matrix[row_index])
            if min_cost_row == 0 or min_cost_row == float("inf"):
                continue
            else:
                lb += min_cost_row
                for col_index in range(0, len(self.original_cities)):
                    reduced_matrix[row_index][col_index] = reduced_matrix[row_index][col_index] - min_cost_row

        for col_index in range(0, len(self.original_cities)):
            min_cost_col = self.get_min_value_per_col(reduced_matrix,col_index)
            if min_cost_col == 0 or min_cost_col == float("inf"):
                continue
            else:
                lb += min_cost_col
                for row_index in range(0, len(self.original_cities)):
                    reduced_matrix[row_index][col_index] = reduced_matrix[row_index][col_index] - min_cost_col
        return lb

    #Time complexity is O(n) since it loops through each row which is the same numbers as the numbers of cities
    #Space complecity is O(n) since col_values array needs to have n spaces to store n cost values from the givien matrix
    def get_min_value_per_col(self,reduced_matrix, col_index):
        col_values = []
        for row_index in range(0, len(self.original_cities)):
            col_values.append(reduced_matrix[row_index][col_index])
        min_value_col = min(col_values)
        return min_value_col

    #The overall time complexity is O(n^2) since we are generating 2D matrix and we have nested for loops that loops n(number of cities)
    # times for row and also n(numbers of cities) times for column.
    #The overall space complexity is also O(n^2) since 2D matrix needs space of n^2 (n rows * n columns.)
    def get_initial_reduced_matrix(self, city_list):
        #Initialization of 2D matrix. Time and spcace complexities are both O(n^2) since it loops through n times to make roww and n times for making columns,
        #and need enough space to hold this n rows * n columns 2D matrix.
        reduced_matrix = [[float("inf") for _ in range(len(city_list))] for _ in range(len(city_list))]

        #Time complexity is O(n^2) since we have nested for loops that loop through city_list that has n cities
        #Space complecity is O(1) since we are reassigning one thing to existing 2D matrix's one specific position,
        # and also assigning a value to cost which need only O(1) space
        for from_index, cur_city in enumerate(city_list):
            for destination_index, dest_city in enumerate(city_list):
                if from_index == destination_index:
                    continue
                cost = cur_city.costTo(dest_city)
                reduced_matrix[from_index][destination_index] = cost
        lower_bound = 0

        # Time complexity is O(n^2) since we have nested for loops and we loop through n(numbers of cities) times
        # for both rows and columns
        #Spcae complexity is O(1) since we are assigning min value to min_cost_row which takes constant time, and also
        #we are just updating lower_bound and reduce_matrix which already exist and these also takes constant time
        for row_index in range(0, len(city_list)):
            min_cost_row = min(reduced_matrix[row_index])
            lower_bound += min_cost_row
            for col_index in range(0, len(city_list)):
                reduced_matrix[row_index][col_index] = reduced_matrix[row_index][col_index] - min_cost_row

        # Time complexity is O(n^2) since we have nested for loops and we loop through n(numbers of cities) times
        # for both rows and columns
        #Space complexity is O(n) since we need space to store n elements to col_values array.
        for col_index in range(0, len(city_list)):
            col_values = []
            for row_index in range(0, len(city_list)):
                col_values.append(reduced_matrix[row_index][col_index])
            # Spcae complexity is O(1) for lines 387, 388, and 390 since we are assigning min value to min_cost_col
            # which takes constant time, and also
            # we are just updating lower_bound and reduce_matrix which already exist and these also takes constant time
            min_cost_col = min(col_values)
            lower_bound += min_cost_col
            for row_index in range(0, len(city_list)):
                reduced_matrix[row_index][col_index] = reduced_matrix[row_index][col_index] - min_cost_col

        return reduced_matrix, lower_bound

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

    def fancy(self, time_allowance=60.0):
        pass


class State:
    #Time and space complexity for initialization is O(1) since it takes constant time to assign value to a variable and
    #we just need one space to store each value
    def __init__(self, matrix, lb, path):
        self.matrix = matrix
        self.lb = lb
        self.path = path

    #Time and space complexity is O(1) since it just need to return the booleant which is constant time and need one
    # space for the booleant(True or False)
    def __lt__(self, other):
        if len(self.path) > len(other.path):
            return True
        if len(self.path) == len(other.path):
            if self.lb < other.lb:
                return True
        else:
            return False