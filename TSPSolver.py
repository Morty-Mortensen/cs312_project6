#!/usr/bin/python3
import math
from copy import deepcopy

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
#elif PYQT_VER == 'PYQT4':
#	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import sys
import greedy



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
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
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
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

	def get_neighbors_costs(self, cur_city, cities_to_visit):
		cost = []
		for city_to_visit in cities_to_visit:
			cost.append(cur_city.costTo(city_to_visit))
		return cost

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












	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		results = {}
		count = 0
		max_queue_size = 0
		total_states_created = 0
		total_states_pruned = 0
		initial_rough_bound = 0.0
		self.BSSF = np.inf
		cities = self._scenario.getCities()
		ncities = len(cities)
		priority_queue = []
		heapq.heapify(priority_queue)
		# --- Space Complexity: O(n^2) -- Initializes a 2d array by the number of cities.
		row_reduced_matrix = [[np.inf] * ncities for i in range(ncities)]

		# Initialize matrix
		# --- Time Complexity: O(n^2) -- In order to initialize the 2d array of the same cities.
		for i in range(ncities):
			for j in range(ncities):
				row_reduced_matrix[i][j] = cities[i].costTo(cities[j])

		row_reduced_matrix, initial_rough_bound = self.reduce_matrix(row_reduced_matrix, ncities)

		# If everything is unreachable, it's probably good to know at the beginning.
		if initial_rough_bound == 0.0:
			raise Exception("Initial Rough Bound cannot be 0 (unless everything is unreachable)")

		# Get initial BSSF from greedy algorithm
		greedy_result = TSPSolver.greedy(self)
		self.BSSF = greedy_result['cost']

		# Initialize the result to the greedy result.
		results = greedy_result

		# Push on the initial state for all cities.
		# --- Time Complexity: O(n) -- In order to iterate through all the cities.
		for city in cities:
			initial_state = State(initial_rough_bound, [city], row_reduced_matrix, 0)
			heapq.heappush(priority_queue, initial_state)


		start_time = time.time()
		# While states are in the queue and it is within the allotted amount of time. Do the TSP.
		# --- Time Complexity: O(n!) -- Goes through each branch in order to find the optimal route.
		while len(priority_queue) != 0 and time.time() - start_time < time_allowance:

			# Update the max queue size
			if len(priority_queue) > max_queue_size:
				max_queue_size = len(priority_queue)

			# Pop off the next highest priority state (based on state's bssf compared to the initial/updated BSSF).
			curr_state = heapq.heappop(priority_queue)

			if curr_state._bound < self.BSSF:
				# Check all next connections from the current state (that that haven't already been attempted)
				# --- Time Complexity: O(n) -- Goes through all the cities that are not currently in the current state's route.
				for to_city in cities:
					if to_city not in curr_state._route:
						# Updated curr_matrix to the curr_state's matrix.
						# --- Time Complexity: O(n^2) -- Creates a new matrix and copies all values from parent matrix over.
						# --- Space Complexity: O(n^2) -- Allocates space for new 2d array.
						new_matrix = [[np.inf] * ncities for i in range(ncities)]
						new_bound = curr_state._bound
						for i in range(ncities):
							for j in range(ncities):
								new_matrix[i][j] = curr_state._reduced_matrix[i][j]

						# Update bound with cost from -> to city.
						from_city = curr_state._route[-1]
						cost_from_last_city_to_curr_city = new_matrix[from_city._index][to_city._index]
						new_bound += cost_from_last_city_to_curr_city

						# Make row of from_city inf and make col of to_city inf (plus the inverse of the from_city,to_city).
						# ---- Time Complexity: O(n^2) --- Goes through and updates no longer valid areas to infinity.
						for i in range(ncities):
							for j in range(ncities):
								if i == from_city._index or j == to_city._index or (i == to_city._index and j == from_city._index):
									new_matrix[i][j] = np.inf

						# Row reduces the current matrix. ---------------------- O(n^2+n^2) --- Goes through the matrix twice, to row and column reduce.
						new_matrix, updated_bound = self.reduce_matrix(new_matrix, ncities)

						# Update bound with new row reduced matrix (add to bound any rows or columns that needed to be updated).
						new_bound += updated_bound
						new_route = curr_state._route.copy()
						new_route.append(to_city)

						# Create new state and increase the depth from its parent state.
						new_state = State(new_bound, new_route, new_matrix, curr_state._depth+1)

						# If no more cities to add, updated BSSF if able to connect back to start.
						if new_state._depth == (ncities - 1):
							bssf = TSPSolution(new_state._route)
							if bssf.cost != np.inf and bssf.cost < self.BSSF:
								results.clear()
								end_time = time.time()
								count += 1
								self.BSSF = bssf.cost
								results['cost'] = bssf.cost
								results['time'] = end_time - start_time
								results['count'] = count
								results['soln'] = bssf
								results['max'] = max_queue_size
								results['total'] = total_states_created
								results['pruned'] = total_states_pruned
						elif new_bound < self.BSSF:
							# If the new state's bound is better than the current BSSF, then add to the priority queue.
							total_states_created += 1
							heapq.heappush(priority_queue, new_state)
						else:
							# This state has been initially pruned.
							total_states_pruned += 1
			else:
				# If bound is greater than the current BSSF, then calculate all the branches that
				# were pruned by not going down this branch (factorial of number of levels left).
				# num_levels_left = ((ncities - 1) - curr_state._depth)
				# total_states_pruned += math.factorial(num_levels_left)
				total_states_pruned += 1
		# Return the best results (either optimal or best within the time limit.
		return results

	def reduce_matrix(self, row_reduced_matrix, ncities):
		# Reduce rows.
		updated_bound = 0.0
		for i in range(ncities):
			min_row_value = np.inf
			for j in range(ncities):
				if row_reduced_matrix[i][j] < min_row_value:
					min_row_value = row_reduced_matrix[i][j]
			for j in range(ncities):
				if min_row_value != np.inf:
					row_reduced_matrix[i][j] -= min_row_value
			if min_row_value != np.inf:
				updated_bound += min_row_value

		# Reduce columns.
		for i in range(ncities):
			min_col_value = np.inf
			for j in range(ncities):
				if row_reduced_matrix[j][i] < min_col_value:
					min_col_value = row_reduced_matrix[j][i]
			for j in range(ncities):
				if min_col_value != np.inf:
					row_reduced_matrix[j][i] -= min_col_value
			if min_col_value != np.inf:
				updated_bound += min_col_value

		return row_reduced_matrix, updated_bound





	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

	def fancy( self,time_allowance=60.0 ):
		count = 0
		start_time = time.time()
		results = {}

		initial_result = self.greedy()
		greedy_bssf = initial_result['soln']
		self.greedy_route = greedy_bssf.route

		self.BSSF = initial_result['cost']
		self.cities = self._scenario.getCities()
		ncities = len(self.cities)

		valid_paths = []

		invalid_from_to_cities = []

		# while True:
		while (time.time() - start_time) < 10:
			city_pairs = []
			for city in self.cities:#city is the starting city
				for inner_city in self.cities:#inner_city is destination city
					if inner_city != city and (city._index,inner_city._index) not in invalid_from_to_cities and city.costTo(inner_city) != float("inf"):
						city_pairs.append((city,inner_city))

			try:
				bssf,path = self.tsp(city_pairs, invalid_from_to_cities)

				if bssf.cost == float("inf") or len(bssf.route) != len(self.cities):
					for i in range(len(path)):
						if i != len(path)-1 and path[i].costTo(path[i+1]) == float("inf"):
							invalid_from_to_cities.append((path[i]._index, path[i+1]._index))
				else:
					print(f'Cost: {bssf.cost} vs BSSF: {self.BSSF}')
					valid_paths.append((bssf.cost, path))
					if bssf.cost < self.BSSF:
						results.clear()
						end_time = time.time()
						count += 1
						self.BSSF = bssf.cost
						results['cost'] = bssf.cost
						results['time'] = end_time - start_time
						results['count'] = count
						results['soln'] = bssf
						results['max'] = None
						results['total'] = None
						results['pruned'] = None
			except:
				print("Invalid Path")



		updated_valid_paths = []

		if len(valid_paths) == 0:
			updated_valid_paths.append((self.BSSF, self.greedy_route))
		else:
			for i in range(len(valid_paths)):
				min_cost = float("inf")
				best_path = []
				for path_cost,path in valid_paths:
					if path_cost < min_cost and (path_cost,path) not in updated_valid_paths:
						min_cost = path_cost
						best_path = path.copy()

				updated_valid_paths.append((min_cost, best_path))



		for path_cost,path in updated_valid_paths:
			for i in range(len(path)):
				if i != 0 and i != len(path) -1:
					orig_cost = path[i].costTo(path[i+1])
					prev_orig_cost = path[i-1].costTo(path[i])
					for curr_city in self.cities:
						if curr_city != path[i] and curr_city != path[i+1]:
							curr_city_cost = curr_city.costTo(path[i+1])
							prev_city_cost = path[i-1].costTo(curr_city)
							if curr_city_cost < orig_cost and prev_city_cost < prev_orig_cost: # if 1500 < 1600 = 100,  if 12000 - 100 < self.BSSF, updated self.BSSF
								updated_path_cost = path_cost - (abs(orig_cost - curr_city_cost) + abs(prev_orig_cost - prev_city_cost))
								if (updated_path_cost < self.BSSF):
									path_cost = updated_path_cost
									new_path = path.copy()
									new_path[i] = curr_city
									bssf = TSPSolution(new_path)
									results.clear()
									end_time = time.time()
									count += 1
									self.BSSF = updated_path_cost
									results['cost'] = updated_path_cost
									results['time'] = end_time - start_time
									results['count'] = count
									results['soln'] = bssf
									results['max'] = None
									results['total'] = None
									results['pruned'] = None
						if time.time() - start_time > time_allowance:
							break
					if time.time() - start_time > time_allowance:
						break
				if time.time() - start_time > time_allowance:
					break
			if time.time() - start_time > time_allowance:
				break



		if len(results) == 0:
			end_time = time.time()
			results['cost'] = self.BSSF
			results['time'] = end_time - start_time
			results['count'] = count
			results['soln'] = self.greedy_route
			results['max'] = None
			results['total'] = None
			results['pruned'] = None

		return results


	def tsp(self, data, invalid_from_to_cities):
		# build a graph
		G = self.build_graph(data)
		# print("Graph: ", G)

		# build a minimum spanning tree
		MSTree = self.minimum_spanning_tree(G)
		# print("MSTree: ", MSTree)

		# find odd vertexes
		odd_vertexes = self.find_odd_vertexes(MSTree)
		print("Odd vertexes in MSTree: ", odd_vertexes)
		# odd_vertexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
		# add minimum weight matching edges to MST
		self.minimum_weight_matching(MSTree, G, odd_vertexes)
		# print("Minimum weight matching: ", MSTree)

		updateMSTree = []
		for src_index,dest_index,cost in MSTree:
			if (src_index, dest_index) not in invalid_from_to_cities:
				updateMSTree.append((src_index,dest_index,cost))

		# print("UPDATED Minimum weight matching: ", updateMSTree)

		# find an eulerian tour
		eulerian_tour = self.find_eulerian_tour(updateMSTree, G)

		# print("Eulerian tour: ", eulerian_tour)

		current = eulerian_tour[0]
		path = [self.cities[current]]
		visited = [False] * len(eulerian_tour)
		visited[current] = True

		length = 0
		v_index = -1
		try:
			for v in eulerian_tour[1:]:
				v_index = v
				if not visited[v]:
					path.append(self.cities[v])
					visited[v] = True

					length += G[current][v]
					current = v
		except:
			invalid_from_to_cities.append((current, v_index))




		print("PATH:")
		for city in path:
			print(city._index, end=" ")
		print()
		bssf = TSPSolution(path)

		return bssf, path


	def get_length(self, x1, y1, x2, y2):
		return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1.0 / 2.0)


	def build_graph(self, data):
		graph = {}

		for src_city,dest_city in data:
			if src_city._index not in graph:
				graph[src_city._index] = {}
			graph[src_city._index][dest_city._index] = src_city.costTo(dest_city)

		return graph



	def minimum_spanning_tree(self, G):
		tree = []
		subtrees = UnionFind()
		for W, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
			if subtrees[u] != subtrees[v]:
				tree.append((u, v, W))
				subtrees.union(u, v)

		return tree


	def find_odd_vertexes(self, MST):
		tmp_g = {}
		vertexes = []
		for edge in MST:
			if edge[0] not in tmp_g:
				tmp_g[edge[0]] = 0

			if edge[1] not in tmp_g:
				tmp_g[edge[1]] = 0

			tmp_g[edge[0]] += 1
			tmp_g[edge[1]] += 1

		for vertex in tmp_g:
			if tmp_g[vertex] % 2 == 1:
				vertexes.append(vertex)

		return vertexes


	def minimum_weight_matching(self, MST, G, odd_vert):
		import random
		random.shuffle(odd_vert)

		while odd_vert:
			v = odd_vert.pop()
			length = float("inf")
			u = 1
			closest = -1
			for u in odd_vert:
				if v != u and u in G[v].keys() and G[v][u] < length:
					length = G[v][u]
					closest = u

			if closest != -1:
				MST.append((v, closest, length))
				odd_vert.remove(closest)

	# MatchedMSTree is a list and stores tuple(u, v, W)
	# G is a map. graph[this._index][another_point._index] = this.costTo(another_point)
	def find_eulerian_tour(self, MatchedMSTree, G):
		# find neigbours
		neighbours = {}
		for edge in MatchedMSTree:
			if edge[0] not in neighbours:
				neighbours[edge[0]] = []

			if edge[1] not in neighbours:
				neighbours[edge[1]] = []

			if self.cities[edge[0]].costTo(self.cities[edge[1]]) != float("inf"):
				neighbours[edge[0]].append(edge[1])

			if self.cities[edge[1]].costTo(self.cities[edge[0]]) != float("inf"):
				neighbours[edge[1]].append(edge[0])



		# finds the hamiltonian circuit
		start_vertex = MatchedMSTree[0][0]
		EP = [neighbours[start_vertex][0]]#

		while len(MatchedMSTree) > 0:
			found_match = False
			for i, v in enumerate(EP):
				if len(neighbours[v]) > 0:
					found_match = True
					break
			if found_match == False:
				break

			while len(neighbours[v]) > 0:
				weight = neighbours[v][0]

				self.remove_edge_from_matchedMST(MatchedMSTree, v, weight)

				# if weight not in removedIndexes:
				del neighbours[v][(neighbours[v].index(weight))]
				if v in neighbours[weight]:
					del neighbours[weight][(neighbours[weight].index(v))]

				i += 1
				EP.insert(i, weight)

				v = weight

		return EP


	def remove_edge_from_matchedMST(self, MatchedMST, v1, v2):

		for i, item in enumerate(MatchedMST):
			if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
				del MatchedMST[i]

		return MatchedMST



class UnionFind:
	def __init__(self):
		self.weights = {}
		self.parents = {}

	def __getitem__(self, object):
		if object not in self.parents:
			self.parents[object] = object
			self.weights[object] = 1
			return object

		# find path of objects leading to the root
		path = [object]
		root = self.parents[object]
		while root != path[-1]:
			path.append(root)
			root = self.parents[root]

		# compress the path and return
		for ancestor in path:
			self.parents[ancestor] = root
		return root

	def __iter__(self):
		return iter(self.parents)

	def union(self, *objects):
		roots = [self[x] for x in objects]
		heaviest = max([(self.weights[r], r) for r in roots])[1]
		for r in roots:
			if r != heaviest:
				self.weights[heaviest] += self.weights[r]
				self.parents[r] = heaviest



















class State:
	def __init__( self, bound, route, reduced_matrix, depth ):
		self._bound = bound
		self._route = route
		self._reduced_matrix = reduced_matrix
		self._depth = depth

	def __lt__(self, other):
		if self._depth > other._depth:
			return True
		elif self._bound < other._bound:
			return True
		else:
			return False
		



