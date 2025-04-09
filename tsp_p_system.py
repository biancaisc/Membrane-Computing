import random
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

seed = 10
np.random.seed(seed)
random.seed(seed)

def generate_cities(city_count):
    return np.random.rand(city_count, 2)

def generate_distance_matrix(cities):
    city_count = len(cities)
    distance_matrix = np.zeros((city_count, city_count))

    for i in range(city_count):
        for j in range(city_count):
            if i != j:
                distance = np.linalg.norm(cities[i] - cities[j])
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
    return distance_matrix

class Ant:
    def __init__(self, distance_matrix, alpha, beta, colony):
        self.alpha = alpha
        self.beta = beta
        self.distance_matrix = distance_matrix
        self.colony = colony

        self.tour = []
        self.distance = 0
        self.current_city = None
        self.visited = set()
        self.pheromone_matrix = None

    def reset(self):
        self.tour = []
        self.distance = 0
        self.current_city = None
        self.visited.clear()
        self.pheromone_matrix = None

    def visit_city(self, city_index):
        self.tour.append(city_index)
        self.visited.add(city_index)  

    def calculate_probabilities(self):
        unvisited = [i for i in range(len(self.distance_matrix)) if i not in self.visited]
        probabilities = []
        
        for city in unvisited:
            pheromone_level = self.pheromone_matrix[self.current_city][city]
            distance = self.distance_matrix[self.current_city][city]
            probability = (pheromone_level ** self.alpha) * ((1 / distance) ** self.beta)
            probabilities.append(probability)
        
        total_probability = sum(probabilities)
        probabilities = [p / total_probability for p in probabilities] if total_probability > 0 else [1/len(probabilities)] * len(probabilities)
        return unvisited, probabilities

    def create_tour(self):
        self.current_city = random.randint(0, len(self.distance_matrix) - 1)
        self.visit_city(self.current_city)

        while len(self.tour) < len(self.distance_matrix):
            unvisited, probabilities = self.calculate_probabilities()
            next_city = np.random.choice(unvisited, p=probabilities)
            self.visit_city(next_city)
            self.current_city = next_city 

        self.visit_city(self.tour[0]) 
        
        self.pheromone_matrix = None
        return self.tour
    
    def step(self):
        if self.pheromone_matrix is None:
            self.colony.tours.append(self.tour)
            self.reset()
        else:
            self.create_tour()

class Ant_colony:
    def __init__(self, cities, distance_matrix, ant_count, iterations, alpha, beta, rho, Q):
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.ant_count = ant_count
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        self.pheromone = np.ones((len(cities), len(cities))) * 0.1

        self.ants = [Ant(distance_matrix=self.distance_matrix, alpha=self.alpha, beta=self.beta, colony=self) for _ in range(ant_count)]
        self.tours = []
        self.rules = []
        self.best_tour = None

    def get_tour_length(self, tour):
        if not tour:
            return float('inf')

        length = 0
        for i in range(len(tour) - 1):
            length += self.distance_matrix[tour[i]][tour[i + 1]]
        return length
    
    def update_pheromone(self):
        self.pheromone *= (1 - self.rho)
        
        for tour in self.tours:
            tour_length = self.get_tour_length(tour)
            if tour_length == 0:
                continue  
            pheromone_deposit = self.Q / tour_length
            
            for i in range(len(tour) - 1):
                city_from = tour[i]
                city_to = tour[i + 1]
                self.pheromone[city_from][city_to] += pheromone_deposit
                self.pheromone[city_to][city_from] += pheromone_deposit

        if self.best_tour is None:
            self.best_tour = min(self.tours, key=self.get_tour_length)
        else:
            for tour in self.tours:
                if self.get_tour_length(tour) < self.get_tour_length(self.best_tour):
                    self.best_tour = tour
        
        print(f"Cel mai bun tur găsit: {self.best_tour} cu lungimea {self.get_tour_length(self.best_tour):.2f}")

        self.tours.clear()

    def step(self):
        if not self.tours:
            for ant in self.ants:
                ant.step()
        else:
            self.update_pheromone()
            for ant in self.ants:
                ant.pheromone_matrix = self.pheromone

    def run(self):
        for i in range(self.iterations):
            self.step()
        return self.best_tour
    
def plot_tour(cities, tour):
    plt.figure(figsize=(8, 8))
    plt.scatter(cities[:, 0], cities[:, 1], marker='o', color='blue')
    plt.plot(cities[tour, 0], cities[tour, 1], color='red')
    for i in range(len(tour) - 1):
        start = tour[i]
        end = tour[i + 1]
        plt.plot([cities[start][0], cities[end][0]], [cities[start][1], cities[end][1]], color='red', linewidth=2)
    plt.title('Ant Tour')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def calculate_mst_length(distance_matrix, tour):
    if tour[0] == tour[-1]:
        nodes = tour[:-1]
    else:
        nodes = tour

    G = nx.Graph()
    G.add_nodes_from(nodes)
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u = nodes[i]
            v = nodes[j]
            weight = distance_matrix[u][v]
            G.add_edge(u, v, weight=weight)
    
    mst = nx.minimum_spanning_tree(G, weight='weight')
    
    mst_length = sum(data['weight'] for u, v, data in mst.edges(data=True))
    
    return mst_length

if __name__ == "__main__":

    city_count = 50
    cities = generate_cities(city_count)
    distance_matrix = generate_distance_matrix(cities)

    ant_count = 50
    iterations = 200
    alpha = 1.0
    beta = 3.0
    rho = 0.1
    Q = 100.0

    colony = Ant_colony(cities, distance_matrix, ant_count, iterations, alpha, beta, rho, Q)
    best_tour = colony.run()
    best_length = colony.get_tour_length(best_tour)

    print(f"Cel mai bun tur găsit: {best_tour} cu lungimea {best_length:.2f}")

    mst_length = calculate_mst_length(distance_matrix, best_tour)
    print(f"Lungimea MST-ului determinat de best tour: {mst_length:.2f}")
    print(f"Raportul dintre lungimea turului și lungimea MST-ului: {best_length / mst_length:.2f}")
    
    plot_tour(cities, best_tour)