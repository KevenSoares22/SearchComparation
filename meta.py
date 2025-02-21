import random 
import math
import time
import matplotlib.pyplot as plt

# Função que calcula a distância entre duas cidades (euclidiana)
def calculate_distance(cities, i, j):
    x1, y1 = cities[i]
    x2, y2 = cities[j]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Função para calcular a distância total do tour
def total_distance(tour, cities):
    distance = 0
    for i in range(len(tour) - 1):
        distance += calculate_distance(cities, tour[i], tour[i + 1])
    distance += calculate_distance(cities, tour[-1], tour[0])
    return distance

# Meta-heurística 1: Busca Aleatória
def random_search(cities):
    tour = list(range(len(cities)))
    random.shuffle(tour)
    return tour

# Função para plotar todos os tours em subplots (adaptado para número variável de algoritmos)
def plot_all_tours(cities, tours, titles):
    n = len(tours)
    cols = 2 if n > 3 else n
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 6, rows * 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, (tour, title) in enumerate(zip(tours, titles)):
        x = [cities[j][0] for j in tour] + [cities[tour[0]][0]]
        y = [cities[j][1] for j in tour] + [cities[tour[0]][1]]
        plt.subplot(rows, cols, i + 1)
        plt.plot(x, y, marker='o', linestyle='-', color=colors[i % len(colors)])
        plt.scatter(x, y, color='black')
        plt.title(title)
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.grid(True)

    plt.tight_layout(pad=7.0, w_pad=4.0, h_pad=4.0)
    plt.show()

# Meta-heurística 2: Algoritmo Genético
def genetic_algorithm(cities, generations=100, population_size=50, mutation_rate=0.05, elite_size=2):
    def create_individual():
        tour = list(range(len(cities)))
        random.shuffle(tour)
        return tour

    def crossover(parent1, parent2):
        size = len(parent1)
        child = [-1] * size
        start, end = sorted([random.randint(0, size-1), random.randint(0, size-1)])
        child[start:end] = parent1[start:end]
        current_pos = 0
        for i in range(size):
            if child[i] == -1:
                while parent2[current_pos] in child:
                    current_pos += 1
                child[i] = parent2[current_pos]
        return child

    def mutate(tour):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]

    def select_parent(population):
        tournament_size = 5
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=lambda x: total_distance(x, cities))

    population = [create_individual() for _ in range(population_size)]
    for generation in range(generations):
        population = sorted(population, key=lambda x: total_distance(x, cities))
        new_population = population[:elite_size]
        while len(new_population) < population_size:
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)
        population = new_population
    return min(population, key=lambda x: total_distance(x, cities))

# Meta-heurística 3: 2-opt (Busca Local)
def two_opt(cities, tour):
    best_tour = tour
    best_distance = total_distance(best_tour, cities)
    improvement = True
    while improvement:
        improvement = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                new_distance = total_distance(new_tour, cities)
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improvement = True
        # Para evitar retorno prematuro, a iteração continua enquanto houver melhoria
    return best_tour

# Meta-heurística 4: Busca Tabu
def tabu_search(cities, max_iterations=100, tabu_list_size=10):
    def generate_neighbors(tour):
        neighbors = []
        for i in range(len(tour)):
            for j in range(i + 1, len(tour)):
                neighbor = tour.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors

    current_tour = random_search(cities)
    best_tour = current_tour
    best_distance = total_distance(best_tour, cities)
    tabu_list = []

    for iteration in range(max_iterations):
        neighbors = generate_neighbors(current_tour)
        best_neighbor = None
        best_neighbor_distance = float('inf')
        for neighbor in neighbors:
            neighbor_distance = total_distance(neighbor, cities)
            if neighbor_distance < best_neighbor_distance and neighbor not in tabu_list:
                best_neighbor = neighbor
                best_neighbor_distance = neighbor_distance
        current_tour = best_neighbor
        current_distance = best_neighbor_distance
        if current_distance < best_distance:
            best_tour = current_tour
            best_distance = current_distance
        tabu_list.append(current_tour)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)
    return best_tour

# Meta-heurística 5: Ant Colony Optimization (ACO)
def ant_colony_optimization(cities, iterations=100, ants=20, alpha=1.0, beta=5.0, evaporation_rate=0.5, Q=100):
    n = len(cities)
    # Pré-computa as distâncias e a matriz de heurística (1/distância)
    distances = [[0]*n for _ in range(n)]
    heuristic = [[0]*n for _ in range(n)]


    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i][j] = 0
                heuristic[i][j] = 0
            else:
                d = calculate_distance(cities, i, j)
                distances[i][j] = d
                heuristic[i][j] = 1 / d if d != 0 else 0
    # Inicializa a matriz de feromônios
    pheromone = [[1 for j in range(n)] for i in range(n)]
    
    best_tour = None
    best_distance = float('inf')
    
    for it in range(iterations):
        all_tours = []
        all_distances = []
        for ant in range(ants):
            tour = []
            start = random.randint(0, n-1)
            tour.append(start)
            unvisited = set(range(n))
            unvisited.remove(start)
            current = start
            while unvisited:
                probabilities = []
                denom = 0
                for city in unvisited:
                    tau = pheromone[current][city] ** alpha
                    eta = heuristic[current][city] ** beta
                    denom += tau * eta
                    probabilities.append((city, tau * eta))
                r = random.random()
                cumulative = 0
                next_city = None
                for city, prob in probabilities:
                    cumulative += prob/denom
                    if r <= cumulative:
                        next_city = city
                        break
                if next_city is None:
                    next_city = probabilities[-1][0]
                tour.append(next_city)
                unvisited.remove(next_city)
                current = next_city
            all_tours.append(tour)
            tour_length = total_distance(tour, cities)
            all_distances.append(tour_length)
            if tour_length < best_distance:
                best_distance = tour_length
                best_tour = tour

        # Evaporação do feromônio
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1 - evaporation_rate)
                if pheromone[i][j] < 0.0001:
                    pheromone[i][j] = 0.0001
        
        # Depósito de feromônio pelos caminhos percorridos
        for tour, tour_length in zip(all_tours, all_distances):
            for i in range(n):
                from_city = tour[i]
                to_city = tour[(i+1) % n]
                pheromone[from_city][to_city] += Q / tour_length
                pheromone[to_city][from_city] += Q / tour_length
    return best_tour

# Função para plotar um único tour com indicação do tempo gasto
def plot_tour(cities, tour, title, totalTime=0):
    x = [cities[i][0] for i in tour]
    y = [cities[i][1] for i in tour]
    x.append(x[0])
    y.append(y[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.scatter(x, y, color='red')
    plt.title(title)
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True)
    plt.text(min(x) - 10, min(y) - 18, f"Tempo gasto: {totalTime:.4f} s".replace(".", ","), fontsize=12, color='red')
    plt.show()

# Função principal
def main():
    # Gerando cidades aleatórias    
    
    
    
    accGenetic = 0
    accTabu = 0
    accTwoOpt = 0
    accACO = 0
    for i in range(20):
        cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
    
    # Execução e medição do tempo para cada meta-heurística
    
    # Busca Aleatória (pode ser usada como base para o 2-opt)
        start = time.time()
        tour_random = random_search(cities)
        distance_random = total_distance(tour_random, cities)
        end = time.time()
    #print(f"Distância (Busca Aleatória): {distance_random:.2f}")
    #print(f"Tempo de execução (Busca Aleatória): {end - start:.4f} segundos")
    
    # Algoritmo Genético
        start = time.time()
        tour_genetic = genetic_algorithm(cities, generations=100, population_size=50)
        distance_genetic = total_distance(tour_genetic, cities)
        end = time.time()
        totalTimeGenetic = end - start    
        accGenetic += totalTimeGenetic

        print(f"Distância (Algoritmo Genético): {distance_genetic:.2f}")
        print(f"Tempo de execução (Algoritmo Genético): {totalTimeGenetic:.4f} segundos")
    
    # 2-opt
        start = time.time()
        tour_two_opt = two_opt(cities, tour_random)
        distance_two_opt = total_distance(tour_two_opt, cities)
        end = time.time()
        totalTimeTwo = end - start   
        accTwoOpt += totalTimeTwo
        print(f"Distância (2-opt): {distance_two_opt:.2f}")
        print(f"Tempo de execução (2-opt): {totalTimeTwo:.4f} segundos")
    
    # Busca Tabu
        start = time.time()
        tour_tabu = tabu_search(cities, max_iterations=100, tabu_list_size=10)
        distance_tabu = total_distance(tour_tabu, cities)
        end = time.time()
        totalTimeTabu = end - start   
        accTabu += totalTimeTabu
        print(f"Distância (Busca Tabu): {distance_tabu:.2f}")
        print(f"Tempo de execução (Busca Tabu): {totalTimeTabu:.4f} segundos")
    
    # Ant Colony Optimization (ACO)
        start = time.time()
        tour_aco = ant_colony_optimization(cities, iterations=100, ants=20)
        distance_aco = total_distance(tour_aco, cities)
        end = time.time()
        totalTimeAco = end - start   
        accACO += totalTimeAco
        print(f"Distância (ACO): {distance_aco:.2f}")
        print(f"Tempo de execução (ACO): {totalTimeAco:.4f} segundos")


    accACO = accACO/20
    accTabu = accTabu/20
    accTwoOpt = accTwoOpt/20
    accGenetic = accGenetic/20
    
    # Plotando os tours obtidos
    tours = [tour_genetic, tour_two_opt, tour_tabu, tour_aco]
    titles = [
            f"Algoritmo Genético ({accGenetic:.4f}s) - Distância: {distance_genetic:.2f} Km ".replace(".", ","),
            f"2-opt ({accTwoOpt:.4f}s) - Distância: {distance_two_opt:.2f} Km ".replace(".", ","),
            f"Busca Tabu ({accTabu:.4f}s) - Distância: {distance_tabu:.2f} Km ".replace(".", ","),
            f"ACO ({accACO:.4f}s) - Distância: {distance_aco:.2f} Km ".replace(".", ",")
        ]


    plot_all_tours(cities, tours, titles)

if __name__ == "__main__":
    main()
