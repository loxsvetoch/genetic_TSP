import random
import math
import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, 
                            QGraphicsLineItem, QApplication, QFormLayout, QSpinBox, QDoubleSpinBox, 
                            QPushButton, QLabel, QHBoxLayout, QWidget, QCheckBox, QTableWidget, 
                            QTableWidgetItem, QDialog, QVBoxLayout, QGraphicsTextItem, QFileDialog)
from PyQt5.QtGui import QPen, QBrush
from PyQt5.QtCore import Qt

# Класс города
class City:
    def __init__(self, index, x=None, y=None):
        self.index = index  # Уникальный номер города
        self.x = x          # Координата X
        self.y = y          # Координата Y

# Расчёт расстояния между городами с использованием матрицы смежности
def distance(city1, city2, adjacency_matrix):
    return adjacency_matrix[city1.index][city2.index]

# Парсинг TSP-файла
def parse_tsp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    dimension = None
    edge_weights = []
    display_data = []
    section = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())
        elif line == "EDGE_WEIGHT_SECTION":
            section = "edge_weights"
            continue
        elif line == "DISPLAY_DATA_SECTION":
            section = "display_data"
            continue
        elif line == "EOF":
            break
        elif section == "edge_weights":
            edge_weights.extend(map(int, line.split()))
        elif section == "display_data":
            parts = line.split()
            if len(parts) == 3:
                index, x, y = map(float, parts)
                display_data.append((int(index), x, y))
    
    # Преобразуем веса в полную симметричную матрицу
    adjacency_matrix = [[0] * dimension for _ in range(dimension)]
    k = 0
    for i in range(dimension):
        for j in range(i + 1, dimension):
            adjacency_matrix[i][j] = edge_weights[k]
            adjacency_matrix[j][i] = edge_weights[k]  # Симметрия
            k += 1
    
    # Создаём список городов
    if display_data:
        cities = [City(int(index - 1), x, y) for index, x, y in display_data]
    else:
        cities = [City(i) for i in range(dimension)]
    
    return cities, adjacency_matrix

# Класс маршрута
class Tour:
    def __init__(self, cities, order=None):
        self.cities = cities
        if order is None:
            self.order = list(range(len(cities)))
            random.shuffle(self.order)
        else:
            self.order = order

    def total_distance(self, adjacency_matrix):
        dist = 0
        for i in range(len(self.order)):
            city1 = self.cities[self.order[i]]
            city2 = self.cities[self.order[(i + 1) % len(self.order)]]
            dist += distance(city1, city2, adjacency_matrix)
        return dist

    def fitness(self, adjacency_matrix):
        return 1 / self.total_distance(adjacency_matrix)

# Класс популяции
class Population:
    def __init__(self, cities, size):
        self.cities = cities
        self.size = size
        self.individuals = [Tour(cities) for _ in range(size)]

    def best_individual(self, adjacency_matrix):
        return min(self.individuals, key=lambda tour: tour.total_distance(adjacency_matrix))

# Турнирная селекция
def tournament_selection(population, k, adjacency_matrix):
    selected = random.sample(population.individuals, k)
    return max(selected, key=lambda tour: tour.fitness(adjacency_matrix))

# Кроссовер (Order Crossover)
def order_crossover(parent1, parent2):
    n = len(parent1.order)
    start, end = sorted(random.sample(range(n), 2))
    child_order = [None] * n
    substring = set(parent1.order[start:end + 1])
    for i in range(start, end + 1):
        child_order[i] = parent1.order[i]
    p2_index = 0
    for i in range(n):
        if child_order[i] is None:
            while parent2.order[p2_index] in substring:
                p2_index += 1
            child_order[i] = parent2.order[p2_index]
            p2_index += 1
    return Tour(parent1.cities, child_order)

# Мутация перестановкой
def swap_mutation(tour):
    i, j = random.sample(range(len(tour.order)), 2)
    tour.order[i], tour.order[j] = tour.order[j], tour.order[i]

# Мутация инверсией
def inversion_mutation(tour):
    i, j = sorted(random.sample(range(len(tour.order)), 2))
    tour.order[i:j + 1] = reversed(tour.order[i:j + 1])

# Генетический алгоритм
def genetic_algorithm(cities, pop_size, num_generations, tournament_size, swap_prob, inversion_prob, use_swap, use_inversion, adjacency_matrix):
    population = Population(cities, pop_size)
    best_distances = []
    for generation in range(num_generations):
        new_population = []
        best = population.best_individual(adjacency_matrix)
        new_population.append(best)
        best_distances.append(best.total_distance(adjacency_matrix))
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, tournament_size, adjacency_matrix)
            parent2 = tournament_selection(population, tournament_size, adjacency_matrix)
            child = order_crossover(parent1, parent2)
            if use_swap and random.random() < swap_prob:
                swap_mutation(child)
            if use_inversion and random.random() < inversion_prob:
                inversion_mutation(child)
            new_population.append(child)
        population.individuals = new_population
    return population.best_individual(adjacency_matrix), best_distances

# Класс для окна с таблицей смежности
class AdjacencyTableWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Таблица смежности")
        self.setGeometry(150, 150, 600, 400)
        self.table = QTableWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

    def update_table(self, cities, adjacency_matrix):
        if not cities or not adjacency_matrix:
            return
        n = len(cities)
        self.table.setRowCount(n)
        self.table.setColumnCount(n)
        headers = [str(city.index) for city in cities]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setVerticalHeaderLabels(headers)
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem(f"{adjacency_matrix[i][j]:.2f}")
                self.table.setItem(i, j, item)

# Главное окно приложения
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Задача коммивояжёра - Генетический алгоритм")
        self.setGeometry(100, 100, 1200, 701)

        # Графическая сцена
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.scale_factor = 5  # Масштаб 5x
        self.scene.setSceneRect(0, 0, 100 * self.scale_factor, 100 * self.scale_factor)
        self.view.setScene(self.scene)

        # Панель управления
        self.panel = QWidget()
        layout = QFormLayout()
        self.num_cities_spin = QSpinBox()
        self.num_cities_spin.setMinimum(3)
        self.num_cities_spin.setValue(10)
        layout.addRow("Количество городов:", self.num_cities_spin)
        self.generate_button = QPushButton("Сгенерировать города")
        self.generate_button.clicked.connect(self.generate_cities)
        layout.addWidget(self.generate_button)
        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setMinimum(10)
        self.pop_size_spin.setValue(50)
        layout.addRow("Размер популяции:", self.pop_size_spin)
        self.num_gen_spin = QSpinBox()
        self.num_gen_spin.setMinimum(10)
        self.num_gen_spin.setValue(100)
        self.num_cities_spin.setMaximum(1000)
        self.pop_size_spin.setMaximum(1000)
        self.num_gen_spin.setMaximum(1000)
        layout.addRow("Количество поколений:", self.num_gen_spin)
        self.tournament_size_spin = QSpinBox()
        self.tournament_size_spin.setMinimum(2)
        self.tournament_size_spin.setValue(5)
        layout.addRow("Размер турнира:", self.tournament_size_spin)
        self.swap_prob_spin = QDoubleSpinBox()
        self.swap_prob_spin.setMinimum(0.0)
        self.swap_prob_spin.setMaximum(1.0)
        self.swap_prob_spin.setSingleStep(0.01)
        self.swap_prob_spin.setValue(0.1)
        layout.addRow("Вероятность мутации перестановкой:", self.swap_prob_spin)
        self.use_swap_checkbox = QCheckBox("Использовать мутацию перестановкой")
        self.use_swap_checkbox.setChecked(True)
        layout.addWidget(self.use_swap_checkbox)
        self.inversion_prob_spin = QDoubleSpinBox()
        self.inversion_prob_spin.setMinimum(0.0)
        self.inversion_prob_spin.setMaximum(1.0)
        self.inversion_prob_spin.setSingleStep(0.01)
        self.inversion_prob_spin.setValue(0.1)
        layout.addRow("Вероятность мутации инверсией:", self.inversion_prob_spin)
        self.use_inversion_checkbox = QCheckBox("Использовать мутацию инверсией")
        self.use_inversion_checkbox.setChecked(True)
        layout.addWidget(self.use_inversion_checkbox)
        self.run_button = QPushButton("Запустить алгоритм")
        self.run_button.clicked.connect(self.run_ga)
        layout.addWidget(self.run_button)
        self.compare_button = QPushButton("Сравнить мутации")
        self.compare_button.clicked.connect(self.compare_mutations)
        layout.addWidget(self.compare_button)
        self.show_table_button = QPushButton("Показать таблицу смежности")
        self.show_table_button.clicked.connect(self.show_adjacency_table)
        layout.addWidget(self.show_table_button)
        self.best_distance_label = QLabel("Лучшее расстояние: Н/Д")
        layout.addWidget(self.best_distance_label)
        self.load_tsp_button = QPushButton("Загрузить TSP файл")
        self.load_tsp_button.clicked.connect(self.load_tsp_file)
        layout.addWidget(self.load_tsp_button)

        # Таблица смежности в главном окне
        self.adjacency_table = QTableWidget()
        self.adjacency_table.setRowCount(0)
        self.adjacency_table.setColumnCount(0)
        layout.addWidget(self.adjacency_table)

        self.panel.setLayout(layout)

        # Основной layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.view, 3)
        main_layout.addWidget(self.panel, 1)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Инициализация переменных
        self.cities = []
        self.best_tour = None
        self.adjacency_matrix = None
        self.adjacency_window = None

    def load_tsp_file(self):
        """Загрузка данных из TSP-файла."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите TSP файл", "", "TSP Files (*.tsp);;All Files (*)")
        if file_path:
            self.cities, self.adjacency_matrix = parse_tsp_file(file_path)
            self.num_cities_spin.setValue(len(self.cities))
            self.generate_cities()

    def generate_cities(self):
        """Генерация и отображение городов с случайными координатами."""
        self.scene.clear()
        num_cities = self.num_cities_spin.value()
        
        # Если города ещё не созданы или их количество изменилось, создаём новый список
        if not self.cities or len(self.cities) != num_cities:
            self.cities = []
            for i in range(num_cities):
                # Генерируем случайные координаты в пределах сцены
                x = random.uniform(0, 100 * self.scale_factor)
                y = random.uniform(0, 100 * self.scale_factor)
                self.cities.append(City(i, x, y))
        
        # Если матрица смежности не загружена, генерируем её на основе расстояний
        if self.adjacency_matrix is None or len(self.adjacency_matrix) != num_cities:
            self.adjacency_matrix = [[0] * num_cities for _ in range(num_cities)]
            for i in range(num_cities):
                for j in range(i + 1, num_cities):
                    dist = math.sqrt((self.cities[i].x - self.cities[j].x)**2 + 
                                    (self.cities[i].y - self.cities[j].y)**2)
                    self.adjacency_matrix[i][j] = dist
                    self.adjacency_matrix[j][i] = dist  # Симметрия
        
        # Масштабируем координаты для отображения на сцене
        min_x = min(city.x for city in self.cities)
        max_x = max(city.x for city in self.cities)
        min_y = min(city.y for city in self.cities)
        max_y = max(city.y for city in self.cities)
        width = max_x - min_x or 1 
        height = max_y - min_y or 1
        for city in self.cities:
            city.scaled_x = (city.x - min_x) / width * 100 * self.scale_factor
            city.scaled_y = (city.y - min_y) / height * 100 * self.scale_factor
        
        # Рисуем города на сцене
        for city in self.cities:
            ellipse = QGraphicsEllipseItem(city.scaled_x - 1, city.scaled_y - 1, 2, 2)
            ellipse.setBrush(QBrush(Qt.black))
            self.scene.addItem(ellipse)
            text = QGraphicsTextItem(str(city.index))
            text.setPos(city.scaled_x, city.scaled_y)
            self.scene.addItem(text)
        
        self.best_distance_label.setText("Лучшее расстояние: Н/Д")
        self.best_tour = None
        self.update_adjacency_table()
        if self.adjacency_window and self.adjacency_window.isVisible():
            self.adjacency_window.update_table(self.cities, self.adjacency_matrix)

    def update_adjacency_table(self):
        """Обновление таблицы смежности в главном окне."""
        if not self.cities or self.adjacency_matrix is None:
            return
        n = len(self.cities)
        self.adjacency_table.setRowCount(n)
        self.adjacency_table.setColumnCount(n)
        headers = [str(city.index) for city in self.cities]
        self.adjacency_table.setHorizontalHeaderLabels(headers)
        self.adjacency_table.setVerticalHeaderLabels(headers)
        for i in range(n):
            for j in range(n):
                item = QTableWidgetItem(f"{self.adjacency_matrix[i][j]:.2f}")
                self.adjacency_table.setItem(i, j, item)

    def show_adjacency_table(self):
        """Отображение таблицы смежности в отдельном окне."""
        if not self.cities or self.adjacency_matrix is None:
            return
        if not self.adjacency_window or not self.adjacency_window.isVisible():
            self.adjacency_window = AdjacencyTableWindow(self)
        self.adjacency_window.update_table(self.cities, self.adjacency_matrix)
        self.adjacency_window.show()

    def run_ga(self):
        """Запуск генетического алгоритма."""
        if not self.cities or self.adjacency_matrix is None:
            return
        pop_size = self.pop_size_spin.value()
        num_generations = self.num_gen_spin.value()
        tournament_size = self.tournament_size_spin.value()
        swap_prob = self.swap_prob_spin.value()
        inversion_prob = self.inversion_prob_spin.value()
        use_swap = self.use_swap_checkbox.isChecked()
        use_inversion = self.use_inversion_checkbox.isChecked()
        self.best_tour, distances = genetic_algorithm(
            self.cities, pop_size, num_generations, tournament_size, 
            swap_prob, inversion_prob, use_swap, use_inversion, self.adjacency_matrix
        )
        self.draw_tour()
        self.best_distance_label.setText(f"Лучшее расстояние: {self.best_tour.total_distance(self.adjacency_matrix):.2f}")
        self.plot_distances(distances)

    def compare_mutations(self):
        """Сравнение мутаций перестановкой и инверсией."""
        if not self.cities or self.adjacency_matrix is None:
            return
        pop_size = self.pop_size_spin.value()
        num_generations = self.num_gen_spin.value()
        tournament_size = self.tournament_size_spin.value()
        swap_prob = self.swap_prob_spin.value()
        inversion_prob = self.inversion_prob_spin.value()
        tour_swap, distances_swap = genetic_algorithm(
            self.cities, pop_size, num_generations, tournament_size, 
            swap_prob, inversion_prob, True, False, self.adjacency_matrix
        )
        tour_inversion, distances_inversion = genetic_algorithm(
            self.cities, pop_size, num_generations, tournament_size, 
            swap_prob, inversion_prob, False, True, self.adjacency_matrix
        )
        print(f"Лучшее расстояние с мутацией перестановкой: {tour_swap.total_distance(self.adjacency_matrix):.2f}")
        print(f"Лучшее расстояние с мутацией инверсией: {tour_inversion.total_distance(self.adjacency_matrix):.2f}")
        self.plot_comparison(distances_swap, distances_inversion)

    def draw_tour(self):
        """Отрисовка маршрута на сцене."""
        for item in self.scene.items():
            if isinstance(item, QGraphicsLineItem):
                self.scene.removeItem(item)
        if self.best_tour:
            order = self.best_tour.order
            for i in range(len(order)):
                city1 = self.cities[order[i]]
                city2 = self.cities[order[(i + 1) % len(order)]]
                line = QGraphicsLineItem(city1.scaled_x, city1.scaled_y, city2.scaled_x, city2.scaled_y)
                line.setPen(QPen(Qt.red, 1))
                self.scene.addItem(line)

    def plot_distances(self, distances):
        """График эволюции лучшего расстояния."""
        plt.figure(figsize=(8, 6))
        plt.plot(distances, label="Лучшее расстояние")
        plt.xlabel("Поколение")
        plt.ylabel("Расстояние")
        plt.title("Эволюция лучшего расстояния")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_comparison(self, distances_swap, distances_inversion):
        """График сравнения мутаций."""
        plt.figure(figsize=(8, 6))
        plt.plot(distances_swap, label="Мутация перестановкой")
        plt.plot(distances_inversion, label="Мутация инверсией")
        plt.xlabel("Поколение")
        plt.ylabel("Расстояние")
        plt.title("Сравнение мутаций")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())