import sys
import time
from dataclasses import dataclass
from typing import List
import numpy as np
import random
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QApplication, QLabel, QVBoxLayout, QLineEdit, QProgressBar, \
    QComboBox, QCheckBox, QMessageBox
import os
import qdarkstyle
import threading

plt.style.use('dark_background')

random.seed()

SIZE = 2
cities = []
total = np.math.factorial(SIZE - 1)
directional = True
if not directional:
    total /= 2
broken = False
distances = np.zeros((SIZE, SIZE))


def calculate_basics(size, direct, bro):
    global SIZE, total, directional, broken, distances, cities
    SIZE = size
    directional = direct
    broken = bro
    total = np.math.factorial(SIZE - 1)
    distances = np.zeros((SIZE, SIZE))
    cities = [rand_coords() for i in range(SIZE)]
    if not directional:
        total /= 2


@dataclass(order=True)
class Node:
    path: List[int]
    cost: float
    remaining: List[int] = None

    def __post_init__(self):
        self.remaining = [i for i in range(SIZE) if i not in self.path]


class MainWindow(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.abort = False

        func_labels = ['DFS', 'BFS']

        layout = QHBoxLayout(self)
        left = QVBoxLayout(self)
        central = QVBoxLayout(self)
        right = QVBoxLayout(self)

        layout.addLayout(left)
        layout.addLayout(central)
        layout.addLayout(right)

        self.left_plot_view = QLabel("left")
        left.addWidget(self.left_plot_view)
        self.left_progress_bar = QProgressBar(self)
        left.addWidget(self.left_progress_bar)
        self.left_output_label = QLabel(self)
        left.addWidget(self.left_output_label)

        self.right_plot_view = QLabel("right")
        right.addWidget(self.right_plot_view)
        self.right_progress_bar = QProgressBar(self)
        right.addWidget(self.right_progress_bar)
        self.right_output_label = QLabel()
        right.addWidget(self.right_output_label)

        self.left_func_choice = QComboBox()
        self.left_func_choice.addItems(func_labels)
        self.right_func_choice = QComboBox()
        self.right_func_choice.addItems(func_labels)
        combo_layout = QHBoxLayout(self)
        combo_layout.addWidget(self.left_func_choice)
        combo_layout.addWidget(self.right_func_choice)
        central.addLayout(combo_layout)

        size_label = QLabel("Ilosc miast")
        self.sizeBox = QLineEdit()
        size_layout = QHBoxLayout(self)
        size_layout.addWidget(self.sizeBox)
        size_layout.addWidget(size_label)
        central.addLayout(size_layout)

        dir_label = QLabel("Kierunkowy")
        self.directional_checkbox = QCheckBox()
        dir_layout = QHBoxLayout(self)
        dir_layout.addWidget(self.directional_checkbox)
        dir_layout.addWidget(dir_label)
        central.addLayout(dir_layout)

        broken_label = QLabel("Zerwane połączenia")
        self.broken_checkbox = QCheckBox()
        broken_layout = QHBoxLayout(self)
        broken_layout.addWidget(self.broken_checkbox)
        broken_layout.addWidget(broken_label)
        central.addLayout(broken_layout)

        self.btn = QPushButton("Start")
        self.btn.clicked.connect(self.search_btn_clicked)
        central.addWidget(self.btn)


        abort_btn = QPushButton("Przerwij")
        abort_btn.clicked.connect(self.abort_search)
        central.addWidget(abort_btn)

        self.setLayout(layout)

        self.functions = [self.dfs_plot, self.bfs_plot]

    def abort_search(self):
        self.abort = True

    def plot_results(self, answer, target_label):
        x = [i[0] for i in answer]
        y = [i[1] for i in answer]
        z = [i[2] for i in answer]

        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(x, y, z, c='red', s=100)
        ax.plot(x, y, z, color='black')
        filename = f'{threading.get_ident()}.png'
        plt.savefig(filename)
        pixmap = QPixmap(filename)
        target_label.setPixmap(pixmap)
        target_label.resize(pixmap.width(), pixmap.height())
        plt.clf()
        plt.cla()
        os.remove(filename)

    def search_btn_clicked(self):
        try:
            size = int(self.sizeBox.text())
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setMinimumSize(600, 600)
            msg.setText("Rozmiar nie jest liczbą")
            msg.setWindowTitle("Błąd konwertowania")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        calculate_basics(size, self.directional_checkbox.isChecked(), self.broken_checkbox.isChecked())

        left_plot = threading.Thread(target=self.functions[self.left_func_choice.currentIndex()],
                                     args=[self.left_plot_view, self.left_progress_bar, self.left_output_label])
        right_plot = threading.Thread(target=self.functions[self.right_func_choice.currentIndex()],
                                      args=[self.right_plot_view, self.right_progress_bar, self.right_output_label])
        left_plot.start()
        left_plot.join()
        right_plot.start()
        right_plot.join()

    def dfs_plot(self, target_pixmap, target_progress_bar, target_output_label: QLabel):
        self.btn.setEnabled(False)
        global cities, SIZE
        start = time.time()
        calculate_costs(distances)

        depth_leaves_full_asym: List[Node] = []

        root = Node([0], 0)
        self.depth_generate_tree_leaves(root, depth_leaves_full_asym, target_progress_bar)
        shortest = root
        if len(depth_leaves_full_asym) > 0:
            shortest = depth_leaves_full_asym[0]
            for leaf in depth_leaves_full_asym:
                if leaf.cost < shortest.cost:
                    shortest = leaf

            self.plot_results([cities[i] for i in shortest.path], target_pixmap)
        _time = round((time.time() - start), 5)
        target_output_label.setText(f'Algorytm DFS znalazl rozwiazanie {shortest.path} z kosztem {shortest.cost}, w czasie {_time}')
        self.abort = False
        self.btn.setEnabled(True)

    def depth_generate_tree_leaves(self, node: Node, leaves: List[Node], progress_bar):
        if not directional and len(leaves) >= total:
            return
        if self.abort:
            return

        if len(node.remaining) == 0:
            if self.abort:
                return
            node.path.append(node.path[0])
            node.cost += distances[node.path[-2], node.path[-1]]
            leaves.append(node)
            progress_bar.setValue(int(len(leaves) / total * 100))
            return

        while len(node.remaining) > 0:
            if self.abort:
                return
            new_path = node.path.copy()
            new_path.append(node.remaining[0])
            child = Node(new_path, node.cost)
            node.remaining.pop(0)
            child.cost += distances[child.path[-2], child.path[-1]]
            self.depth_generate_tree_leaves(child, leaves, progress_bar)

    def bfs_plot(self, target_pixmap, target_progress_bar, target_output_label: QLabel):
        self.btn.setEnabled(False)
        global cities, SIZE
        start = time.time()
        calculate_costs(distances)

        breadth_leaves_full_asym: List[Node] = []

        root = Node([0], 0)
        self.breadth_generate_tree_leaves([root], breadth_leaves_full_asym, target_progress_bar)
        shortest = root
        if len(breadth_leaves_full_asym) > 0:
            shortest = breadth_leaves_full_asym[0]
            for leaf in breadth_leaves_full_asym:
                if leaf.cost < shortest.cost:
                    shortest = leaf

            self.plot_results([cities[i] for i in shortest.path], target_pixmap)

        _time = round((time.time() - start), 5)
        target_output_label.setText(f'Algorytm BFS znalazl rozwiazanie {shortest.path} z kosztem {shortest.cost}, w czasie {_time}')
        self.abort = False
        self.btn.setEnabled(True)

    def breadth_generate_tree_leaves(self, nodes: List[Node], leaves: List[Node], progress_bar):
        if not directional and len(leaves) >= total:
            return

        next_level_nodes = []
        for node in nodes:
            if len(node.remaining) == 0:
                node.path.append(node.path[0])
                node.cost += distances[node.path[-2], node.path[-1]]
                leaves.append(node)

            for i in node.remaining:
                new_path = node.path.copy()
                new_path.append(i)
                child = Node(new_path, node.cost)
                child.cost += distances[child.path[-2], child.path[-1]]
                next_level_nodes.append(child)
                progress_bar.setValue(int(max(len(next_level_nodes), len(nodes)) / total * 100))
                if len(next_level_nodes) >= total:
                    break
            if len(next_level_nodes) >= total:
                break

        if len(next_level_nodes) == 0:
            return
        self.breadth_generate_tree_leaves(next_level_nodes, leaves, progress_bar)


def rand_coords():
    return np.array((random.randrange(-100, 100), random.randrange(-100, 100), random.randrange(0, 50)))


def cost_between(a, b, asym):
    cost = np.linalg.norm(a - b)
    if asym and a[2] > b[2]:
        cost *= 0.9
    elif asym and a[2] < b[2]:
        cost *= 1.1
    return cost


def calculate_costs(dsts):
    for from_counter, from_node in enumerate(cities):
        for to_counter, to_node in enumerate(cities):
            if from_counter == to_counter:
                dsts[from_counter][to_counter] = 0
            else:
                dsts[from_counter][to_counter] = np.linalg.norm(from_node - to_node)
                if directional and from_node[2] > to_node[2]:
                    dsts[from_counter][to_counter] *= 0.9
                elif directional and from_node[2] < to_node[2]:
                    dsts[from_counter][to_counter] *= 1.1
            if broken and random.random() > 0.8:
                dsts[from_counter][to_counter] = float("inf")

    return dsts


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    myApp = MainWindow()
    myApp.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')


if __name__ == '__main__':
    main()
