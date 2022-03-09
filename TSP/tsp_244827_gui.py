import sys
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
    global SIZE, total, directional, broken, distances
    SIZE = size
    directional = direct
    broken = bro
    total = np.math.factorial(SIZE - 1)
    distances = np.zeros((SIZE, SIZE))
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

        layout = QHBoxLayout(self)
        right = QVBoxLayout(self)

        self.plot_view = QLabel()
        layout.addWidget(self.plot_view)

        layout.addLayout(right)

        self.func_choice = QComboBox()
        self.func_choice.addItems(['DFS', 'BFS'])
        right.addWidget(self.func_choice)

        size_label = QLabel("Ilosc miast")
        self.sizeBox = QLineEdit()
        size_layout = QHBoxLayout(self)
        size_layout.addWidget(self.sizeBox)
        size_layout.addWidget(size_label)
        right.addLayout(size_layout)

        dir_label = QLabel("Kierunkowy")
        self.directional_checkbox = QCheckBox()
        dir_layout = QHBoxLayout(self)
        dir_layout.addWidget(self.directional_checkbox)
        dir_layout.addWidget(dir_label)
        right.addLayout(dir_layout)

        broken_label = QLabel("Zerwane połączenia")
        self.broken_checkbox = QCheckBox()
        broken_layout = QHBoxLayout(self)
        broken_layout.addWidget(self.broken_checkbox)
        broken_layout.addWidget(broken_label)
        right.addLayout(broken_layout)

        self.btn = QPushButton("Start")
        self.btn.clicked.connect(self.search_btn_clicked)
        right.addWidget(self.btn)

        self.progress_bar = QProgressBar(self)
        right.addWidget(self.progress_bar)

        abort_btn = QPushButton("Przerwij")
        abort_btn.clicked.connect(self.abort_search)
        right.addWidget(abort_btn)

        self.setLayout(layout)

        self.functions = [self.dfs_plot, self.bfs_plot]

    def abort_search(self):
        self.abort = True

    def plot_results(self, answer):
        x = [i[0] for i in answer]
        y = [i[1] for i in answer]
        z = [i[2] for i in answer]

        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(x, y, z, c='red', s=100)
        ax.plot(x, y, z, color='black')
        plt.savefig('temp_plot.png')
        pixmap = QPixmap('temp_plot.png')
        self.plot_view.setPixmap(pixmap)
        self.plot_view.resize(pixmap.width(), pixmap.height())
        os.remove('temp_plot.png')

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

        th = threading.Thread(target=self.functions[self.func_choice.currentIndex()])
        th.start()

    def dfs_plot(self):
        self.btn.setEnabled(False)
        global cities, SIZE
        cities = [rand_coords() for i in range(SIZE)]
        calculate_costs(distances)

        depth_leaves_full_asym: List[Node] = []

        root = Node([0], 0)
        self.depth_generate_tree_leaves(root, depth_leaves_full_asym)
        if len(depth_leaves_full_asym) > 0:
            shortest = depth_leaves_full_asym[0]
            for leaf in depth_leaves_full_asym:
                if leaf.cost < shortest.cost:
                    shortest = leaf

            self.plot_results([cities[i] for i in shortest.path])
        self.abort = False
        self.btn.setEnabled(True)

    def depth_generate_tree_leaves(self, node: Node, leaves: List[Node]):
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
            self.progress_bar.setValue(int(len(leaves) / total * 100))
            return

        while len(node.remaining) > 0:
            if self.abort:
                return
            new_path = node.path.copy()
            new_path.append(node.remaining[0])
            child = Node(new_path, node.cost)
            node.remaining.pop(0)
            child.cost += distances[child.path[-2], child.path[-1]]
            self.depth_generate_tree_leaves(child, leaves)

    def bfs_plot(self):
        self.btn.setEnabled(False)
        global cities, SIZE
        cities = [rand_coords() for i in range(SIZE)]
        calculate_costs(distances)

        breadth_leaves_full_asym: List[Node] = []

        root = Node([0], 0)
        self.depth_generate_tree_leaves(root, breadth_leaves_full_asym)
        if len(breadth_leaves_full_asym) > 0:
            shortest = breadth_leaves_full_asym[0]
            for leaf in breadth_leaves_full_asym:
                if leaf.cost < shortest.cost:
                    shortest = leaf

            self.plot_results([cities[i] for i in shortest.path])
        self.abort = False
        self.btn.setEnabled(True)

    def breadth_generate_tree_leaves(self, nodes: List[Node], leaves: List[Node]):
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
                self.progress_bar.setValue(int(len(next_level_nodes) / total * 100))
                if len(next_level_nodes) >= total:
                    break
            if len(next_level_nodes) >= total:
                break

        if len(next_level_nodes) == 0:
            return
        self.breadth_generate_tree_leaves(next_level_nodes, leaves)


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
