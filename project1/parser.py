import csv
import random


def is_moon(csv_file: str):
    if "Moons" in csv_file:
        return True
    return False

class Parser:
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.25
    # testing gets whatever is left (5%)

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.csv_file = None
        self.data = []  # raw csv rows
        self.points = []  # every point as [x1, x2, x3]
        self.training = []
        self.validation = []
        self.testing = []
        self.dimension = None
        self.moons = False

    def parse(self, csv_file: str):
        """
        load the given csv. returns (training, validation, testing)
        """
        self.csv_file = csv_file
        self._load()
        self._extract()
        self._split()
        return self.training, self.validation, self.testing

    def _load(self):
        rows = []
        with open(self.csv_file, newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                rows.append([float(c) for c in row])
        self.data = rows

        self.moons = is_moon(self.csv_file)
        self.dimension = len(rows[0]) // 2
    
    def _extract(self):
        d = self.dimension
        self.points = []
        for row in self.data:
            self.points.append(row[:d]    + [0.0])   # class 0
            self.points.append(row[d:2*d] + [1.0])   # class 1

    def _split(self):
        points = self.points[:]
        random.Random(self.seed).shuffle(points)

        n = len(points)
        n_train = round(n * self.TRAIN_SPLIT)
        n_val = round(n * self.VAL_SPLIT)

        self.training = points[:n_train]
        self.validation = points[n_train:n_train + n_val]
        self.testing = points[n_train + n_val:]
