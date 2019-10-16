import random

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, item):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, size):
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)