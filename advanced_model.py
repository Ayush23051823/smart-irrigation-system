import numpy as np

class SimpleKalman:
    def __init__(self):
        self.x = 0.5   # initial soil moisture
        self.P = 1     # uncertainty
        self.Q = 0.01  # process noise
        self.R = 0.05  # measurement noise

    def predict(self):
        # prediction step
        self.P = self.P + self.Q

    def update(self, measurement):
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P
        return self.x