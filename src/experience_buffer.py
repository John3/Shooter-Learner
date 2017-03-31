import numpy as np
import random
import parameter_config as cfg


class ExperienceBuffer():
    def __init__(self, buffer_size=-1):
        self.buffer = []
        if buffer_size < 0:
            self.buffer_size = cfg.buffer_size
        else:
            self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(len(self.buffer) + 1) - self.buffer_size] = [] #todo review this piece
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0,len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces,[batch_size*trace_length,5])

    def save(self, path):
        np.savez("%s/experience_buffer.npz" % path, buffer=self.buffer, buffer_size=self.buffer_size)

    def load(self, path):
        file = np.load("%s/experience_buffer.npz" % path)
        self.buffer = list(file["buffer"])
        self.buffer_size = file["buffer_size"]

