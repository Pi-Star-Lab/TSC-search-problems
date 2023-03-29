from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from models.conv_net import ConvNet, TwoHeadedConvNet, HeuristicConvNet
from models.ff_net import FeedForwardNet, TwoHeadedFeedForwardNet, HeuristicFeedForwardNet

class KerasModel():
    def __init__(self, conv = True, num_actions = 4):
        self.mutex = Lock()
        self.model = None
        self.conv = conv
        self.num_actions = num_actions

    def initialize(self, loss_name, search_algorithm, two_headed_model=False):
        if self.conv:
            self._init_conv_backend(loss_name, search_algorithm, two_headed_model)
        else:
            self._init_ffnn_backend(loss_name, search_algorithm, two_headed_model)

    def _init_ffnn_backend(self, loss_name, search_algorithm, two_headed_model=False):

        if (search_algorithm == 'Levin'
            or search_algorithm == 'LevinMult'
            or search_algorithm == 'LevinStar'
            or search_algorithm == 'PUCT'):
            if two_headed_model:
                self.model = TwoHeadedFeedForwardNet(1024, self.num_actions, loss_name)
            else:
                self.model = FeedForwardNet(1024, self.num_actions, loss_name)
        if search_algorithm == 'AStar' or search_algorithm == 'GBFS':
                self.model = HeuristicFeedForwardNet(1024, self.num_actions)

    def _init_conv_backend(self, loss_name, search_algorithm, two_headed_model=False):

        if (search_algorithm == 'Levin'
            or search_algorithm == 'LevinMult'
            or search_algorithm == 'LevinStar'
            or search_algorithm == 'PUCT'):
            if two_headed_model:
                self.model = TwoHeadedConvNet((2, 2), 32, self.num_actions, loss_name)
            else:
                self.model = ConvNet((2, 2), 32, self.num_actions, loss_name)
        if search_algorithm == 'AStar' or search_algorithm == 'GBFS':
                self.model = HeuristicConvNet((2, 2), 32, self.num_actions)


    def predict(self, x):
        with self.mutex:
            return self.model.predict(x)

    def train_with_memory(self, memory):
        return self.model.train_with_memory(memory)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath).expect_partial()

class KerasManager(BaseManager):
    pass
