import math

def sigmoid(net: float):
    if net >= 0:
        return 1.0 / (1.0 + math.exp(-net))
    z = math.exp(net)
    return z / (1.0 + z)


class Neuron:
    def __init__(self, weights: list, hidden: bool):
        self.weights = list(weights)
        self.hidden = hidden
        self.net = 0.0 # weighted sum 
        self.output = 0.0 # last output


    def get_weights(self):
        return self.weights[:]

    def set_weights(self, loss: list):
        """
        Update the weights: w_new = w_old + loss
        """
        if len(loss) != len(self.weights):
            raise ValueError("Loss is not the same size as Weights")

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + loss[i]

    def is_hidden(self) -> bool:
        return self.hidden

    def run_neuron(self, inputs):
        """
        Run the AN, if the value is hidden then return the f(net). Not hidden then return the sigmoidal
        """
        bias = self.weights[0]
        weighted_sum = 0.0
        for w, x in zip(self.weights[1:], inputs):
            weighted_sum = weighted_sum + (w * x)

        self.net = bias + weighted_sum

        if self.hidden:
            self.output = sigmoid(self.net)

        else:
            self.output = 1 if sigmoid(self.net) > 0.5 else 0

        return self.output


