import neuron

class Network:

    # Assumes one hidden layer with one binary output
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.hidden_layers = list()
        self.output_layer = False

    def add_neuron(self, weights, isHidden: bool):
        if isHidden:
            if len(self.hidden_layers) == self.dimensions:
                print("Layers already full")
                return
            self.hidden_layers.append(neuron(weights, isHidden))
        elif not self.output_layer:
                self.output_layer = neuron(weights, isHidden)
        else:
            print("output layer already full")
            return

    def run(self, point):
        if len(point) != self.dimensions + 1:
            print("Point is not of correct size")
            return -1
        else:
            y = point[len(point) - 1]
            out_layer_point = list()
            for neuron in self.hidden_layers:
                out_layer_point.append(neuron.run_neuron(point))
            out_layer_point.append(y)
            output = self.output_layer.run_neuron(out_layer_point)

            return output



        

        

