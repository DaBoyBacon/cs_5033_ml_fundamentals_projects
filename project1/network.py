from neuron import Neuron
import csv

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
            self.hidden_layers.append(Neuron(weights, isHidden))
        elif not self.output_layer:
                self.output_layer = Neuron(weights, isHidden)
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

    def getNeurons(self):
        output = list()
        for neuron in self.hidden_layers:
            output.append(neuron)
        output.append(self.output_layer)
        return output

    def resetNeurons(self, neuron_list):
        self.output_layer = neuron_list.pop()
        self.hidden_layers = list()
        for n in neuron_list:
            self.hidden_layers.append(n)
        return



    def outputWeights(self, csv_name):
        out_location = csv_name[:-4]
        out_location += "_weights.csv"
        

        with open(out_location, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for n in self.hidden_layers:
                writer.writerow(n.get_weights())
            writer.writerow(self.output_layer.get_weights())

        print(self.hidden_layers)

        return


        

        

