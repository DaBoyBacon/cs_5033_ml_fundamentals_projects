from parser import Parser
import random
from network import Network

# Helper function: Coordinate the updating of weights
def update_weights(ann: Network, point: list, loss_rate):
    curr_loss = 0
    annList = ann.getNeurons()
    res = ann.run(point)
    for i in range(len(annList) - 1, -1, -1):
        if i == len(annList) - 1:
            loss_list, annList[i] = update_weight_output(annList[i], point, loss_rate, res)
        else:
            annList[i] = update_weight_hidden(loss_list.pop(), annList[i], point, loss_rate)

    ann.resetNeurons(annList)

# Helper function: Update weights of end neuron
def update_weight_output(neuron, point, loss_rate, output):
    loss_list = list()
    for i in range(len(point)): #for all but the last value of point:
        if i > 0:
            loss = loss_rate * (point[len(point) - 1] - output) * point[i - 1]
            loss_list.append(loss)
        else:
            loss = loss_rate * (point[len(point) - 1] - output)
            loss_list.append(loss)

    neuron.set_weights(loss_list)
    return loss_list, neuron

# Helper function: Update weights of hidden neurons (assumes one layer)
def update_weight_hidden(loss_val, neuron, point, loss_rate):
    loss_list = list()
    for i in range(len(point)):
        if i > 0:
            loss = loss_rate * point[i-1] * loss_val
            loss_list.append(loss)
        else:
            loss = loss_rate * loss_val
            loss_list.append(loss)
    neuron.set_weights(loss_list)
    return neuron


def trainANN(dimensions: int, csv_name: str, loss_rate = 0.001):
    p = Parser()
    training, validation, test = p.parse(csv_name)
    print("Parser completed: ", csv_name)
    
    ann = Network(dimensions)
    for i in range(dimensions + 1):
        curr_node = list()
        for j in range(dimensions + 1):
            curr_node.append(random.uniform(-1, 1))
        if i < dimensions:
            ann.add_neuron(curr_node, True)
        else:
            ann.add_neuron(curr_node, False)

    
    eval = 0    
    # ACTUALLY TRAIN! Evaluate after every run
    epochs = 0
    while eval <= 0.85 and epochs < 150:
        random.shuffle(training)
        curr_point = training.pop(0)
        update_weights(ann, curr_point, loss_rate)
        training.append(curr_point)
        eval = 0

        for point in validation:
            if ann.run(point) == point[len(point) - 1]:
                eval += 1
        eval = eval / len(validation)
        epochs += 1 / len(training)
        print(epochs)
    print("Ran over ", epochs, " epochs")

    

                
    # Get confidence values
    confidence_score = 0
    for point in test:
        if ann.run(point) == point[len(point) - 1]:
            confidence_score += 1
        
    confidence_score = confidence_score / len(test) 

    print("Confidence in network: ", confidence_score)
    ann.outputWeights(csv_name)
    return confidence_score




    
trainANN(2, "project1\\Data\\Moons 2D Wide.csv")