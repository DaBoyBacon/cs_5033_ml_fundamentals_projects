# import parser
# import neuron
import random

# Helper function: Coordinate the updating of weights
def update_weights(ann: list, point: list, loss_rate):
    curr_loss = 0
    for i in range(len(ann) - 1, -1, -1):
        if i == len(ann) - 1:
            loss_list, ann[i] = update_weight_output(ann[i], point, loss_rate)
        else:
            ann[i] = update_weight_hidden(loss_list.pop(), ann[i], point, loss_rate)

# Helper function: Update weights of end neuron
def update_weight_output(neuron, point, loss_rate):
    weights = neuron.getWeights()
    output = neuron.runNode(point)
    loss_list = list()
    for i in range(len(point) - 1): #for all but the last value of point:
        loss = loss_rate * (point[len(point) - 1] - output) * point[i]
        loss_list.append(loss)
    neuron.updateWeights(loss_list)
    return loss_list, neuron

# Helper function: Update weights of hidden neurons (assumes one layer)
def update_weight_hidden(loss_val, neuron, point, loss_rate):
    loss_list = list()
    for i in range(len(point) - 1):
        loss = loss_rate * point[i] * loss_val
        loss_list.append(loss)
    neuron.updateWeights(loss_list)
    return neuron


def trainANN(dimensions: int, csv_name: str, output_location: str, loss_rate = 0.4):
    training, validation, test = parser.parse(csv_name)
    print("Parser completed: ", csv_name)
    
    ann = list()
    for i in range(dimensions):
        curr_node = list()
        for i in range(dimensions + 1):
            curr_node.append(random.uniform(-1, 1))
        if i < dimensions - 1:
            ann.addNeuron(neuron(curr_node), True)
        else:
            ann.addNeuron(neuron(curr_node), False)

    
    eval = 0    
    # ACTUALLY TRAIN! Evaluate after every run
    while eval <= 0.8:
        random.shuffle(training)
        curr_point = training.pop(0)
        update_weights(ann, curr_point, loss_rate)
        training.append(curr_point)
        eval = 0

        for point in validation:
            if ann.run(point) == point[len(point) - 1]:
                eval += 1
        eval = eval / len(validation)

    

                
    # Get confidence values
    confidence_score = 0
    for point in test:
        if ann.run(point) == point[len(point) - 1]:
            confidence_score += 1
        
    confidence_score = confidence_score / len(test) 

    print("Confidence in network: ", confidence_score)
    ann.outputWeights(csv_name)
    return confidence_score


    
        