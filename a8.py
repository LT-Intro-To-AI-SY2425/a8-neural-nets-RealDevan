from neural import NeuralNet  
from typing import Tuple, List  

def parse_line(line: str) -> Tuple[List[float], List[float]]:
    tokens = line.strip().split(",")  
    
    out = tokens[-1]
    
    if out == "Iris-setosa":
        output = [1.0, 0.0, 0.0]
    elif out == "Iris-versicolor":
        output = [0.0, 1.0, 0.0]
    elif out == "Iris-virginica":
        output = [0.0, 0.0, 1.0]
    else:
        raise ValueError(f"Unexpected class label: {out}")

    inpt = [float(x) for x in tokens[:-1]]
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    leasts = len(data[0][0]) * [float("inf")] 
    mosts = len(data[0][0]) * [float("-inf")]  

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            leasts[j] = min(leasts[j], data[i][0][j])
            mosts[j] = max(mosts[j], data[i][0][j])

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            
            if mosts[j] - leasts[j] != 0:
                data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
            else:
                data[i][0][j] = 0.0
    return data

with open("iris_data.csv", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line.strip()) > 0]

td = normalize(training_data)


input_size = len(td[0][0])
output_size = len(td[0][1])
nn = NeuralNet(input_size, 3, output_size) 


nn.train(td, iters=1_000, print_interval=100, learning_rate=0.1)


for i in nn.test_with_expected(td):
    print(f"desired: {i[1]}, actual: {i[2]}")