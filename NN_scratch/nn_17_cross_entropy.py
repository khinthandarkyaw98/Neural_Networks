import math

softmax_output = [0.7, 0.1, 0.2]
# index 1 of target_output is hot
target_output = [1, 0, 0]

loss = -(target_output[0] * (math.log(softmax_output[0]))+
         target_output[1] * (math.log(softmax_output[1]))+
         target_output[2] * (math.log(softmax_output[2])))

print(loss)