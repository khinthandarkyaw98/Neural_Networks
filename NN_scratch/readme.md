# Diclaimers # 
Pictures used in this folder belongs to [sentdex](https://www.youtube.com/@sentdex).

### Why do we have only one bias?
<img width="200" alt="a single neurong with 3 inputs" src="Image/1.png">
<img width="600" alt="a single neuron with 3 inputs explanation" src="Image/2.png">
<p align="justify">So, it is intuitive that a single neuron must have only a single bias regardless of many inputs.<br> If there are 4 inputs, there must be 4 weights.</p>
<p>
For nn_2.py, 3 Neurons(outputs) with 4 inputs are as follows.</p>
<img width="200" alt="3 Neurons with 4 inputs" src="Image/3.png">
<p>
For the above picture, each neuron has its own weight set and unique bias as shown below.</p>
<img width="200" alt="3 Neurons with 4 inputs and one bias for each neuron" src="Image/4.png">
<img width="500" alt="3 Neurons with 4 inputs explanation" src="Image/5.png">

#### **Important!**

<p align="justify">The inputs cannot be altered.<br>For this reason, weights and biases must be tweaked to get the desired output.</p>

So, the weight is trying to change the magnitude of the input and the bias is offsetting it. This is not different form $y = mx + c$ (line_equation). So basically, even if weights or inputs are in negative values, the bias will offset those to retrieve positive values. Intutively, you can figure this out.

### **Let's talk about Shapes!**

<img width="500" alt="1D array, vector" src="Image/6.png">

<p align="center">A list in python = 1D array in numpy = Vector in Mathematics</p>

<img width="500" alt="2D array, Matrix" src="Image/7.png">

<p align="center">A list of list in python = 2D array in numpy = Matrix in Mathematics</p>

In other words, a list of vectors is a matrix.

<img width="500" alt="3D array, a list of list of lists" src="Image/8.png">

<p align="center">A list of lists of lists = 3D array</p>

#### **Tensor**

<p align="justify">A tensor is an object which can be represented as an array.</p>

### **Dot Product**

<img width="500" alt="Dot Product" src="Image/9.png">

<p align="justify">As seen above, the dot product of two vectors gives out the scalar value.
</p>

### **Batches**

<img width="500" alt="Batches" src="Image/10.png">

<p align="justify">Each batch means each row of variable "batch".</p>

<img width="500" alt="batch size" src="Image/23.png">

<p align="justify">Increasing the size of the batch is great. However, we need to keep in mind to consider the possibility of overfitting. Batch size should be '32' or '64' or '128'.</p>

### **Dot Product with Batches**

<img width="500" alt="dot_product" src="Image/24.png">

<img width="500" alt="dot_product" src="Image/25.png">

<img width="500" alt="dot_product" src="Image/26.png">

<img width="500" alt="dot_product" src="Image/27.png">

### **Transpose of a matrix**

<img width="500" alt="Matrix Transpose" src="Image/28.png">

### **Why do we need to transpose in dot product?**

<img width="500" alt="Matrix Transpose" src="Image/29.png">

<img width="500" alt="Matrix Transpose" src="Image/30.png">

<img width="500" alt="Matrix Transpose" src="Image/31.png">

<img width="500" alt="Matrix Transpose" src="Image/32.png">

<img width="500" alt="Matrix Transpose" src="Image/33.png">

### **Neural Network**

<img width="500" alt="NN" src="Image/0.png">

<p align="justify">This is a neural network in which each nueron has its own inputs along with weights and a unique bias.</p>

<img width="500" alt="Input-Output-NN" src="Image/11.png">

<img width="500" alt="I/O RS" src="Image/12.png">

<img width="500" alt="Comparison with line eq to NN" src="Image/13.png">

<p align="justify">The formula for the output of NN is much similar to the line equation we already know.</p>

<img width="500" alt="y-x" src="Image/14.png">

<img width="500" alt="y-x" src="Image/15.png">

<p align="justify">As you can see in above 2 figures, changing the values of weight also cause changes in output.</p>

<img width="500" alt="bias" src="Image/16.png">

<img width="500" alt="bias" src="Image/17.png">

<p align="justify">Similarly, changing the values of bias also cause changes(shifts) in output.</p>

<img width="500" alt="bias" src="Image/18.png">

<p align="justify">So, we have to change the values of not only weights but also biases.</p>

<img width="500" alt="bias" src="Image/19.png">

### **Why do we need ACTIVATION function?**

<p align="justify">Activation function is either to amplify or repress the local signal recevied from the addtition of the multiplication of inputs and weights to the bias to produce output. If activation makes us more grandualrity when we do backward propagate to calculate loss. There are lots of activation functions.<br> The most common ones are as follows.<ol><li>step function</li><li>sigmoid function</li><li>RELU(rectified linear unit) fucntion</li></ol><br>However, a neural network with the step function as an activation function cannot be found. Let's say that we want to fit the non-linear data. Only non-linear activation functions can mostly fit the given non-linear data. RELU is recommended to choose over sigmoid as the latter has the gradient vanishing problem.</p>

#### **Activation Functions**

<div align="center">
    <p align="center">Step Functions</p>
    <img align="center" width=20.5% height=100px alt="step" src="Image/step.png">
    <img align="center" width=20.5% height=100px alt="step_impact.png" src="Image/step_impact.png">
</div>

<div align="center">
    <p align="center">Sigmoid Functions</p>
    <img align="center" width=20.5% height=100px alt="sigmoid" src="Image/sigmoid.png">
    <img align="center" width=20.5% height=100px alt="sigmoid_impact.png" src="Image/sigmoid_impact.png">
</div>

<div align="center">
    <p align="center">RELU(rectified linear unit) Functions</p>
    <img align="center" width=20.5% height=100px alt="relu" src="Image/RELU.png">
    <img align="center" width=20.5% height=100px alt="RELU_impact.png" src="Image/RELU_impact.png">
</div>

### **Why do we need more than ONE hidden layers?**
