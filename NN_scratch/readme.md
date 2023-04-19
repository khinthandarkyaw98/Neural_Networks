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