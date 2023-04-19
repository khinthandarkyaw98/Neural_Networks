This is just for the educational purpose.

# **How does a neural network compute?**

<img width="300" src="Image/1.png">

<ul>
    <li>Step 1 Construct input vector from training data</li>
    <li>Step 2 Muliply weight matrix by column vector</li>
    <li>Apply activation function to resulting column vector</li>
</ul>
<em>Repeat for all layers untl you produce the output</em>

<img width="500" src="Image/2.png">

<p>As you can see, the very first step, delta4 is calculating the error between the predicted value and the ground truth ones.<br>Follwoing this, we will add bias to the previous layer output. <br> Then delta3 is the multiplication of the dot product of current weights_3_transpose and the next delta_4 and the current_sigmoid_gradient_z3.</p>

<img width="500" src="Image/3.png">

<p>As shown above, the current gradient is the dot product of the current input and the next delta.</p>
