# PyRNN
LSTM-RNN implementation.
For more details about usage check examples folder.

## Usage
1.  Create own class with superclass ProblemObject and inherit its methods
2.  Create RNN using 

  rnn = RNN(input_dimension, hidden_dimension, output_dimension, alpha = 0.1)
3.  Learn RNN with dataset. Learining is supervized. X is list of problemObjects

  rnn.learn(X)

4. Use learned RNN. To predict values of problemObject with unknown result.
  
  rnn.predict(problem)


### TODO
* Visualisation of convergence
* Bad input checking
* Add more examples
* Parallelisation on Spark

### Acknowledgement
I want to thank Andrew Trask for great tutorial which I used during implementation. [RNN tutorial](https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/)
