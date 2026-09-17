// neuralnet.js
// A minimal feedforward neural network with backpropagation, built from scratch.
// Architecture: 2 inputs -> hidden layer (tanh) -> 1 output (sigmoid)

class NeuralNetwork {
  constructor(inputSize = 2, hiddenSize = 8, outputSize = 1, learningRate = 0.05) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.learningRate = learningRate;

    // Weights initialized small and random (common practice: keeps early
    // activations from saturating tanh/sigmoid)
    this.W1 = this.randomMatrix(inputSize, hiddenSize);
    this.b1 = new Array(hiddenSize).fill(0);

    this.W2 = this.randomMatrix(hiddenSize, outputSize);
    this.b2 = new Array(outputSize).fill(0);
  }

  randomMatrix(rows, cols) {
    const m = [];
    for (let i = 0; i < rows; i++) {
      const row = [];
      for (let j = 0; j < cols; j++) {
        row.push((Math.random() * 2 - 1) * 0.5); // range: -0.5 to 0.5
      }
      m.push(row);
    }
    return m;
  }

  tanh(x) { return Math.tanh(x); }
  tanhDerivative(y) { return 1 - y * y; } // y = tanh(x) already computed

  sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  sigmoidDerivative(y) { return y * (1 - y); } // y = sigmoid(x) already computed

  // Forward pass: input -> hidden -> output
  // Returns everything needed for backprop (activations at each layer)
  forward(input) {
    const hidden = new Array(this.hiddenSize);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += input[i] * this.W1[i][j];
      }
      hidden[j] = this.tanh(sum);
    }

    const output = new Array(this.outputSize);
    for (let k = 0; k < this.outputSize; k++) {
      let sum = this.b2[k];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hidden[j] * this.W2[j][k];
      }
      output[k] = this.sigmoid(sum);
    }

    return { input, hidden, output };
  }

  // Trains on ONE example (x, y, label) using backpropagation.
  // This is standard stochastic gradient descent, one point at a time.
  trainOne(input, target) {
    const { hidden, output } = this.forward(input);

    // ---- Output layer error ----
    // dL/dOutput using binary cross-entropy-style gradient simplification:
    // for sigmoid + this loss, the gradient simplifies to (output - target)
    const outputError = new Array(this.outputSize);
    for (let k = 0; k < this.outputSize; k++) {
      outputError[k] = output[k] - target[k];
    }

    // ---- Hidden layer error (backpropagate through W2) ----
    const hiddenError = new Array(this.hiddenSize).fill(0);
    for (let j = 0; j < this.hiddenSize; j++) {
      let errSum = 0;
      for (let k = 0; k < this.outputSize; k++) {
        errSum += outputError[k] * this.W2[j][k];
      }
      hiddenError[j] = errSum * this.tanhDerivative(hidden[j]);
    }

    // ---- Update W2, b2 (gradient descent step) ----
    for (let j = 0; j < this.hiddenSize; j++) {
      for (let k = 0; k < this.outputSize; k++) {
        this.W2[j][k] -= this.learningRate * outputError[k] * hidden[j];
      }
    }
    for (let k = 0; k < this.outputSize; k++) {
      this.b2[k] -= this.learningRate * outputError[k];
    }

    // ---- Update W1, b1 ----
    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.W1[i][j] -= this.learningRate * hiddenError[j] * input[i];
      }
    }
    for (let j = 0; j < this.hiddenSize; j++) {
      this.b1[j] -= this.learningRate * hiddenError[j];
    }

    // Return the loss for this example (binary cross-entropy)
    const eps = 1e-8; // avoid log(0)
    const y = target[0], p = output[0];
    return -(y * Math.log(p + eps) + (1 - y) * Math.log(1 - p + eps));
  }

  // Trains one full pass over the dataset (one epoch), returns average loss
  trainEpoch(dataset) {
    let totalLoss = 0;
    for (const point of dataset) {
      const input = [point.x, point.y];
      const target = [point.label];
      totalLoss += this.trainOne(input, target);
    }
    return totalLoss / dataset.length;
  }

  predict(input) {
    return this.forward(input).output[0]; // probability of class 1
  }
}