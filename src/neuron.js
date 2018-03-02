export default class Neuron {
  constructor(inputs = null) {
    this.inputs = inputs;
    this.activationFn = x => 1 / (1 + Math.exp(x * -1)); // sigmoid function

    // set random weights - except for the input layer because we don't know the
    // number of inputs for this one yet
    this.weights = inputs ?
      getRandomWeights(inputs.length) :
      null;
    this.output = null;
    this.gradient = null;
  }

  calcOutput(customInputValues = null) {
    const inputValues = customInputValues ?
      customInputValues :
      this.inputs.map(neuron => neuron.getOutput());

    if (customInputValues) {
      this.inputValues = customInputValues;
    }

    // use defiend weights (or generate random ones now for input layer)
    this.weights = this.weights || getRandomWeights(inputValues.length);

    const sum = dotProduct(this.weights, inputValues);
    this.output = this.activationFn(sum);
  }

  getOutput() {
    return this.output;
  }
}

function dotProduct(a, b) {
  if (a.length !== b.length) {
    throw('Not same length!');
  }

  let sum = 0;

  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }

  return sum;
}

function getRandomWeights(count) {
  return Array.from({length: count}).map(() => Math.random() - 0.5);
}
