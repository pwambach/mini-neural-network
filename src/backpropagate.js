export default function backpropagate(layers, targetOutputs, learningRate) {
  const outputLayer = layers[layers.length - 1];
  const hiddenLayers = layers.slice(0, layers.length - 1);

  // calc output gradients
  outputLayer.forEach((neuron, index) => {
    const target = targetOutputs[index];
    const output = neuron.getOutput();
    const error = target - output;
    const derivate = (1 - output) * output; // sigmoid derivate

    neuron.gradient = derivate * error;
  });

  // calc hidden gradients
  hiddenLayers.forEach((hiddenLayer, index) => {
    const nextLayer = hiddenLayers[index + 1] || outputLayer;

    hiddenLayer.forEach((neuron, index) => {
      const output = neuron.getOutput();
      const derivate = (1 - output) * output;

      const sum = nextLayer
        .map(nextNeuron => nextNeuron.gradient * nextNeuron.weights[index])
        .reduce((total, value) => total += value, 0);

      neuron.gradient = derivate * sum;
    });
  });

  // update weights
  [...layers].reverse().forEach(layer => {
    layer.forEach(neuron => {
      // console.log(neuron.gradient);
      neuron.weights.forEach((weight, index) => {
        const inputValue = neuron.inputs ?
          neuron.inputs[index].getOutput() :
          neuron.inputValues[index];

        const delta = learningRate * neuron.gradient * inputValue;
        const newWeight = weight + delta;

        neuron.weights[index] = newWeight;
      });
    });
  });
}
