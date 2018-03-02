import Neuron from './neuron';
import loadMNIST from './mnist-helper';
import {isCorrect, getError, getAnswer} from './helpers';
import backpropagate from './backpropagate';

const TRAINING_SAMPLES = 60000; // max 60000
const TEST_SAMPLES = 5000; // max 10000
const LEARNING_RATE = 0.15;
let correct = 0;
const trainButton = document.querySelector('button#train');
const testButton = document.querySelector('button#test');

// init network with 2 layers
const layers = [];

// hidden layer
layers.push(Array
  .from({length: 50})
  .map(() => new Neuron())
);

// output layer
layers.push(Array
  .from({length: 10})
  .map(() => new Neuron(layers[0]))
);

// hook up buttons
trainButton.addEventListener('click', () => train());
testButton.addEventListener('click', () => test());

function computeOutput(layers, sample) {
  // get data from sample
  const {pixels, number} = sample;
  const normalizedPixels = pixels.map(x => x / 255);

  // run input through layers
  layers[0].forEach(neuron => neuron.calcOutput(normalizedPixels));
  layers[1].forEach(neuron => neuron.calcOutput());

  return layers[1].map(neuron => neuron.getOutput());
}

// train the network with the training samples
async function train() {
  console.log(`Start Training with Learning Rate: ${LEARNING_RATE} and ${TRAINING_SAMPLES} samples`);

  const trainingData = await loadMNIST('train', 0, TRAINING_SAMPLES);
  correct = 0;

  for (let i = 0; i < trainingData.length; i++) {
    const sample = trainingData[i];
    const {targets} = sample;
    const outputs = computeOutput(layers, sample);

    // log error rate
    if (i % 1000 === 0) {
      console.log(`Sample ${i} - Loss: ${getError(outputs, targets)}`);
    }

    // count correct predictions
    if (isCorrect(outputs, targets)) {
      correct++;
    }

    // update weights
    backpropagate(layers, targets, LEARNING_RATE);
  }

  console.log(`${correct} out of ${TRAINING_SAMPLES} correct during training (${(correct / TRAINING_SAMPLES * 100).toFixed(1)}%)`);
}

// test the network with the test samples
async function test() {
  console.log(`Start Test with ${TEST_SAMPLES} samples`);

  const testData = await loadMNIST('test', 0, TEST_SAMPLES);
  correct = 0;

  for (let i = 0; i < testData.length; i++) {
    const sample = testData[i];
    const {targets} = sample;
    const outputs = computeOutput(layers, sample);

    // count correct predictions
    if (isCorrect(outputs, targets)) {
      correct++;
    }
  }

  console.log(`${correct} out of ${TEST_SAMPLES} correct (${(correct / TEST_SAMPLES * 100).toFixed(1)}%)`);
}
