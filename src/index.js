import Neuron from './neuron';
import loadMNIST from './mnist-helper';
import {isCorrect, getError, getIndexWithHighestValue} from './helpers';
import backpropagate from './backpropagate';
import renderNumber from './render-number';

const TRAINING_SAMPLES = 10000; // max 60000
const TEST_SAMPLES = 5000; // max 10000
const LEARNING_RATE = 0.15;
let correct = 0;
const trainButton = document.querySelector('#train');
const testButton = document.querySelector('#test');
const singleTestButton = document.querySelector('#singleTest');
const canvas = document.querySelector('#canvas');
const result = document.querySelector('.singleTest .result');

const layers = [];
let trainingData = null;
let testData = null;

async function init() {
  console.log("Loading MNIST data...");
  trainingData = await loadMNIST('train', 0, TRAINING_SAMPLES);
  testData = await loadMNIST('test', 0, TEST_SAMPLES);
  console.log("Finished loading data");

  // init network with 2 layers

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
  singleTestButton.addEventListener('click', () => singleTest());
}

init();

function computeOutput(layers, sample) {
  // get data from sample
  const {pixels} = sample;
  const normalizedPixels = pixels.map(x => x / 255);

  // run input through layers
  layers[0].forEach(neuron => neuron.calcOutput(normalizedPixels));
  layers[1].forEach(neuron => neuron.calcOutput());

  return layers[1].map(neuron => neuron.getOutput());
}

// train the network with the training samples
async function train() {
  console.log(`Start Training with Learning Rate: ${LEARNING_RATE} and ${TRAINING_SAMPLES} samples`);

  
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

// performs a single visible test on screen to actually see something :)
function singleTest() {
  const randomIndex = Math.floor(Math.random() * TEST_SAMPLES);
  const sample = testData[randomIndex];
  const outputs = computeOutput(layers, sample);
  const prediction = getIndexWithHighestValue(outputs);
  result.innerHTML = prediction;

  renderNumber(canvas, sample.pixels, 28, 28);
}
