/**
 * Returns true if the prediction matches the target
 * @param {Array.<number>} outputs The predicted output values
 * @param {Array.<number>} targets The actual target values
 * @returns {Boolean} true if the prediction matches the target
 */
export function isCorrect(outputs, targets) {
  return getIndexWithHighestValue(outputs) === getIndexWithHighestValue(targets);
}

// -> returns 3 if the network predicts a 3
function getIndexWithHighestValue(outputs) {
  let highestValue = 0;
  let index = 0;

  for (let i = 0; i < outputs.length; i++) {
    const value = outputs[i];
    if (value > highestValue) {
      highestValue = value;
      index = i;
    }
  }

  return index;
}

/**
 * Returns the loss from a prediction.
 * (for all 10 outputs: target - output and then Math.sqrt)
 * @param {Array.<number>} outputs The predicted output values
 * @param {Array.<number>} targets The actual target values
 * @returns {number} The loss value
 */
export function getError(outputs, targets) {
  let sum = 0;

  outputs.forEach((output, index) => {
    const difference = targets[index] - output;
    sum += Math.sqrt(difference * difference);
  });

  return sum;
}
