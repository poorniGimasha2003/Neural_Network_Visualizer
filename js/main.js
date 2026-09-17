// main.js
// Wires together data.js, neuralnet.js, and visualize.js.
// Handles UI controls: play/pause training, step, reset, learning rate.

let network;
let dataset;
let epoch = 0;
let isTraining = false;
let animationId = null;

function initNetwork() {
  const learningRate = parseFloat(document.getElementById('lr-slider').value);
  network = new NeuralNetwork(2, 8, 1, learningRate);
  dataset = getDataset();
  epoch = 0;
  resetLossChart();
  updateEpochLabel();
  render(network, dataset);
}

function trainStep() {
  const avgLoss = network.trainEpoch(dataset);
  epoch++;
  recordLoss(avgLoss);
  updateEpochLabel();
  render(network, dataset);
}

function updateEpochLabel() {
  document.getElementById('epoch-label').textContent = `Epoch: ${epoch}`;
}

function trainingLoop() {
  if (!isTraining) return;
  trainStep();
  animationId = requestAnimationFrame(trainingLoop);
}

function togglePlay() {
  isTraining = !isTraining;
  const btn = document.getElementById('play-btn');
  btn.textContent = isTraining ? 'Pause' : 'Play';
  if (isTraining) trainingLoop();
  else cancelAnimationFrame(animationId);
}

function stepOnce() {
  if (isTraining) togglePlay(); // pause if running
  trainStep();
}

function resetAll() {
  isTraining = false;
  cancelAnimationFrame(animationId);
  document.getElementById('play-btn').textContent = 'Play';
  initNetwork();
}

// Wire up UI once the page loads
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('play-btn').addEventListener('click', togglePlay);
  document.getElementById('step-btn').addEventListener('click', stepOnce);
  document.getElementById('reset-btn').addEventListener('click', resetAll);

  document.getElementById('lr-slider').addEventListener('input', (e) => {
    document.getElementById('lr-value').textContent = e.target.value;
    if (network) network.learningRate = parseFloat(e.target.value);
  });

  initNetwork();
});