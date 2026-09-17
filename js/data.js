// data.js
// Generates a "two moons" toy dataset: two interleaving crescent shapes.
// This is not linearly separable, so it's a good test for a neural network.

function generateMoonsData(numPointsPerClass = 100, noise = 0.15) {
  const points = [];

  // Class 0: upper moon
  for (let i = 0; i < numPointsPerClass; i++) {
    const angle = Math.PI * (i / numPointsPerClass); // 0 to PI
    const x = Math.cos(angle) + (Math.random() - 0.5) * noise;
    const y = Math.sin(angle) + (Math.random() - 0.5) * noise;
    points.push({ x, y, label: 0 });
  }

  // Class 1: lower moon, shifted right and down
  for (let i = 0; i < numPointsPerClass; i++) {
    const angle = Math.PI * (i / numPointsPerClass);
    const x = 1 - Math.cos(angle) + (Math.random() - 0.5) * noise;
    const y = 1 - Math.sin(angle) - 0.5 + (Math.random() - 0.5) * noise;
    points.push({ x, y, label: 1 });
  }

  return points;
}

// Shuffle so the network doesn't see all of class 0 then all of class 1 in order
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function getDataset() {
  return shuffleArray(generateMoonsData());
}