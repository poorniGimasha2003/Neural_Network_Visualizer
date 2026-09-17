// visualize.js
// Handles all canvas drawing: decision boundary background, data points, and loss chart.

const canvas = document.getElementById('nn-canvas');
const ctx = canvas.getContext('2d');

const lossCanvas = document.getElementById('loss-canvas');
const lossCtx = lossCanvas.getContext('2d');

// Our data lives roughly in the range [-1.5, 2.5] on both axes.
// These helpers convert data coordinates <-> pixel coordinates.
const DATA_MIN = -1.5;
const DATA_MAX = 2.5;

function dataToPixelX(x) {
  return ((x - DATA_MIN) / (DATA_MAX - DATA_MIN)) * canvas.width;
}
function dataToPixelY(y) {
  // Flip Y so positive is up, like a normal graph
  return canvas.height - ((y - DATA_MIN) / (DATA_MAX - DATA_MIN)) * canvas.height;
}
function pixelToDataX(px) {
  return DATA_MIN + (px / canvas.width) * (DATA_MAX - DATA_MIN);
}
function pixelToDataY(py) {
  return DATA_MIN + ((canvas.height - py) / canvas.height) * (DATA_MAX - DATA_MIN);
}

// Draws the decision boundary by classifying a grid of pixels and coloring
// each one based on the network's prediction (this is the "wow" visual).
function drawDecisionBoundary(network) {
  const resolution = 4; // pixels per grid cell — lower = smoother but slower
  const imageData = ctx.createImageData(canvas.width, canvas.height);

  for (let px = 0; px < canvas.width; px += resolution) {
    for (let py = 0; py < canvas.height; py += resolution) {
      const x = pixelToDataX(px);
      const y = pixelToDataY(py);
      const prediction = network.predict([x, y]); // 0 to 1

      // Blend between two colors based on prediction confidence
      // Class 0 = blue-ish, Class 1 = orange-ish
      const r = Math.floor(255 * prediction + 80 * (1 - prediction));
      const g = Math.floor(140 * prediction + 120 * (1 - prediction));
      const b = Math.floor(60 * prediction + 220 * (1 - prediction));

      // Fill a resolution x resolution block with this color
      for (let dx = 0; dx < resolution; dx++) {
        for (let dy = 0; dy < resolution; dy++) {
          const x2 = px + dx, y2 = py + dy;
          if (x2 >= canvas.width || y2 >= canvas.height) continue;
          const idx = (y2 * canvas.width + x2) * 4;
          imageData.data[idx] = r;
          imageData.data[idx + 1] = g;
          imageData.data[idx + 2] = b;
          imageData.data[idx + 3] = 255;
        }
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

// Draws the actual data points on top of the boundary
function drawDataPoints(dataset) {
  for (const point of dataset) {
    const px = dataToPixelX(point.x);
    const py = dataToPixelY(point.y);

    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fillStyle = point.label === 1 ? '#ff8c00' : '#1e50dc';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.fill();
    ctx.stroke();
  }
}

// One full redraw: boundary + points
function render(network, dataset) {
  drawDecisionBoundary(network);
  drawDataPoints(dataset);
}

// ---- Loss chart ----
let lossHistory = [];

function resetLossChart() {
  lossHistory = [];
}

function drawLossChart() {
  lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
  if (lossHistory.length < 2) return;

  const maxLoss = Math.max(...lossHistory, 0.1);
  const stepX = lossCanvas.width / (lossHistory.length - 1);

  lossCtx.beginPath();
  lossCtx.strokeStyle = '#1e50dc';
  lossCtx.lineWidth = 2;

  lossHistory.forEach((loss, i) => {
    const px = i * stepX;
    const py = lossCanvas.height - (loss / maxLoss) * lossCanvas.height;
    if (i === 0) lossCtx.moveTo(px, py);
    else lossCtx.lineTo(px, py);
  });

  lossCtx.stroke();
}

function recordLoss(loss) {
  lossHistory.push(loss);
  drawLossChart();
}