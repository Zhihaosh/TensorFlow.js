dl = require('deeplearn');

class ComplementaryColorModel{
  
    constructor(){
        this.math = dl.ENV.math;
        this.learningRate = 0.1;
        this.session = null;
        this.optimizer = null;
        this.batchSize = 50;
        this.inputTensor = null;
        this.targetTensor = null;
        this.costTensor = null;
        this.predictionTensor = null;
        this.feedEntries = null;
        this.optimizer = new dl.SGDOptimizer(this.learningRate);
    }
    setUpSession(){
        const g = new dl.Graph();
        this.inputTensor = g.placeholder('input RGB value', [3]);
        this.targetTensor = g.placeholder('output RGB value', [3]);
        let fullConnectedLayer = this.createFullyConnectedLayer(g, this.inputTensor, 0, 64);
        fullConnectedLayer = this.createFullyConnectedLayer(g, fullConnectedLayer, 1, 32);
        fullConnectedLayer = this.createFullyConnectedLayer(g, fullConnectedLayer, 2, 16);
        this.predictionTensor = this.createFullyConnectedLayer(g, fullConnectedLayer, 3, 3);
        this.costTensor = g.meanSquaredCost(this.targetTensor, this.predictionTensor);
        this.session = new dl.Session(g, this.math);
        this.generateTrainingData(1e5);
    }
    createFullyConnectedLayer(graph, inputLayer, layerIndex, sizeOfThisLayer) {
      return graph.layers.dense(
          `fully_connected_${layerIndex}`, inputLayer, sizeOfThisLayer,
          (x) => graph.relu(x));
    }
    train1Batch(shouldFetchCost){
        const nextlearningRate = this.learningRate * Math.pow(0.85, Math.floor(step / 42));
        this.learningRate = nextlearningRate;
        let costValue = -1;
        this.math.scope(()=>{
            const cost = this.session.train(
                this.costTensor, this.feedEntries, this.batchSize, this.optimizer,
                shouldFetchCost? dl.CostReduction.MEAN : dl.CostReduction.NONE
            );
            if(!shouldFetchCost)
                return ;
            costValue = cost.get();
        });
        return costValue
    }
    predict(rgbColor) {
      let complementColor = [];
      this.math.scope(() => {
        const mapping = [{
          tensor: this.inputTensor,
          data: dl.Array1D.new(this.normalizeColor(rgbColor)),
        }];
        const evalOutput = this.session.eval(this.predictionTensor, mapping);
        const values = evalOutput.dataSync();
        const colors = this.denormalizeColor(Array.prototype.slice.call(values));
  
        // Make sure the values are within range.
        complementColor =
            colors.map(v => Math.round(Math.max(Math.min(v, 255), 0)));
      });
      return complementColor;
    }


    generateTrainingData(exampleCount) {
        const rawInputs = new Array(exampleCount);
    
        for (let i = 0; i < exampleCount; i++) {
          rawInputs[i] = [
            this.generateRandomChannelValue(), this.generateRandomChannelValue(),
            this.generateRandomChannelValue()
          ];
        }
        
        const inputArray = rawInputs.map(v => dl.Array1D.new(this.normalizeColor(v)));
        const targetArray = rawInputs.map(v => dl.Array1D.new(this.normalizeColor(this.computeComplementaryColor(v))));
        
        const shuffledInputProviderBuilder =
        new dl.InCPUMemoryShuffledInputProviderBuilder(
            [inputArray, targetArray]);
        const [inputProvider, targetProvider] = shuffledInputProviderBuilder.getInputProviders();

        this.feedEntries =[
            {tensor: this.inputTensor, data: inputProvider},
            {tensor: this.targetTensor, data: targetProvider}
        ];
    }

    normalizeColor(rgbColor) {
        return rgbColor.map(v => v / 255);
      }
    
    denormalizeColor(normalizedRgbColor) {
        return normalizedRgbColor.map(v => v * 255);
      }
    


    generateRandomChannelValue() {
        return Math.floor(Math.random() * 256);
      }

      computeComplementaryColor(rgbColor) {
        let r = rgbColor[0];
        let g = rgbColor[1];
        let b = rgbColor[2];
    
        // Convert RGB to HSL
        // Adapted from answer by 0x000f http://stackoverflow.com/a/34946092/4939630
        r /= 255.0;
        g /= 255.0;
        b /= 255.0;
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        let h = (max + min) / 2.0;
        let s = h;
        const l = h;
    
        if (max === min) {
          h = s = 0;  // achromatic
        } else {
          const d = max - min;
          s = (l > 0.5 ? d / (2.0 - max - min) : d / (max + min));
    
          if (max === r && g >= b) {
            h = 1.0472 * (g - b) / d;
          } else if (max === r && g < b) {
            h = 1.0472 * (g - b) / d + 6.2832;
          } else if (max === g) {
            h = 1.0472 * (b - r) / d + 2.0944;
          } else if (max === b) {
            h = 1.0472 * (r - g) / d + 4.1888;
          }
        }
    
        h = h / 6.2832 * 360.0 + 0;
    
        // Shift hue to opposite side of wheel and convert to [0-1] value
        h += 180;
        if (h > 360) {
          h -= 360;
        }
        h /= 360;
    
        // Convert h s and l values into r g and b values
        // Adapted from answer by Mohsen http://stackoverflow.com/a/9493060/4939630
        if (s === 0) {
          r = g = b = l;  // achromatic
        } else {
          const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
          };
    
          const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
          const p = 2 * l - q;
    
          r = hue2rgb(p, q, h + 1 / 3);
          g = hue2rgb(p, q, h);
          b = hue2rgb(p, q, h - 1 / 3);
        }
    
        return [r, g, b].map(v => Math.round(v * 255));
    }
}

const model = new ComplementaryColorModel();


model.setUpSession();
let step = 0;
function trainAndMaybeRender() {
  if (step > 1800) {
    // Stop training.
    return;
  }
  requestAnimationFrame(trainAndMaybeRender);
  // Schedule the next batch to be tb b rained.

  // We only fetch the cost every 5 steps because doing so requires a transfer
  // of data from the GPU.
  const localStepsToRun = 10;
  let cost;
  for (let i = 0; i < localStepsToRun; i++) {
    cost = model.train1Batch(i === localStepsToRun - 1);
    step++;
  }

  // Print data to console so the user can inspect.
  console.log('step', step - 1, 'cost', cost);

  // Visualize the predicted complement.
  const colorRows = document.querySelectorAll('tr[data-original-color]');
  for (let i = 0; i < colorRows.length; i++) {
    const rowElement = colorRows[i];
    const tds = rowElement.querySelectorAll('td');
    const originalColor =
        (rowElement.getAttribute('data-original-color'))
            .split(',')
            .map(v => parseInt(v, 10));
  
    // Visualize the predicted color.
    predictedColor = model.predict(originalColor)
     populateContainerWithColor(tds[2], predictedColor[0], predictedColor[1], predictedColor[2]);
  }
}

function populateContainerWithColor(
    container, r, g, b) {
  const originalColorString = 'rgb(' + [r, g, b].join(',') + ')';
  container.textContent = originalColorString;

  const colorBox = document.createElement('div');
  colorBox.classList.add('color-box');
  colorBox.style.background = originalColorString;
  container.appendChild(colorBox);
}

function initializeUi() {
  const colorRows = document.querySelectorAll('tr[data-original-color]');
  for (let i = 0; i < colorRows.length; i++) {
    const rowElement = colorRows[i];
    const tds = rowElement.querySelectorAll('td');
    const originalColor =
        (rowElement.getAttribute('data-original-color'))
            .split(',')
            .map(v => parseInt(v, 10));

    // Visualize the original color.
    populateContainerWithColor(
        tds[0], originalColor[0], originalColor[1], originalColor[2]);

    // Visualize the complementary color.
    const complement =
    model.computeComplementaryColor(originalColor);
    populateContainerWithColor(
        tds[1], complement[0], complement[1], complement[2]);
  }
}

// Kick off training.
initializeUi();
trainAndMaybeRender();



