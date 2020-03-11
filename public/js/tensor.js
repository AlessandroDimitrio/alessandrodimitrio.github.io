var temperature = document.getElementById("temperature").value;
var length = document.getElementById("length").value;
var openBrackets = ["(", "{", "["];
var closedBrackets = [")", "}", "]"];
var zeroState = {c: [],h: []};
var state = {c: [],h: []};
var model = {};
var cellsAmount = 0;
var cells = [];
var vocab = {};
var checkpointManifest;
var vocabSize = 0;
var variablesNN;
var probabilities = [];

$( document ).ready(function() {
  initMol();
});

async function loadCheckpoints() {
  var vars = await loadModel();
  cellsAmount = 0;
  Object.keys(vars).forEach((key) => {
    if (key.match(/cell_[0-9]|lstm_[0-9]/gi)) {
      if (key.match(/weights|weight|kernel|kernels|w/gi)) {
        model[`Kernel_${key.match(/[0-9]/)[0]}`] = vars[key];
        cellsAmount += 1;
      } else {
        model[`Bias_${key.match(/[0-9]/)[0]}`] = vars[key];
      }
    } else if (key.match(/softmax/gi)) {
      if (key.match(/weights|weight|kernel|kernels|w/gi)) {
        model.fullyConnectedWeights = vars[key];
      } else {
        model.fullyConnectedBiases = vars[key];
      }
    } else {
      model[key] = vars[key];
    }
  });
  const json = await fetch(`public/models/16/vocab.json`)
  .then(response => response.json())
  .catch(err => console.error(err));
  vocab = json;
  vocabSize = Object.keys(json).length;
  cells = [];
  zeroState = { c: [], h: [] };
  const forgetBias = tf.tensor(1.0);
  const lstm = (i) => {
    const cell = (DATA,  C, H) =>
      tf.basicLSTMCell(forgetBias, model[`Kernel_${i}`], model[`Bias_${i}`], DATA, C, H);
    return cell;
  };
  for (let i = 0; i < cellsAmount; i += 1) {
    zeroState.c.push(tf.zeros([1, model[`Bias_${i}`].shape[0] / 4]));
    zeroState.h.push(tf.zeros([1, model[`Bias_${i}`].shape[0] / 4]));
    cells.push(lstm(i));
  }
  state = zeroState;
}

async function loadModel() {
  if (variablesNN != null) {
    return Promise.resolve(variablesNN);
  }
  if (checkpointManifest == null) {
    checkpointManifest = await fetch("public/models/16/manifest.json")
    .then(response => response.json()
    .then(json => { return json;
    }));
  }
  var variableNames = Object.keys(checkpointManifest);
  var variablePromises = variableNames.map(varName => getVariable(varName));
  return Promise.all(variablePromises).then((variables) => {
          variablesNN = {};
          for (let i = 0; i < variables.length; i += 1) {
              variablesNN[variableNames[i]] = variables[i];
          }
          return variablesNN;
      });
}
const sampleFromDistribution = (input) => {
  const randomValue = Math.random();
  let sum = 0;
  let result;
  for (let j = 0; j < input.length; j += 1) {
    sum += input[j];
    if (randomValue < sum) {
      result = j;
      break;
    }
  }
  return result;
};

function getVariable(varName) {
    const variableRequestPromiseMethod = (resolve) => {
        const xhr = new XMLHttpRequest();
        xhr.responseType = 'arraybuffer';
        const fname = checkpointManifest[varName].filename;
        xhr.open('GET', "public/models/16/" + fname);
        xhr.onload = () => {
            const values = new Float32Array(xhr.response);
            const tensor = tf.Tensor.make(checkpointManifest[varName].shape, {
                values
            });
            resolve(tensor);
        };
        xhr.send();
    };
    return new Promise(variableRequestPromiseMethod);
}
function initMol(){
  loaderPresent("Caricamento in corso...");
  var rnn_setting = {
    seed: "CC(C(=O)NCC1=CC=CC(N2C=NN=N2)=C1)N1C=CN=C1",
    length: document.getElementById("length").value,
    temperature: document.getElementById("temperature").value
};
  generateMol(rnn_setting).then(res => {
    var n = res.sample.split("\n")[1];
    var t = n.charAt(n.length - 1);
    smile = "(" == t || "{" == t || "[" == t ? n.substring(0, n.length - 1) : n, console.log(smile);
    var l, s = brcheck(smile);
    if (ck_reversed = s.reverse(), 0 != ck_reversed.length){
        for (l = 0; l < ck_reversed.length; l++){
          smile += closedBrackets[openBrackets.indexOf(ck_reversed[l])];
        }
      }
    let molecule = new SmilesDrawer.Drawer({
        width: 300,
        height: 300,
        terminalCarbons: true,
        bondThickness: 1
    });
    SmilesDrawer.parse(smile, function(tree) {
      molecule.draw(tree, "molcanvas", "light", false);
      loaderDismiss();
    }, function (err) {
      console.log(err);
      loaderDismiss();
      loaderPresent("Qualcosa è andato storto, riprovo...");
      initMol();
  })
});
}

async function generateMol(options) {
    await loadCheckpoints();
    const seed = options.seed;
    const length = +options.length;
    const temperature = +options.temperature;
    const stateful = options.stateful;

    if (!stateful) {
        state = zeroState;
    }
    const results = [];
    const userInput = Array.from(seed);
    const encodedInput = [];

    userInput.forEach((char) => {
        encodedInput.push(vocab[char]);
    });

    let input = encodedInput[0];
    let probabilitiesNormalized = [];
    for (let i = 0; i < userInput.length + length + -1; i += 1) {
        const onehotBuffer = await tf.buffer([1, vocabSize]);
        onehotBuffer.set(1.0, 0, input);
        const onehot = onehotBuffer.toTensor();
        let output;
        if (model.embedding) {
            const embedded = tf.matMul(onehot, model.embedding);
            output = tf.multiRNNCell(cells, embedded, state.c, state.h);
        } else {
            output = tf.multiRNNCell(cells, onehot, state.c, state.h);
        }

        state.c = output[0];
        state.h = output[1];

        const outputH = state.h[1];
        const weightedResult = tf.matMul(outputH, model.fullyConnectedWeights);
        const logits = tf.add(weightedResult, model.fullyConnectedBiases);
        const divided = tf.div(logits, tf.tensor(temperature));
        const probabilities = tf.exp(divided);
        probabilitiesNormalized = await tf.div(probabilities, tf.sum(probabilities)).data();
        if (i < userInput.length - 1) {
            input = encodedInput[i + 1];
        } else {
            input = sampleFromDistribution(probabilitiesNormalized);
            results.push(input);
        }
    }
    let generated = '';
    results.forEach((char) => {
        const mapped = Object.keys(vocab).find(key => vocab[key] === char);
        if (mapped) {
            generated += mapped;
        }
    });
    probabilities = probabilitiesNormalized;
    return {
        sample: generated,
        state: state,
    };
}

function brcheck(e) {
  let r = [];
  for (let n of e)
      if (openBrackets.includes(n)) r.push(n);
      else if (closedBrackets.includes(n)) {
      let e = openBrackets[closedBrackets.indexOf(n)];
      if (r[r.length - 1] !== e) {
          r.push(n);
          break
      }
      r.splice(-1, 1)
  }
  return r
}
