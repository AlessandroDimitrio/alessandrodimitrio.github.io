var temperature = document.getElementById("temperature").value,
    lenght = document.getElementById("length").value,
    openBrackets = ["(", "{", "["],
    closedBrackets = [")", "}", "]"],
    zeroState = { c: [], h: [] },
    state = { c: [], h: [] },
    rnn_setting = {
      seed: "CC(C(=O)NCC1=CC=CC(N2C=NN=N2)=C1)N1C=CN=C1",
      length: lenght,
      temperature: temperature
    },
    model = {},
    cellsAmount = 0,
    cells = [],
    vocab = {},
    vocabSize = 0,
    probabilities = [];
    
function generateCallback(e, r) {
    var n = r.sample.split("\n")[1],
        t = n.charAt(n.length - 1);
    smile = "(" == t || "{" == t || "[" == t ? n.substring(0, n.length - 1) : n, console.log(smile);
    var l, s = brcheck(smile);
    if (ck_reversed = s.reverse(), 0 != ck_reversed.length)
        for (l = 0; l < ck_reversed.length; l++) console.log(closedBrackets[openBrackets.indexOf(ck_reversed[l])]), smile += closedBrackets[openBrackets.indexOf(ck_reversed[l])];
    console.log(smile);
    let c = new SmilesDrawer.Drawer({
        width: 300,
        height: 300,
        terminalCarbons: true,
        bondThickness: 1
    });
    SmilesDrawer.parse(smile, function(e) {
        c.draw(e, "canvas", "light", !1);
        loaderDismiss();
    }, function(e) {
        console.log(e);
        loaderDismiss();
    })
}

//const rnn = ml5.charRNN("public/models/16/", modelLoaded);
loadCheckpoints();

async function loadCheckpoints(){
  var path = "public/models/16/";
  var vars;
  //const vars = await reader.getAllVariables();

  const xhr = new XMLHttpRequest();
  xhr.open('GET', this.urlPath + "manifest.json");
  xhr.onload = () => {
    var checkpointManifest = JSON.parse(xhr.responseText);
    const variableNames = Object.keys(this.checkpointManifest);
    const variablePromises = variableNames.map(v => this.getVariable(v));
    return Promise.all(variablePromises).then((variables) => {
      this.variables = {};
      for (let i = 0; i < variables.length; i += 1) {
        this.variables[variableNames[i]] = variables[i];
      }
      return this.variables;
    });
  };
  xhr.send();

  Object.keys(vars).forEach((key) => {
    if (key.match(regexCell)) {
      if (key.match(regexWeights)) {
        this.model[`Kernel_${key.match(/[0-9]/)[0]}`] = vars[key];
        this.cellsAmount += 1;
      } else {
        this.model[`Bias_${key.match(/[0-9]/)[0]}`] = vars[key];
      }
    } else if (key.match(regexFullyConnected)) {
      if (key.match(regexWeights)) {
        this.model.fullyConnectedWeights = vars[key];
      } else {
        this.model.fullyConnectedBiases = vars[key];
      }
    } else {
      this.model[key] = vars[key];
    }
  });
  await this.loadVocab(path);
  await this.initCells();
  return this;

}

async function loadCheckpoints(path) {
  //const reader = new CheckpointLoader(path);
  if (path.charAt(path.length - 1) !== '/') {
    path += '/';
  }
  const vars = await reader.getAllVariables();

  Object.keys(vars).forEach((key) => {
    if (key.match(regexCell)) {
      if (key.match(regexWeights)) {
        this.model[`Kernel_${key.match(/[0-9]/)[0]}`] = vars[key];
        this.cellsAmount += 1;
      } else {
        this.model[`Bias_${key.match(/[0-9]/)[0]}`] = vars[key];
      }
    } else if (key.match(regexFullyConnected)) {
      if (key.match(regexWeights)) {
        this.model.fullyConnectedWeights = vars[key];
      } else {
        this.model.fullyConnectedBiases = vars[key];
      }
    } else {
      this.model[key] = vars[key];
    }
  });
  await this.loadVocab(path);
  await this.initCells();
  return this;
}

function generate(options, callback) {
    state = zeroState;
    return callCallback(this.generateInternal(options), callback);
  }
function generateInternal(options){
  //await this.ready;
  var seed = options.seed,
      length = options.length,
      temperature = options.temperature,
      stateful = options.stateful;

  if (!stateful) {
    state = zeroState;
  }
  const results = [];
  const userInput = Array.from(seed);
  const encodedInput = [];

  userInput.forEach((char) => {
    encodedInput.push(this.vocab[char]);
  });

  let input = encodedInput[0];
  let probabilitiesNormalized = [];

  for (let i = 0; i < userInput.length + length + -1; i += 1) {
    const onehotBuffer = await tf.buffer([1, this.vocabSize]);
    onehotBuffer.set(1.0, 0, input);
    const onehot = onehotBuffer.toTensor();
    let output;
    if (this.model.embedding) {
      const embedded = tf.matMul(onehot, this.model.embedding);
      output = tf.multiRNNCell(this.cells, embedded, this.state.c, this.state.h);
    } else {
      output = tf.multiRNNCell(this.cells, onehot, this.state.c, this.state.h);
    }
    state.c = output[0];
    state.h = output[1];

    const outputH = state.h[1];
    const weightedResult = tf.matMul(outputH, this.model.fullyConnectedWeights);
    const logits = tf.add(weightedResult, this.model.fullyConnectedBiases);
    const divided = tf.div(logits, tf.tensor(temperature));
    const probabilities = tf.exp(divided);

    probabilitiesNormalized = await tf.div(probabilities,tf.sum(probabilities)).data();

    if (i < userInput.length - 1) {
      input = encodedInput[i + 1];
    } else {
      input = sampleFromDistribution(probabilitiesNormalized);
      results.push(input);
    }
  }

  let generated = '';
  results.forEach((char) => {
    const mapped = Object.keys(this.vocab).find(key => this.vocab[key] === char);
    if (mapped) {
      generated += mapped;
    }
  });
  this.probabilities = probabilitiesNormalized;
  return {
    sample: generated,
    state: this.state,
  };
}

//const model = tf.loadGraphModel("file:///Users/alessandrodimitrio/Documents/GitHub/alessandrodimitrio.github.io/public/models/16/manifest.json");

/* const controller = document.querySelector('ion-loading-controller');
var loading;

async function loaderPresent(message) {
  if (loading) {
    await loading.dismiss();
  }
  loading = document.createElement('ion-loading');
  loading.message = message;

  document.body.appendChild(loading);
  return await loading.present();
}

async function loaderDismiss(){
  if (loading) {
    await loading.dismiss();
  }
  $("ion-loading").remove();
}

var formula = document.getElementById("formula"),
    openBrackets = ["(", "{", "["],
    closedBrackets = [")", "}", "]"];

function modelLoaded() {}
var i = 0;

function generate() {
    var e = document.getElementById("temperature").value,
        r = document.getElementById("length").value;
        loaderPresent("Caricamento in corso...")
    rnn.generate({
        seed: "CC(C(=O)NCC1=CC=CC(N2C=NN=N2)=C1)N1C=CN=C1",
        length: r,
        temperature: e
    }, function(e, r) {
        var n = r.sample.split("\n")[1],
            t = n.charAt(n.length - 1);
        smile = "(" == t || "{" == t || "[" == t ? n.substring(0, n.length - 1) : n, console.log(smile);
        var l, s = brcheck(smile);
        if (ck_reversed = s.reverse(), 0 != ck_reversed.length)
            for (l = 0; l < ck_reversed.length; l++) console.log(closedBrackets[openBrackets.indexOf(ck_reversed[l])]), smile += closedBrackets[openBrackets.indexOf(ck_reversed[l])];
        console.log(smile);
        let c = new SmilesDrawer.Drawer({
            width: 300,
            height: 300,
            terminalCarbons: true,
            bondThickness: 1
        });
        SmilesDrawer.parse(smile, function(e) {
            c.draw(e, "canvas", "light", !1);
            loaderDismiss();
        }, function(e) {
            console.log(e);
            loaderDismiss();
        })
    })
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
} */