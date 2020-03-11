const controller = document.querySelector('ion-loading-controller');
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

let colors = ["rgb(255, 255, 255)", "rgb(55, 66, 250)", "rgb(30, 144, 255)", "rgb(0, 0, 0)"];

let clip = [
    'circle(100% at 0 0)',
    'circle(100% at 100% 0)',
    'circle(100% at 100% 100%)',
    'circle(100% at 0 100%)',
    'circle(50% at 50% 50%)'
];


document.addEventListener("DOMContentLoaded", function(event) {
    createPattern();
    var colorsdiv = document.getElementById("colorsdiv"); 
    for(color in colors){
        colorsdiv.innerHTML +=   `
            <ion-chip outline onclick='changeColor(${color})' id="chip-${color}" style="--color:${colors[color]}; border-color:${colors[color]}">
                <ion-label>${color}</ion-label>
            </ion-chip> `
    }
});
function changeColor(index){
    loaderPresent("Prendo i valori...");
    var rgb = colors[index];
    rgb = rgb.substring(4, rgb.length-1)
            .replace(/ /g, '')
            .split(',');
    var redSlider = document.getElementById("redSlider"); 
    var greenSlider = document.getElementById("greenSlider"); 
    var blueSlider = document.getElementById("blueSlider");
    redSlider.ionChange = console.log("ciao");
    greenSlider.ionChange = console.log("ciao");
    blueSlider.ionChange = console.log("ciao");
    redSlider.value = rgb[0];
    greenSlider.value = rgb[1];
    blueSlider.value = rgb[2];
    loaderDismiss()
    
    
}

let containerPattern = document.querySelector(".pattern");

function random(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}
for (let i = 0; i < 6; i++) {
    const item = document.createElement("div");
    item.className = "item";
    let bg = random(colors);
    let fg = random(colors);
    while (fg === bg) {
        fg = random(colors);
    }
    item.style.setProperty("--bg-color", bg);
    item.style.setProperty("--fg-color", fg);
    item.style.setProperty("--clip", random(clip));

    const inner = document.createElement("div");
    inner.className = "inner";

    item.append(inner);
}

function createPattern() {
    containerPattern.innerHTML = "";
    for (let i = 0; i < 25; i++) {
        const item = document.createElement("div");
        item.className = "item";
        let bg = random(colors);
        let fg = random(colors);

        while (fg === bg) {
            fg = random(colors);
        }
        item.style.setProperty("--bg-color", bg);
        item.style.setProperty("--fg-color", fg);
        item.style.setProperty("--clip", random(clip));

        const inner = document.createElement("div");
        inner.className = "inner";

        item.append(inner);
        containerPattern.append(item);
    }
}
