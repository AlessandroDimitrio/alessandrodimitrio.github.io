let colors = ["#ffffff", "#e84118", "#c23616"];

let clip = [
    'circle(100% at 0 0)',
    'circle(100% at 100% 0)',
    'circle(100% at 100% 100%)',
    'circle(100% at 0 100%)',
    'circle(50% at 50% 50%)'
];
let osteologia = ["Cavita orbitaria", "Gabbia toracica", "Osso etmoide", "Ossa del bacino", "Fossa infratemporale e pterigopalatina", "Fossa cranica posteriore", "Base esterna del cranio", "Osso sfenoide", "Cavità nasale", "Sterno e coste", "Osso sfenoide", "Vertebre cervicali", "Mandibola"]
let artrologia = ["Articolazione coxofemorale", "Articolazione del gomito", "Generalità di una diartrosi", "Generalità di una sinartrosi", "Articolazioni della colonna vertebrale", "Articolazione temporomandibolare", "Articolazione della caviglia"]
let miologia = ["Muscoli della loggia anteriore e posteriore della coscia", "Muscoli del braccio", "Muscoli dell'addome e canale inguinale", "Descrizione del cavo ascellare", "Triangolo femorale e canale degli adduttori", "Muscoli della loggia anteriore dell'avambraccio", "Fasce del collo e muscoli sopra e sottoioidei", "Muscoli della spalla", "Muscoli della loggia posteriore dell'avambraccio", "Muscoli della parete anterolaterale dell'addome", "Descrizione del triangolo femorale e dei muscoli", "Muscoli e descrizione della loggia poplitea", "Muscoli del cavo ascellare"]
let chimica = ["Acido benzoico", "Acido butanoico", "Acido etanoico(acetico)", "Acido metanoico(formico)", "Acido fumarico", "Acido fosfatidico", "Acido maleico", "Acido oleico", "Acido propanoico", "Acido stearico", "Ac. tricloracetico", "Alanina", "Anidride acetica", "Anilina", "Benzaldeide", "1,2-benzochinone", "1,4-benzochinone", "Benzo-idrochinone", "1-butanolo", "L 2-butanolo", "D 2-butanolo", "2-metil 1-propanolo(1-isobutanolo)", "2-metil 2-propanolo(2-isobutanolo)", "Butanale", "Butanone(metiletilchetone)", "1-butene", "Cis 2-butene", "Trans 2-butene", "Ciclopropano", "Ciclobutano", "Ciclopentano", "Cicloesano", "1,3-cicloesadiene", "1,3,5-cicloesatriene", "Etano", "Etanolo", "Etanale(acetaldeide)", "Etene (etilene)", "Etere difenilico", "Etino (acetilene)", "Fenili-etil etere", "Fenolo", "Fruttosio", "Furano", "Tetraidrofurano", "Glicerolo(1,2,3-propantriolo)", "Glicina", "Gliceraldeide", "Glucosio", "Imidazolo", "Maltosio", "Metanale(formaldeide)", "Metanolo", "Metilbenzene(toluene)", "Naftalene", "1-propanolo", "2-propanolo", "Propanone(acetone)", "Pirano", "Tetraidropirano", "Piridina", "Pirimidina", "Pirrolo", "Purina", "Ribosio", "Saccarosio", "Urea", "Triclorometano(cloroformio)"];


document.addEventListener("DOMContentLoaded", function(event) {
    createQuestionAnatomia();
    createQuestionChimica();
    createPattern();
});

function createQuestionAnatomia() {
    var q1 = osteologia[Math.floor(Math.random() * osteologia.length)];
    var q2 = artrologia[Math.floor(Math.random() * artrologia.length)];
    var q3 = miologia[Math.floor(Math.random() * miologia.length)];
    var osteologiatext = document.getElementById("osteologia");
    var artologiatext = document.getElementById("artologia");
    var miologiatext = document.getElementById("miologia");
    osteologiatext.innerHTML = q1;
    artologiatext.innerHTML = q2;
    miologiatext.innerHTML = q3;
}

function createQuestionChimica() {
    var q3 = chimica[Math.floor(Math.random() * chimica.length)];
    var chimicatext = document.getElementById("chimica");
    chimicatext.innerHTML = q3;
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