const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const physicalBrainSize = 1;

const freeformIterations = 50/*00*/;//6000

let NN = new TraditionalNN({
    learningRate: 0.0005,
    layerSizes: [12,10,8,1],
    // coordinateDescentTime: 6000
});

let activationNN = cloneNN(NN);

console.log('traditional error: ', NN.calculateRealError(data.inputs, data.outputs), activationNN.calculateRealError(data.inputs, data.outputs) );

window.trainingMode = 'freeform';

for(let i = 0; i < freeformIterations; i++){
    activationNN.train(data.inputs, data.outputs);
}

window.trainingMode = 'traditional';

console.log('now DAFN should be ahead: ', activationNN.calculateRealError(data.inputs, data.outputs) );

// Object.assign(NN, pretrainedNN);
// for(let i = 0; i < NN.layers.length; i++){
//     for(let j = 0; j < NN.layers[i].length; j++){
//         const l = NN.layers[i][j];
//         NN.layers[i][j] = new Neuron(NN.layers[i][j].trainableParams.length - CONSTANTS.extraParams - 1);
//         Object.assign(NN.layers[i][j], l);
//     }
    
// }

let camera;
function initCamera(){
    camera = {
        x: canvas.w / 2,
        y: canvas.h / 2,
        xv: 0, yv: 0, sv: 0,
        zoom: 500, targetZoom: 600
    };
}

const TAU = Math.PI * 2;
function renderNetwork(){
    // requestAnimationFrame(renderNetwork);

    ctx.fillStyle = 'black';
    ctx.fillRect(0,0,canvas.width,canvas.height);

    camera.x -= camera.xv * 5;
    camera.y -= camera.yv * 5;
    handleScroll({deltaY: camera.sv * 60});
    camera.zoom = interpolate(camera.zoom, camera.targetZoom, 0.1);

    ctx.save();

    ctx.translate(camera.x, camera.y);

    ctx.translate(physicalBrainSize/2, physicalBrainSize/2);
    ctx.scale(camera.zoom, camera.zoom);
    ctx.translate(-physicalBrainSize/2, -physicalBrainSize/2);

    // border
    ctx.strokeStyle = 'white';
    ctx.lineCap = 'round';
    ctx.lineWidth = 4.5/1920;
    ctx.beginPath();
    ctx.roundRect(0, 0, physicalBrainSize, physicalBrainSize, 100/1920);
    ctx.stroke();
    ctx.closePath();
    
    ctx.fillStyle = 'white';

    const margin = 0.03;
    const layerWidth = subdivide(1, NN.layers.length, margin);

    let x = margin;
    let lastPositions = [];
    let currentPositions = [];
    for(let i = 0; i < NN.layers.length; i++){
        // const layerStart = x;
        // const layerEnd = x + margin;
        const layerHeight = subdivide(1, NN.layers[i].length, margin);
        let y = margin;
        for(let j = 0; j < NN.layers[i].length; j++){
            // drawLayer(x,y, layerWidth, layerHeight)
            // drawDescisionTree(x, y, layerWidth, layerHeight, NN.layers[i][j]);
            ctx.beginPath();
            const position = [x+layerWidth/2,y+layerHeight/2,Math.min(layerWidth, layerHeight) / 2];
            ctx.arc(...position, 0, Math.PI * 2);
            currentPositions.push(position);
            ctx.stroke();
            ctx.closePath();

            y += margin + layerHeight;

            for(let k = 0; k < lastPositions.length; k++){
                ctx.beginPath();
                ctx.moveTo(...lastPositions[k]);
                ctx.lineTo(...position);
                ctx.stroke();
                ctx.closePath();
            }
        }

        x += margin + layerWidth;
        lastPositions = currentPositions;
        currentPositions = [];
    }

    ctx.lineWidth = 3;
    ctx.font = '400 56px Inter';
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'black';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const MSE = NN.calculateRealError(data.inputs, data.outputs);
    lastMSE = MSE;

    const MSE2 = activationNN.calculateRealError(data.inputs, data.outputs);
    lastMSE2 = MSE2;

    const t = "Error: [" + MSE.toFixed(2) + " | " + MSE2.toFixed(2) + "] Iterations: [" + iterations + "]";
    // const t = "Target: [" + data.outputs.toString() + "]" + " Outputs: [" + data.inputs.map(o => NN.forward(o)[0]?.toFixed(2)).toString() + "]";
    ctx.font = '500 1px Inter';
    const metrics = ctx.measureText(t);
    const width = metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight;

    ctx.font = `500 ${physicalBrainSize / (width + 0)}px Inter`;

    ctx.fillText(t, physicalBrainSize/2, -physicalBrainSize/2 * .09);

    ctx.restore();

    renderedFrame = true;
}

function subdivide(width, numRegions, margin){
    const withoutMargin = width - margin * (numRegions+1);
    return withoutMargin / numRegions;
}

window.addEventListener("keydown", (e)=>{return handleKey(e, true)});
window.addEventListener("keyup", (e)=>{return handleKey(e, false)});
window.addEventListener("wheel", (e)=>{return handleScroll(e);});

let input = {
    up: false,
    down: false,
    left: false,
    right: false,
    shift: false,
    z: false,
    x: false
};
function handleKey(e, isDown) {
    if(e.repeat) return e.preventDefault();

    if(e.code === 'Digit0' && isDown){
        return initCamera();
    } else if(e.code === 'KeyZ' || e.code === 'KeyX'){
        if(e.code === 'KeyZ') input.z = isDown;
        else input.x = isDown;
        camera.sv = input.x - input.z;
    }

    if(e.code === 'KeyW'){
        input.up = isDown;
    } else if(e.code === 'KeyA'){
        input.right = isDown;
    } else if(e.code === 'KeyS'){
        input.down = isDown;
    } else if(e.code === 'KeyD'){
        input.left = isDown;
    } else if(e.code === 'ShiftLeft' || e.code === 'ShiftRight'){
        input.shift = isDown;
    }

    camera.xv = (input.left - input.right);
    camera.yv = (input.down - input.up);

    if(input.shift === true) {
        camera.xv *= 3;
        camera.yv *= 3;
    }
}
function handleScroll(e){
    camera.targetZoom *= 1 - e.deltaY * 0.6 / 1000;
}

function interpolate(start, end, t){
    return (1-t) * start + end * t;
}

const width = 1600;
const height = 900;
function resize() {
    canvas.w = canvas.width = window.innerWidth;
    canvas.h = canvas.height = window.innerHeight;

    initCamera();
}
resize();
window.addEventListener('resize', resize);

const arrowAngle = Math.PI / 4;
const arrowLength = 12/1920;
function drawArrowOnLine([startX, startY], [endX, endY]){
    const middle = [
        (startX + endX) / 2,
        (startY + endY) / 2,
    ]

    const angle = Math.atan2(endY - startY, endX - startX);

    ctx.beginPath();
    ctx.translate(...middle);
    ctx.rotate(angle + arrowAngle / 2);
    ctx.moveTo(0,0);
    ctx.lineTo(arrowLength, 0);
    ctx.moveTo(0,0);
    ctx.rotate(-arrowAngle);
    ctx.lineTo(arrowLength, 0);
    ctx.moveTo(0,0);
    ctx.rotate(-angle + arrowAngle / 2);
    ctx.stroke();
    ctx.translate(-middle[0], -middle[1]);
    ctx.closePath();
}

function text(txt, x,y){
    ctx.strokeText(txt,x,y);
    ctx.fillText(txt,x,y);
}

function interpolate(a,b,t){
    return (1-t) * a + b * t;
}

// function run(){
//     NN.train(data.inputs, data.outputs);

//     renderNetwork();

//     requestAnimationFrame(run);

//     // setTimeout(() => {
//     //     run();
//     // }, 100)
// }

// run();

let iterations = 0;
let renderedFrame = false;
let lastMSE = NN.calculateRealError(data.inputs, data.outputs);
let lastMSE2 = activationNN.calculateRealError(data.inputs, data.outputs);
let logFlag1k = false, logFlag2k = false, logFlag3k = false, logFlag4k = false, logFlag5k = false;

// window.trainingTraditional = false;
let savedDAFN = null;

const numBatches = 100;///*00*/;
async function run(){
    if(window.stop === true) return;
    renderedFrame = false;

    for(let i = 0; i < numBatches; i++){
        // train one iteration
        NN.train(data.inputs, data.outputs);
        iterations++;
    }

    for(let i = 0; i < numBatches; i++){
        // train one iteration
        activationNN.train(data.inputs, data.outputs);
        // iterations++;
    }

    // if(window.trainingTraditional){
    //     if(iterations >= 0 && logFlag3 === false){
    //         logFlag3 = true;
    //         console.log('Traditional Immediate:', lastMSE / data.inputs.length);
    //     }
    
    //     else if(iterations >= 10_000 && logFlag === false){
    //         logFlag = true;
    //         console.log('Traditional 10k:', lastMSE / data.inputs.length);
    //     }
    
    //     else if(iterations >= 100_000 && logFlag2 === false){
    //         iterations = 0;
    //         logFlag = logFlag2 = logFlag3 = false;
    //         console.log('Traditional 100k', lastMSE / data.inputs.length);

    //         NN = savedDAFN;
    //         savedDAFN = null;
    //         window.trainingTraditional = false;
    //         window.redefineConstants();
    //     }
    // }

    // else {

    renderNetwork();

        // if(logFlag3 === false){
        //     logFlag3 = true;
        //     console.log(`Immediate: ` + [lastMSE / data.inputs.length, lastMSE2 / data.inputs.length]);
        // }

        if(iterations >= 0 && logFlag1k === false){
            logFlag1k = true;
            console.log(`Immediate: ` + [lastMSE / data.inputs.length, lastMSE2 / data.inputs.length]);
        }

        else if(iterations >= 10_000 && logFlag2k === false){
            logFlag2k = true;
            console.log(`10k: ` + [lastMSE / data.inputs.length, lastMSE2 / data.inputs.length]);
        }

        // else if(iterations >= 3_000 && logFlag3k === false){
        //     logFlag3k = true;
        //     console.log(`3k: ` + [lastMSE / data.inputs.length, lastMSE2 / data.inputs.length]);
        // }
    
        // else if(iterations >= 4_000 && logFlag4k === false){
        //     logFlag4k = true;
        //     console.log(`4k: ` + [lastMSE / data.inputs.length, lastMSE2 / data.inputs.length]);
        // }
    
        else if(iterations >= 100_000 && logFlag5k === false){
            iterations = 0;
            logFlag1k = logFlag2k = logFlag3k = logFlag4k = logFlag5k = false;
            console.log(`100k: ` + [lastMSE / data.inputs.length, lastMSE2 / data.inputs.length]);
    
            let err = NaN;
            let flag = true;
            while(isNaN(err)){
                if(flag === true){
                    flag = false;
                } else {
                    console.log('retrying!');
                }
                
                window.trainingMode = 'traditional';

                NN = new TraditionalNN({
                    learningRate: 0.0005,
                    layerSizes: [12,10,8,1],
                    // coordinateDescentTime: 6000
                });
    
                activationNN = cloneNN(NN);
                // savedDAFN = cloneNN(NN);
                // window.trainingTraditional = true;
                // window.redefineConstants();
    
                window.trainingMode = 'freeform';
    
                for(let i = 0; i < freeformIterations; i++){
                    activationNN.train(data.inputs, data.outputs);
                }

                err = NN.calculateRealError(data.inputs, data.outputs);
            }
            

            window.trainingMode = 'traditional';
        }
    // }

    
    
    // // wait until we render the frame
    // await until(() => {return renderedFrame;}, .1);

    // // run again
    // run();

    requestAnimationFrame(run);
}

function cloneNN(NN){
    const newNN = new TraditionalNN({
        learningRate: 0.0005,
        layerSizes: [12,10,8,1],
        // coordinateDescentTime: 6000
    });

    let tmp = [];

    for(let i = 0; i < NN.layers.length; i++){
        tmp[i] = [];
        for(let j = 0; j < NN.layers[i].length; j++){
            const n = NN.layers[i][j];
            tmp[i][j] = {process: n.process, processDerivative: n.processDerivative, activationParamDerivative: n.activationParamDerivative};
            delete n.process;
            delete n.processDerivative;
            delete n.activationParamDerivative;
        }
    }

    Object.assign(newNN, structuredClone(NN));
    for(let i = 0; i < newNN.layers.length; i++){
        for(let j = 0; j < newNN.layers[i].length; j++){
            const l = newNN.layers[i][j];
            newNN.layers[i][j] = new Neuron(newNN.layers[i][j].trainableParams.length - CONSTANTS.extraParams - 1);
            Object.assign(newNN.layers[i][j], l);
        }
    }

    for(let i = 0; i < NN.layers.length; i++){
        for(let j = 0; j < NN.layers[i].length; j++){
            for(let key in tmp[i][j]){
                NN.layers[i][j][key] = tmp[i][j][key];
            }
        }
    }

    return newNN;
}

// function until(condition, checkInterval=400) {
//     if(!!condition()) return true;
//     return new Promise(resolve => {
//         let interval = setInterval(() => {
//             if (!condition()) return;
//             clearInterval(interval);
//             resolve();
//         }, checkInterval)
//     })
// }

// renderNetwork();

// window.redefineConstants();
run();