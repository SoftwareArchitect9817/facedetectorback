const path = require("path")
const fs = require('fs');
const log = require('@vladmandic/pilogger');
const canvas = require('canvas');

// eslint-disable-next-line import/no-extraneous-dependencies, no-unused-vars, @typescript-eslint/no-unused-vars
const tf = require('@tensorflow/tfjs-node'); // in nodejs environments tfjs-node is required to be loaded before face-api
const faceapi = require('@vladmandic/face-api'); // use this when face-api is installed as module (majority of use cases)
const e = require("express");
const modelPathRoot = '../model';
const outPathRoot = '../out/';
const minConfidence = 0.15;
const imgPathRoot = path.join(__dirname, "../input/");
const maxResults = 5;
let optionsSSDMobileNet;

async function image(input) {
    const img = await canvas.loadImage(input);
    const c = canvas.createCanvas(img.width, img.height);
    const ctx = c.getContext('2d');
    ctx.drawImage(img, 0, 0, img.width, img.height);
    // const out = fs.createWriteStream('test.jpg');
    // const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
    // stream.pipe(out);
    return c;
}

async function drawFaces(input, data, imgname) {
    const img = await canvas.loadImage(input);
    const c = canvas.createCanvas(img.width, img.height);
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(img, 0, 0, img.width, img.height);
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    // draw title
    ctx.font = 'small-caps 20px "Segoe UI"';
    ctx.fillStyle = 'white';
    // ctx.fillText(`FPS: ${fps}`, 10, 25);
    for (const person of data) {
        // draw box around each face
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'deepskyblue';
        ctx.fillStyle = 'deepskyblue';
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.rect(person.detection.box.x, person.detection.box.y, person.detection.box.width, person.detection.box.height);
        ctx.stroke();
        ctx.globalAlpha = 1;
        // const expression = person.expressions.sort((a, b) => Object.values(a)[0] - Object.values(b)[0]);
        const expression = Object.entries(person.expressions).sort((a, b) => b[1] - a[1]);
        ctx.fillStyle = 'black';
        // ctx.fillText(`gender: ${Math.round(100 * person.genderProbability)}% ${person.gender}`, person.detection.box.x, person.detection.box.y - 59);
        // ctx.fillText(`expression: ${Math.round(100 * expression[0][1])}% ${expression[0][0]}`, person.detection.box.x, person.detection.box.y - 41);
        // ctx.fillText(`age: ${Math.round(person.age)} years`, person.detection.box.x, person.detection.box.y - 23);
        // ctx.fillText(`roll:${person.angle.roll.toFixed(3)} pitch:${person.angle.pitch.toFixed(3)} yaw:${person.angle.yaw.toFixed(3)}`, person.detection.box.x, person.detection.box.y - 5);
        ctx.fillStyle = 'lightblue';
        // ctx.fillText(`gender: ${Math.round(100 * person.genderProbability)}% ${person.gender}`, person.detection.box.x, person.detection.box.y - 60);
        // ctx.fillText(`expression: ${Math.round(100 * expression[0][1])}% ${expression[0][0]}`, person.detection.box.x, person.detection.box.y - 42);
        // ctx.fillText(`age: ${Math.round(person.age)} years`, person.detection.box.x, person.detection.box.y - 24);
        // ctx.fillText(`roll:${person.angle.roll.toFixed(3)} pitch:${person.angle.pitch.toFixed(3)} yaw:${person.angle.yaw.toFixed(3)}`, person.detection.box.x, person.detection.box.y - 6);
        // draw face points for each face
        ctx.globalAlpha = 0.8;
        ctx.fillStyle = 'lightblue';
        const pointSize = 2;
        for (let i = 0; i < person.landmarks.positions.length; i++) {
            ctx.beginPath();
            ctx.arc(person.landmarks.positions[i].x, person.landmarks.positions[i].y, pointSize, 0, 2 * Math.PI);
            // ctx.fillText(`${i}`, person.landmarks.positions[i].x + 4, person.landmarks.positions[i].y + 4);
            ctx.fill();
        }
    }

    const out = fs.createWriteStream(path.join(__dirname, outPathRoot) + imgname);
    const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
    stream.pipe(out);
    return ctx
}

async function detect(tensor) {
    const result = await faceapi
        .detectAllFaces(tensor, optionsSSDMobileNet)
        .withFaceLandmarks()
        .withFaceExpressions()
        .withFaceDescriptors()
        .withAgeAndGender();
    return result;
}

function print(face) {
    const expression = Object.entries(face.expressions).reduce((acc, val) => ((val[1] > acc[1]) ? val : acc), ['', 0]);
    const box = [face.alignedRect._box._x, face.alignedRect._box._y, face.alignedRect._box._width, face.alignedRect._box._height];
    const gender = `Gender: ${Math.round(100 * face.genderProbability)}% ${face.gender}`;
    log.data(`Detection confidence: ${Math.round(100 * face.detection._score)}% ${gender} Age: ${Math.round(10 * face.age) / 10} Expression: ${Math.round(100 * expression[1])}% ${expression[0]} Box: ${box.map((a) => Math.round(a))}`);
}

async function main(img) {
    log.header();
    log.info('FaceAPI single-process test');

    faceapi.env.monkeyPatch({ Canvas: canvas.Canvas, Image: canvas.Image, ImageData: canvas.ImageData });

    await faceapi.tf.setBackend('tensorflow');
    await faceapi.tf.enableProdMode();
    await faceapi.tf.ENV.set('DEBUG', false);
    await faceapi.tf.ready();

    log.state(`Version: TensorFlow/JS ${faceapi.tf?.version_core} FaceAPI ${faceapi.version} Backend: ${faceapi.tf?.getBackend()}`);

    log.info('Loading FaceAPI models');
    const modelPath = path.join(__dirname, modelPathRoot);
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
    await faceapi.nets.ageGenderNet.loadFromDisk(modelPath);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
    await faceapi.nets.faceExpressionNet.loadFromDisk(modelPath);
    optionsSSDMobileNet = new faceapi.SsdMobilenetv1Options({ minConfidence, maxResults });
    const c = await image(path.join(imgPathRoot, img));
    const result = await detect(c);
    if (result.length) {
        await drawFaces(path.join(imgPathRoot, img), result, img)
        return true
    } else {
        return false
    }

}


exports.upload = async (req, res, next) => {
    let d = req.files
    let row = {}
    for (let i in d) {
        row[d[i].fieldname] = d[i].filename
    }
    console.log(row.file)
    // req.images = row
    if (row.file) {
        let result = await main(row.file)
        if (result) {
            res.send({data: "http://192.168.114.41:3998/" + row.file, status: true})
            return next()
        } else {
            res.send({
                status: false,
                data: "fail"
            })
            return next()
        }
    } else {
        res.send({
            status: false,
            data: "fail"
        })
        return next()
    }
}