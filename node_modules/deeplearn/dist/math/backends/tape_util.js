"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var environment_1 = require("../../environment");
var util = require("../../util");
var ndarray_1 = require("../ndarray");
function getFilteredNodesXToY(tape, xs, y) {
    var arraysFromX = {};
    var nodesFromX = {};
    for (var i = 0; i < xs.length; i++) {
        arraysFromX[xs[i].id] = true;
    }
    for (var i = 0; i < tape.length; i++) {
        var node = tape[i];
        var nodeInputs = node.inputAndArgs.inputs;
        for (var inputName in nodeInputs) {
            var input = nodeInputs[inputName];
            var anyInputFromX = false;
            for (var j = 0; j < xs.length; j++) {
                if (arraysFromX[input.id]) {
                    if (node.output instanceof ndarray_1.NDArray) {
                        arraysFromX[node.output.id] = true;
                    }
                    else {
                        var keys = Object.keys(node.output);
                        for (var _i = 0, keys_1 = keys; _i < keys_1.length; _i++) {
                            var key = keys_1[_i];
                            arraysFromX[node.output[key].id] = true;
                        }
                    }
                    anyInputFromX = true;
                    nodesFromX[node.id] = true;
                    break;
                }
            }
            if (anyInputFromX) {
                break;
            }
        }
    }
    var arraysLeadToY = {};
    arraysLeadToY[y.id] = true;
    var nodesToY = {};
    for (var i = tape.length - 1; i >= 0; i--) {
        var node = tape[i];
        var nodeInputs = node.inputAndArgs.inputs;
        var outputs = [];
        if (node.output instanceof ndarray_1.NDArray) {
            outputs.push(node.output);
        }
        else {
            var keys = Object.keys(node.output);
            for (var _a = 0, keys_2 = keys; _a < keys_2.length; _a++) {
                var key = keys_2[_a];
                outputs.push(node.output[key]);
            }
        }
        for (var j = 0; j < outputs.length; j++) {
            if (arraysLeadToY[outputs[j].id]) {
                for (var inputName in nodeInputs) {
                    arraysLeadToY[nodeInputs[inputName].id] = true;
                    nodesToY[node.id] = true;
                }
                break;
            }
        }
    }
    var filteredTape = [];
    for (var i = 0; i < tape.length; i++) {
        var node = tape[i];
        if (nodesFromX[node.id] && nodesToY[node.id]) {
            var prunedInputs = {};
            for (var inputName in node.inputAndArgs.inputs) {
                var nodeInput = node.inputAndArgs.inputs[inputName];
                if (arraysFromX[nodeInput.id]) {
                    prunedInputs[inputName] = nodeInput;
                }
            }
            var prunedOutputs = void 0;
            if (node.output instanceof ndarray_1.NDArray) {
                prunedOutputs = node.output;
            }
            else {
                prunedOutputs = {};
                for (var outputName in node.output) {
                    var output = node.output[outputName];
                    if (arraysLeadToY[output.id]) {
                        prunedOutputs[outputName] = node.output[outputName];
                    }
                }
            }
            var prunedNode = Object.assign({}, node);
            prunedNode.inputAndArgs = { inputs: prunedInputs };
            prunedNode.output = prunedOutputs;
            filteredTape.push(prunedNode);
        }
    }
    return filteredTape;
}
exports.getFilteredNodesXToY = getFilteredNodesXToY;
function backpropagateGradients(arrayAccumulatedGradientMap, filteredTape) {
    for (var i = filteredTape.length - 1; i >= 0; i--) {
        var node = filteredTape[i];
        var dy = void 0;
        if (node.output instanceof ndarray_1.NDArray) {
            dy = arrayAccumulatedGradientMap[node.output.id];
        }
        else {
            dy = {};
            var keys = Object.keys(node.output);
            for (var _i = 0, keys_3 = keys; _i < keys_3.length; _i++) {
                var key = keys_3[_i];
                dy[key] = arrayAccumulatedGradientMap[node.output[key].id];
            }
        }
        if (node.gradient == null) {
            throw new Error("Cannot compute gradient: gradient function not found " +
                ("for " + node.name + "."));
        }
        var inputGradients = node.gradient(dy, node.output);
        for (var inputName in node.inputAndArgs.inputs) {
            if (!(inputName in inputGradients)) {
                throw new Error("Cannot backprop through input " + inputName + ". " +
                    ("Available gradients found: " + Object.keys(inputGradients) + "."));
            }
            var dx = inputGradients[inputName]();
            var x = node.inputAndArgs.inputs[inputName];
            if (!util.arraysEqual(dx.shape, x.shape)) {
                throw new Error("Error in gradient for op " + node.name + ". The gradient of input " +
                    ("'" + inputName + "' has shape '" + dx.shape + "', which does not match ") +
                    ("the shape of the input '" + x.shape + "'"));
            }
            if (arrayAccumulatedGradientMap[x.id] == null) {
                arrayAccumulatedGradientMap[x.id] = dx;
            }
            else {
                var curGradient = arrayAccumulatedGradientMap[x.id];
                arrayAccumulatedGradientMap[x.id] = environment_1.ENV.math.add(curGradient, dx);
                curGradient.dispose();
            }
        }
    }
}
exports.backpropagateGradients = backpropagateGradients;
function computeInputs(tape) {
    var outputArrays = {};
    for (var i = 0; i < tape.length; i++) {
        var node = tape[i];
        if (node.output instanceof ndarray_1.NDArray) {
            outputArrays[node.output.id] = true;
        }
        else {
            var keys = Object.keys(node.output);
            for (var _i = 0, keys_4 = keys; _i < keys_4.length; _i++) {
                var key = keys_4[_i];
                outputArrays[node.output[key].id] = true;
            }
        }
    }
    var inputArrays = {};
    var inputArraysSeen = {};
    var idx = 0;
    for (var i = 0; i < tape.length; i++) {
        var node = tape[i];
        var inputs = node.inputAndArgs.inputs;
        var keys = Object.keys(inputs);
        for (var _a = 0, keys_5 = keys; _a < keys_5.length; _a++) {
            var key = keys_5[_a];
            if (!outputArrays[inputs[key].id] && !inputArraysSeen[inputs[key].id]) {
                inputArrays[(idx++).toString()] = inputs[key];
                inputArraysSeen[inputs[key].id] = true;
            }
        }
    }
    return inputArrays;
}
exports.computeInputs = computeInputs;
function extractNDArraysFromScopeResult(result) {
    if (result == null) {
        return [];
    }
    if (result instanceof ndarray_1.NDArray) {
        return [result];
    }
    var list = [];
    var resultObj = result;
    for (var k in resultObj) {
        var val = resultObj[k];
        if (val instanceof ndarray_1.NDArray) {
            list.push(val);
        }
    }
    return list;
}
exports.extractNDArraysFromScopeResult = extractNDArraysFromScopeResult;
function stripUndefinedInputsFromInputConfig(config) {
    var keys = Object.keys(config.inputs);
    keys.forEach(function (key) {
        if (config.inputs[key] == null) {
            delete config.inputs[key];
        }
    });
    return config;
}
exports.stripUndefinedInputsFromInputConfig = stripUndefinedInputsFromInputConfig;
