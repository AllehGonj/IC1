
// Variable that composes the instance of Vue.js

let app = new Vue({
    el: '#app',
    data() {
        return {
            variablesNum: 1,
            maxNumData: 1,
            learningFactor: 0.25,
            bias: 2.0,
            conditionResults: [
                { value: 1, label: 'Verdadero' },
                { value: 0, label: 'Falso' }
            ],
            variablesValues: [
                { min: 0, max: 0 }
            ],
            min: [],
            max: [],
            arrayLoss: [],
            iterations: 0,
            formula: null,
            formulaData: null,
            rawDataValues: [],
            normalizeDataValues: [],
            rawDataConditions: [],
            rawDataHeaders: [],
            showIterationsAlert: false
        }
    },
    watch: {
        variablesNum: function(value) {
            this.variablesValues = [];
            this.rawDataValues = [];
            this.rawDataHeaders = [];

            for (let i = 0; i < value; i++) {
                this.addVariableValue();
            }

            this.addRawDataValue();
        },
        maxNumData: function () {
            this.rawDataHeaders = [];
            this.rawDataValues = [];
            this.addRawDataValue();
        }
    },
    methods: {
        validateVarNum() {
            if (this.variablesNum < 1) {
                this.variablesNum = 1
            }

            if (this.variablesNum > 20) {
                this.variablesNum = 20
            }
        },
        validateMaxNumData() {
            if (this.maxNumData < 1) {
                this.maxNumData = 1
            }

            if (this.maxNumData > 20) {
                this.maxNumData = 20
            }
        },
        validateLearnFactor() {
            if (this.learningFactor < 0) {
                this.learningFactor = 0
            }

            if (this.learningFactor > 1) {
                this.learningFactor = 1
            }
        },
        validateBias() {
            if (this.learningFactor < -10) {
                this.learningFactor = -10
            }

            if (this.learningFactor > 10) {
                this.learningFactor = 10
            }
        },
        addVariableValue() {
            this.variablesValues.push(
                { min: 0, max: 0 }
            )
        },
        addRawDataValue() {
            for(let i = 0; i < this.maxNumData; i++) {
                this.rawDataValues[i] = [];
                for(let j = 0; j <= this.variablesNum; j++) {
                    this.rawDataValues[i][j] = 0;
                }
            }
            for(let i = 0; i < this.variablesNum; i++) {
                this.rawDataHeaders.push(`X${i + 1}`)
            }
            this.rawDataHeaders.push('Target')
        },

        // Normaliza el array de datos de un solo Feature
        generateNormalizeData(rawData, xMin, xMax) {
            let allData = [];
            rawData.forEach((data, index) => {
                allData.push(
                    this.normalizeFeature(xMin, xMax, data)
                );
            });
            return allData;
        },
        minMax(x, xMin, xMax) {
            return  (x - xMin) / (xMax - xMin)
        },
        // Σ (weightsi * Xi) - Bias
        perceptronMainFunction(features, weights, bias) {
            let weightFeature = features.map((value, index) => {
                return features[index] * weights[index];
            });
            let sum = weightFeature.reduce((a, b) => {
                return a + b
            });
            return sum - bias;
        },

        
        lossFunction(target, y) {
            return target - y
        },
        deltaXi(lr, loss, xi) {
            return lr * loss * xi
        },
        recalculateWeight(weight, delta) {
            return Number(weight) + Number(delta)
        },
        recalculateBias(lr, loss, bias) {
            return bias - (loss * lr)
        },
        hardLimit(data) {
            if (data >= 0) {
                return 1
            } else {
                return 0
            }
        },

        // Llena array con las perdidas para detener las iteraciones
        populateLoss(value){
            this.arrayLoss = new Array(value);
            for (let index in this.arrayLoss) {
                this.arrayLoss[index] = 0
            }
        },

        // Da comienzo a las iteraciones
        initCalculate(features, initialWeight, initialBias, lr, targets, ) {
            let i = 0;
            let weights = [initialWeight, initialBias];
            let control = false;
            while (this.iterations < 1000 && !control) {
                if(this.iterations > features.length &&  !this.validateValue(this.arrayLoss)){
                    control = true
                }

                if(i === features.length){
                    i = 0
                }

                weights = this.calculateFormula(features[i], targets, weights[0], weights[1], lr, i);

                i++;
                this.iterations++

                if (this.iterations >= 1000)
                    this.showIterationsAlert = true;
            }
            return weights
        },
        // Recalcula Bias y Pesos para cada iteracion
        calculateFormula(features, targets, initialWeight, initialBias, lr, iteration) {
            let weights = initialWeight;
            let sumatory = this.perceptronMainFunction(features, initialWeight, initialBias);
            let hardLimit = this.hardLimit(sumatory);
            let loss = this.lossFunction(targets[iteration], hardLimit);
            this.arrayLoss[iteration] = (loss);
            let newBias = initialBias;
            if (loss !== 0.0){
                for (let i in features){
                    let deltaX = this.deltaXi(lr, loss, features[i]);
                    weights[i] = this.recalculateWeight(initialWeight[i], deltaX).toFixed(4)
                }
                newBias = this.recalculateBias(lr,loss,newBias)
            }

            return [weights, newBias]
        },

        getMaxMinArrays() {
            this.variablesValues.map(value => {
                this.min.push(Number(value.min));
                this.max.push(Number(value.max));
            });
        },
        generateRandom(min, max) {
            return Math.random() * (+max - +min) + +min;
        },
        normalizeFeature(xMin, xMax, data) {
            let array = data.map((value, index) => {
                return this.minMax(Number(value), xMin[index], xMax[index])
            })
            array.pop();
            return array;
        },
        fillListBinary(features) {
            let targets = [];
            for (let i in features) {
                let array = features[i];
                for (let j in array) {
                    if (j == array.length - 1) {
                        targets.push(array[j])
                    }
                }
            }
            return targets
        },
        validateValue(array) {
            for (let i in array) {
                if (array[i] !== 0)
                    return true
            }
            return false
        },
        generateFormula() {
            let featuresPart = '';
            for(let variable in this.formulaData[0]) {
                let index = Number(variable) + 1;
                featuresPart += `(${this.formulaData[0][variable]}) X${index} + `
            }
            this.formula = `${featuresPart} (${this.formulaData[1]})`;
        },
        calculatePerceptron() {
            this.getMaxMinArrays();
            let weights = [];

            let target = this.fillListBinary(this.rawDataValues);

            for (let i = 0; i < this.variablesValues.length; i++) {
                weights.push(this.generateRandom(-2, 2).toFixed(2));
            }

            let finalData =  this.generateNormalizeData(this.rawDataValues, this.min, this.max);
            this.populateLoss(target.length);
            this.formulaData = this.initCalculate(finalData, weights, this.bias, this.learningFactor, target);
            this.generateFormula();
            this.generateDataTables(finalData, target);

            console.log(this.formulaData)
        },
        generateDataTables(finalData, target) {
            finalData.forEach((value, index) => {
                let dataValues = value;
                dataValues.push(target[index])
                this.normalizeDataValues.push(dataValues)
            });
        }
    }
});