var arrayLoss: MutableList<Double> = mutableListOf<Double>()
var iteraciones = 0
var lr = 0.2
var min = listOf(0.0, 0.0)
var max = listOf(3.0, 3.0)
fun main() {

    var i1 = listOf(3.0,3.0)
    var i2 = listOf(2.0,2.0)
    var i3 = listOf(1.0,1.0)
    var i4 = listOf(2.0,0.0)
    var i5 = listOf(3.0,0.0)
    var i6 = listOf(0.0,2.0)

    val data: List<List<Double>> = listOf(i1, i2, i3, i4, i5, i6)
    onePerceptron(data)
    firstPerceptron(data)
    secondPerceptron(data)
}

// Esta funcion prueba que no es posible realizar el ejercicio con un solo perceptron
fun onePerceptron(data: List<List<Double>>) {
    var bias = 1.0
    var weights = listOf(0.0, 0.0)
    var target = listOf(1, 1, 1, 0, 0, 0)
    var finalData = generateNormalaizeData(data, min, max)
    println("---------------Resultado perceptron simple--------------------------- ")
    populateLoss(target.size)
    println("Y = ${initCalculate(finalData, weights, bias, lr, target)}")
}

// Esta funcion calcula los pesos y bias para el primer perceptron (conjunto A)
fun firstPerceptron(data: List<List<Double>>) {

    var bias = 1.0
    iteraciones = 0
    var target = listOf(1, 1, 1, 0, 0, 1)
    var weights = listOf(0.0, 0.0)
    var finalData2 = generateNormalaizeData(data, min, max)
    println("--------------Resultado conjunto A primer perceptron-------------------------- ")
    populateLoss(target.size)
    println("Y = ${initCalculate(finalData2, weights, bias, lr, target)}")


}

// Esta funcion calcula los pesos y bias para el segundo perceptron (conjunto B)
fun secondPerceptron(data: List<List<Double>>) {

    arrayLoss = mutableListOf<Double>()
    iteraciones = 0
    var bias = 1.0
    var target = listOf(0, 0, 0, 0, 0, 1)
    var weights = listOf(0.0, 0.0)
    var finalData2 = generateNormalaizeData(data, min, max)
    println("--------------Resultado conjunto B segundo perceptron-------------------------- ")
    populateLoss(target.size)
    println("Y = ${initCalculate(finalData2, weights, bias, lr, target)}")

}
// genera la informacion normalizada (Normalizacion min-max)
fun generateNormalaizeData(rawData: List<List<Double>>, xmin: List<Double>, xmax: List<Double>): List<MutableList<Double>> {
    var arrayAllData: MutableList<MutableList<Double>> = mutableListOf<MutableList<Double>>()
    for (data in rawData) {
        arrayAllData.add(normalizeFeature(xmin, xmax, data) as MutableList<Double>)
    }
    return arrayAllData.toList()
}

// Σ(weight_i *x_i) - bias
fun perceptronMainFunction(features: List<Double>, weights: List<Double>, bias: Double): Double {
    val sums = (0 until features.size).map { features[it] * weights[it] }
    return sums.sum() - bias
}

fun minMax(x: Double, xmin: Double, xmax: Double): Double {
    return (x - xmin) / (xmax - xmin)
}

//normaliza el array de datos de un solo feature
fun normalizeFeature(xmin: List<Double>, xmax: List<Double>, data: List<Double>): List<Double> {
    val normalizeList = (0 until data.size).map { minMax(data[it], xmin[it.toInt()], xmax[it.toInt()]) }
    return normalizeList

}

//calcula la perdida de cada iteracion
fun lossFunction(target: Int, y: Int): Double {
    return (target - y).toDouble()
}

//calcula el delta de un dato de un feature
fun deltaXi(lr: Double, loss: Double, xi: Double): Double {
    return lr * loss * xi
}
//recalcula el peso del feature dado
fun recalculateWeight(weight: Double, delta: Double): Double {
    return truncate(weight + delta)
}
//recalcula el bias
fun recalculateBias(lr: Double, loss: Double, bias: Double): Double {
    return truncate(bias - (loss * lr))
}


fun hardLimit(data: Double): Int {
    if (data >= 0) {
        return 1
    } else {
        return 0
    }
}

//llena el array de las perdidas para saber cuando terminar las iteraciones
private fun populateLoss(value: Int) {
    arrayLoss = MutableList(value) { index -> 0.0 }
}


//Inicia las iteraciones
fun initCalculate(features: List<List<Double>>, initialWeight: List<Double>, initialBias: Double, lr: Double, targets: List<Int>): Pair<MutableList<Double>, Double> {

    var weights = Pair(initialWeight.toMutableList(), initialBias)
    var control = false
    var j = 0
    while (iteraciones < 10000 && !control) {
        if (iteraciones > features.size && !arrayLoss.any { it < 0 || it > 0 }) {
            println("Deltas/Perdidas ${arrayLoss}")
            control = true
        }
        if (j == features.size) j = 0
        weights = calculateFormula(features[j], targets, weights.first, weights.second, lr, j)
        j++
        iteraciones++

    }
    if (iteraciones == 10000) {
        System.err.println("No se pudo realizar el aprendizaje")
        System.err.println("Deltas/Perdidas ${arrayLoss}")
    }
    println("Iteraciones ${iteraciones}")
    return weights
}

//Recacula bias y pesos para cada iteracion
fun calculateFormula(features: List<Double>, targets: List<Int>, initialWeight: List<Double>, initialBias: Double, lr: Double, iteration: Int): Pair<MutableList<Double>, Double> {
    var weights = initialWeight.toMutableList()
    var sumatory = perceptronMainFunction(features, initialWeight, initialBias)

    var hardLimit = hardLimit(sumatory)
    var loss = lossFunction(targets[iteration], hardLimit)
    arrayLoss[iteration] = (loss)
    var newBias = initialBias
    if (loss != 0.0) {
        for (i in features.indices) {
            val deltaX = deltaXi(lr, loss, features[i])
            weights[i] = String.format("%.4f", recalculateWeight(initialWeight[i], deltaX)).toDouble()
        }
        newBias = recalculateBias(lr, loss, newBias)

    }
    return Pair(weights, newBias)

}

fun truncate(num: Double): Double {
    return String.format("%.3f", num).toDouble()
}

