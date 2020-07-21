var arrayLoss: MutableList<Double> = mutableListOf<Double>()
var iteraciones = 0
var lr = 0.2
var min = listOf(140.0, 44.0,60.0)
var max = listOf(198.0, 90.0,300.0)
var validationTarget= ArrayList<Int>()
fun main() {

    // datos de entrenamiento
    var i1 = listOf(175.0,60.0,273.0)
    var i2 = listOf(160.0,75.0,235.0)
    var i3 = listOf(180.0,63.0,171.0)
    var i4 = listOf(150.0,90.0,113.0)
    var i5 = listOf(178.0,70.0,126.0)
    var i6 = listOf(140.0,88.0,165.0)
    var i7 = listOf(198.0,65.0,227.0)



    val rawData: List<List<Double>> = listOf(i1, i2, i3, i4, i5, i6,i7)


    var bias = 1.0
    var weights = listOf(0.0, 0.0,0.0)
    var target = listOf(0, 0, 1, 1, 1, 1,0)

    var finalData = generateNormalaizeData(rawData, min, max)

    populateLoss(target.size)
    var result = initCalculate(finalData, weights, bias, lr, target)
    println("Y = $result")
    println("---------------Validacion del aprendizaje --------------------------- ")


    //datos de validacion

    var v1 = listOf(155.0,84.0,78.0)
    var v2 = listOf(173.0,55.0,85.0)
    var v3 = listOf(169.0,44.0,219.0)
    var v4 = listOf(168.0,70.0,160.0)
    var v5 = listOf(167.0,84.0,92.0)
    var v6 = listOf(170.0,66.0,172.0)
    var v7 = listOf(169.0,44.0,195.0)

    val validationData: List<List<Double>> = listOf(v1, v2, v3, v4, v5, v6,v7)

    var finalWeights = result.first
    var finalBias = result.second

     var finalDataValidation = generateNormalaizeData(validationData, min, max)

    validateData(finalDataValidation,finalWeights,finalBias)


}



fun validateData(features: List<List<Double>>, initialWeight: List<Double>, initialBias: Double) {

    for (feature in features){
        var sumatory = perceptronMainFunction(feature, initialWeight, initialBias)
        var hardLimit = hardLimit(sumatory)
        validationTarget.add(hardLimit)

    }
    println("result Target from data validation  ${validationTarget}")
    var womenQuantity = validationTarget.count { it.equals(0) }
    var menQuantity = validationTarget.count { it.equals(1) }

    println("cantidad mujeres = ${womenQuantity} \n cantidad Hombres = ${menQuantity} \n" +
            "${if (menQuantity > womenQuantity) "la mayoria es masculina" else "lamayoria no es masculina"}")




}

// genera la informacion normalizada (Normalizacion min-max)
fun generateNormalaizeData(rawData: List<List<Double>>, xmin: List<Double>, xmax: List<Double>): List<MutableList<Double>> {
    var arrayAllData: MutableList<MutableList<Double>> = mutableListOf<MutableList<Double>>()
    for (data in rawData) {
        arrayAllData.add(normalizeFeature(xmin, xmax, data) as MutableList<Double>)
    }
    return arrayAllData.toList()
}


fun minMax(x: Double, xmin: Double, xmax: Double): Double {
    return (x - xmin) / (xmax - xmin)
}

//normaliza el array de datos de un solo feature
fun normalizeFeature(xmin: List<Double>, xmax: List<Double>, data: List<Double>): List<Double> {
    val normalizeList = (0 until data.size).map { minMax(data[it], xmin[it.toInt()], xmax[it.toInt()]) }
    return normalizeList

}

// Σ(weight_i *x_i) - bias
fun perceptronMainFunction(features: List<Double>, weights: List<Double>, bias: Double): Double {
    val sums = (0 until features.size).map { features[it] * weights[it] }
    return sums.sum() - bias
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

