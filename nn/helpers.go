package nn

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func sumCols(m *mat.Dense) *mat.Dense {

	_, c := m.Dims()

	var output *mat.Dense

	data := make([]float64, c)
	for i := 0; i < c; i++ {
		col := mat.Col(nil, i, m)
		data[i] = floats.Sum(col)
	}
	output = mat.NewDense(1, c, data)

	return output
}

// sigmoid is an elementwise func
// this applied over every element
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func applySigmoid(_, _ int, v float64) float64 {
	return sigmoid(v)
}

func sigmoidprime(x float64) float64 {
	return sigmoid(x) * (1 - sigmoid(x))
}

func applySigmoidprime(_, _ int, v float64) float64 {
	return sigmoidprime(v)
}
