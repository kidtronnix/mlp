package nn

import (
	"gonum.org/v1/gonum/mat"
)

func (n *MLP) Evaluate(x, y mat.Matrix) float64 {

	p := n.Predict(x)
	N, _ := p.Dims()

	var correct int
	for n := 0; n < N; n++ {

		ry := mat.Row(nil, n, y)
		truth := oneHotDecode(ry)

		rp := mat.Row(nil, n, p)
		predicted := prediction(rp)

		if predicted == truth {
			correct++
		}
	}

	accuracy := float64(correct) / float64(N)

	return accuracy * 100
}

// get prediction as max prob in row
func prediction(vs []float64) int {
	var max float64
	var p int
	for i, v := range vs {
		if v > max {
			p = i
			max = v
		}
	}

	return p
}

// get index of category
func oneHotDecode(vs []float64) int {
	for i, v := range vs {
		if v == 1.0 {
			return i
		}
	}

	panic("Unable to one hot decode as empty array given.")
}
