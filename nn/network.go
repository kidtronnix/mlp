package nn

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Config struct {
	Epochs    int
	BatchSize int
	Eta       float64
}

type MLP struct {
	numLayers int
	sizes     []int
	biases    []*mat.Dense
	weights   []*mat.Dense
	config    Config
}

func New(c Config, sizes ...int) *MLP {

	// generate some random weights and biases
	bs := []*mat.Dense{}
	ws := []*mat.Dense{}

	// len of slices we will make
	// don't need any biases for input layer
	// don't need any weights for output layer
	l := len(sizes) - 1

	for j := 0; j < l; j++ {
		y := sizes[1:][j] // y starts from layer after input layer to output layer
		x := sizes[:l][j] // x starts from input layer to layer before output layer

		// make a random init biases matrix of y*1
		b := make([]float64, y)
		for i := range b {
			b[i] = rand.NormFloat64()
		}
		bs = append(bs, mat.NewDense(y, 1, b))

		// make a random init weights matrix of y*x
		w := make([]float64, y*x)
		for i := range w {
			w[i] = rand.NormFloat64()
		}
		ws = append(ws, mat.NewDense(x, y, w)) // P:changed the order of row and column

	}

	return &MLP{
		numLayers: len(sizes),
		sizes:     sizes,
		biases:    bs,
		weights:   ws,
		config:    c,
	}
}
