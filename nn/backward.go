package nn

import "gonum.org/v1/gonum/mat"

func (n *MLP) backward(x, y mat.Matrix) {

	// get activations
	as, zs := n.forward(x)

	// final z
	z := zs[len(zs)-1]
	out := as[len(as)-1]

	// error
	err := new(mat.Dense)

	err.Sub(out, y)

	// delta of last layer
	// delta = (out - y).sigmoidprime(last_z)
	sp := new(mat.Dense)
	sp.Apply(applySigmoidprime, z)

	delta := new(mat.Dense)
	delta.MulElem(err, sp)

	// prop delta through layers

	nbs := make([]*mat.Dense, len(n.weights))
	nws := make([]*mat.Dense, len(n.weights))

	nbs[len(nbs)-1] = delta

	a := as[len(as)-2]

	nw := new(mat.Dense)
	nw.Mul(a.T(), delta)
	nws[len(nws)-1] = nw

	// go back through layers
	for i := n.numLayers - 2; i > 0; i-- {
		z := zs[i-1] // -1?

		sp := new(mat.Dense)
		sp.Apply(applySigmoidprime, z)

		wdelta := new(mat.Dense)
		w := n.weights[i]

		wdelta.Mul(delta, w.T())

		nextdelta := new(mat.Dense)
		nextdelta.MulElem(wdelta, sp)
		delta = nextdelta

		nbs[i-1] = delta

		a := as[i-1]
		nw := new(mat.Dense)
		nw.Mul(a.T(), delta)
		nws[i-1] = nw
	}

	N, _ := x.Dims()

	weights := make([]*mat.Dense, len(n.weights))
	biases := make([]*mat.Dense, len(n.biases))

	for i := 0; i < len(n.weights); i++ {
		w := n.weights[i]
		nw := nws[i]

		b := n.biases[i]
		nb := sumCols(nbs[i]).T()

		// w' = w - (eta/N) * nw

		alpha := n.config.Eta / float64(N)
		scalednw := new(mat.Dense)
		scalednw.Scale(alpha, nw)

		scalednb := new(mat.Dense)
		scalednb.Scale(alpha, nb)

		wprime := new(mat.Dense)
		wprime.Sub(w, scalednw)

		bprime := new(mat.Dense)
		bprime.Sub(b, scalednb)

		weights[i] = wprime
		biases[i] = bprime

	}

	n.weights = weights
	n.biases = biases
}
