package nn

import "gonum.org/v1/gonum/mat"

func (n *MLP) Predict(x mat.Matrix) mat.Matrix {

	as, _ := n.forward(x)

	return as[len(as)-1]
}
