package nn

import (
	"gonum.org/v1/gonum/mat"
)

func (n *MLP) Train(x, y *mat.Dense) {

	r, cx := x.Dims()
	_, cy := y.Dims()

	b := n.config.BatchSize

	for e := 1; e < n.config.Epochs+1; e++ {

		for i := 0; i < r; i += b {
			k := i + b
			if k > r {
				k = r
			}
			_x := x.Slice(i, k, 0, cx)
			_y := y.Slice(i, k, 0, cy)

			n.backward(_x, _y)

		}
	}
}
