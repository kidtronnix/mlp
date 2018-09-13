package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/elementai/mlp/nn"
	"gonum.org/v1/gonum/mat"
)

// ints is a type used as a flag var
type ints []int

func (i *ints) String() string {
	return fmt.Sprintf("%d", *i)
}

func (i *ints) Set(s string) error {
	vals := strings.Split(s, ",")
	for _, v := range vals {

		tmp, err := strconv.Atoi(v)
		if err != nil {
			return err
		}
		*i = append(*i, tmp)
	}
	return nil
}

var (
	epochs    int
	eta       float64
	batchSize int
	arch      = ints{}
	train     string
	test      string
)

func main() {

	// args
	flag.IntVar(&epochs, "epochs", 1000, "Number of training epochs")
	flag.Float64Var(&eta, "eta", 0.3, "Learning rate")
	flag.IntVar(&batchSize, "batch", 32, "Size of training batch")
	flag.Var(&arch, "arch", "Architecture of neurons (e.g. 5,3,2)")
	flag.StringVar(&train, "train", "train.csv", "Path of training csv")
	flag.StringVar(&test, "test", "test.csv", "Path of testing csv")
	flag.Parse()

	if len(arch) < 2 {
		panic("A nn must have 2 layers at minimum (input and output)!")
	}

	i := arch[0]
	o := arch[len(arch)-1]

	con := nn.Config{
		Epochs:    epochs,
		Eta:       eta,
		BatchSize: batchSize,
	}

	// make a new network
	n := nn.New(con, arch...)

	// train
	x, y := load(train, i, o)

	n.Train(x, y)

	// evaluate
	x, y = load(test, i, o)
	accuracy := n.Evaluate(x, y)

	fmt.Printf("accuracy = %0.1f%%\n", accuracy)
}

func load(path string, xFields, yFields int) (*mat.Dense, *mat.Dense) {

	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	r := csv.NewReader(f)

	x := []float64{}
	y := []float64{}

	n := 0
	for {
		fields, err := r.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic("Error parsing csv: " + err.Error())
		}

		for i, v := range fields {

			fl := string2float64(v)

			if i < xFields {
				x = append(x, fl)
			} else {
				y = append(y, fl)
			}
		}
		n++
	}

	return mat.NewDense(n, xFields, x), mat.NewDense(n, yFields, y)
}

func string2float64(v string) float64 {

	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		panic("Error parsing field as float!")
	}

	return f

}
