package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strconv"
)

func main() {
	xCols := 4

	// out, err := os.Create("data.csv")
	resp, err := http.Get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
	if err != nil {
		panic("Error downloading data!")
	}

	lines := []string{}

	r := csv.NewReader(resp.Body)
	for {
		fields, err := r.Read()
		if err == io.EOF {
			break
		}

		if err != nil {
			panic("Error reading from csv! " + err.Error())
		}

		line := ""
		for i, f := range fields {
			if i < xCols {
				v := parseX(f)
				m, sd := colStats(i)
				v = normalize(v, m, sd)

				line += fmt.Sprintf("%0.5f,", v)
			} else {
				v := parseY(f)
				line += fmt.Sprintf("%s\n", v)
				lines = append(lines, line)
			}
		}

	}

	train, err := os.Create("train.csv")
	if err != nil {
		panic("Error creating out file!")
	}
	defer train.Close()

	test, err := os.Create("test.csv")
	if err != nil {
		panic("Error creating out file!")
	}
	defer test.Close()

	counts := make(map[int]int, 0)

	for _, i := range rand.Perm(len(lines)) {

		var c int
		if i < 50 {
			c = 0
		} else if i > 100 {
			c = 2
		} else {
			c = 1
		}

		if counts[c] < 33 {
			train.WriteString(lines[i])
		} else {
			test.WriteString(lines[i])
		}

		counts[c]++
	}

	train.Sync()
	test.Sync()
}

/*
Summary Statistics:
	             Min  Max   Mean  SD      Class Correlation
   sepal length: 4.3  7.9   5.84  0.83    0.7826
    sepal width: 2.0  4.4   3.05  0.43   -0.4194
   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
*/

var means = []float64{5.84, 3.05, 3.76, 1.2}
var sds = []float64{0.83, 0.43, 1.76, 0.76}

func colStats(i int) (mean float64, sd float64) {
	return means[i], sds[i]
}

func parseX(s string) float64 {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		panic("Error parsing float!")
	}

	return f
}

func parseY(s string) string {

	if s == "Iris-setosa" {
		return "1.0,0.0,0.0"
	}

	if s == "Iris-versicolor" {
		return "0.0,1.0,0.0"
	}

	if s == "Iris-virginica" {
		return "0.0,0.0,1.0"
	}

	panic("Unkown iris species used as label!")
}

func normalize(v, mean, sd float64) float64 {
	return (v - mean) / sd
}
