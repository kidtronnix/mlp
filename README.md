# mlp

This is an example implementation of a Mult-Layered Perceptron. Written in Golang.

The code is used in this educational article here.

## usage

To run the mlp...

```bash
git clone https://github.com/kidtronnix/mlp.git
cd mlp
go build
./mlp -train="data/train.csv" -test="data/test.csv" -eta=0.1 -epochs=3000 -arch="4,3,3"
```

To generate the data from source...

```
# from root of repo
cd data/
go run main.go
# this generates 'train.csv' and 'test.csv'
```
