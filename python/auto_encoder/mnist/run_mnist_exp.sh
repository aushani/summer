#/bin/sh
python run_mnist_ae_first.py
#python run_mnist_classifier_first.py
python run_mnist_both.py -c 0.01
python run_mnist_both.py -c 0.02
python run_mnist_both.py -c 0.05
python run_mnist_both.py -c 0.1
python run_mnist_both.py -c 0.2
python run_mnist_both.py -c 0.5
python run_mnist_both.py -c 1.0
