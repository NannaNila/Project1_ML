# Heart Disease Classification — ML Project 1

NumPy implementations of six linear methods and two SVM variants for predicting heart disease from BRFSS survey data. The selected weighted squared-hinge SVM achieves local test F1 **0.42672**; methods and experiments are described in `report.tex`.

Place `x_train.csv`, `y_train.csv`, and `x_test.csv` in `data/dataset/`, then run (requires Python and NumPy):

```bash
python run.py --model svm_squared_hinge --lambda 0.001 --positive-weight 4 --max-missing 0.95 --svm-iters 20000000 --export --output-dir output/report_reproduction
```

This reproduces the 80/20 split and validation-based threshold selection, saving metrics, model weights, and `submission.csv` under `output/report_reproduction/svm_squared_hinge/`.
