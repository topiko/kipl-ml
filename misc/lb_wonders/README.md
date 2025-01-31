There is a discreparency between our local results and the once they obtain in [Laserbeak](https://github.com/notem/Laserbeak-WF-Classifier) paper. However, these are most likely caused by few minute deviations from proper practices the authors in LB have chosen...

1. The train and test set are overlapping (though only by ~ 1 % in the testset)... However, since the models are (for undefended case) more or less perfect in the train set this overlap increase e.g., accuracy by approx 1 % UNIT...
2. Despite declaring 10 fold xv, I can only find a subset of traces in the "full" testset.

Both of the above issues are manifested by running:

`uv run python gen_xv_lb.py`

You'll though need to make sure to modify the root dir in the script and make it point into the directory where you have the laserbeak datas.

I have verified they at least have the same traces as our "bigenough" dataset. To verify run:

`uv run python test_laserbeak_data.py`


I have also verified we have same preprocessing:

`uv run python test_prep_pipes.py`


