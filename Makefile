PYTHON = python3
PREPROCESSING = src/image_preprocessor.py
VECTORING = src/vectoring.py
TRAINING_SET = src/generate_csv.py
TESTING_SET = src/testing_set.py
ALGORITHM_TUNING = src/model_training.py
MODEL_EXAMPLE = src/model_test.py

image_preprocessing:
	$(PYTHON) $(PREPROCESSING)

vectoring:
	$(PYTHON) $(VECTORING)

trainig_set:
	$(PYTHON) $(TRAINING_SET)

testing_set:
	$(PYTHON) $(TESTING_SET)

train_models:
	$(PYTHON) $(ALGORITHM_TUNING)

test_models:
	$(PYTHON) $(MODEL_EXAMPLE)

full_training: image_preprocessing vectoring trainig_set testing_set train_models

clean:
	rm -rf resized_images image_vectors training_set testing_set models