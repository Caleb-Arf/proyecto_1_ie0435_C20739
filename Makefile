PYTHON = python3
PREPROCESSING = src/image_preprocessor.py
VECTORING = src/vectoring.py
TRAINING_SET = src/generate_csv.py
TESTING_SET = src/testing_set.py

image_preprocessing:
	$(PYTHON) $(PREPROCESSING)

vectoring:
	$(PYTHON) $(VECTORING)

trainig_set:
	$(PYTHON) $(TRAINING_SET)

testing_set:
	$(PYTHON) $(TESTING_SET)

clean:
	rm -rf resized_images image_vectors training_set testing_set