PYTHON = python3
PREPROCESSING = src/image_preprocessor.py
VECTORING = src/vectoring.py
CSV = src/generate_csv.py

image_preprocessing:
	$(PYTHON) $(PREPROCESSING)

vectoring:
	$(PYTHON) $(VECTORING)

csv:
	$(PYTHON) $(CSV)

clean:
	rm -rf resized_images image_vectors vectors.csv