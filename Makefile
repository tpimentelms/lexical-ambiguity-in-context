LANGUAGE := en
N_CHARS_FILESYSTEM := 2
MAX_LINES := 100000
TRAIN_LINES := 100000

N_CHARS_FILESYSTEM := $(if $(filter-out $(LANGUAGE), ar),$(N_CHARS_FILESYSTEM),3)

MAX_LINES := $(if $(filter-out $(LANGUAGE), yo),$(MAX_LINES),80000)
TRAIN_LINES := $(if $(filter-out $(LANGUAGE), yo),$(TRAIN_LINES),10000)

MAX_LINES := $(if $(filter-out $(LANGUAGE), tl),$(MAX_LINES),800000)
TRAIN_LINES := $(if $(filter-out $(LANGUAGE), tl),$(TRAIN_LINES),100000)

MAX_LINES_ALL := $(shell expr $(MAX_LINES) + $(TRAIN_LINES))

DATA_DIR_BASE := ./data
DATA_DIR := $(DATA_DIR_BASE)/$(LANGUAGE)
CHECKPOINTS_DIR_BASE := ./checkpoints
CHECKPOINTS_DIR := $(CHECKPOINTS_DIR_BASE)/$(LANGUAGE)/
RESULTS_DIR_BASE := ./results
RESULTS_DIR := $(RESULTS_DIR_BASE)/$(LANGUAGE)/

WIKI_RAW_FILE := $(DATA_DIR)/src.txt
WIKI_SHUFFLED_FILE := $(DATA_DIR)/shuffled.txt
WIKI_ANALYSIS_FILE := $(DATA_DIR)/analyse.txt
WIKI_TRAIN_FILE := $(DATA_DIR)/train.txt
WIKI_WORDS_FILE := $(DATA_DIR)/tgt_words.pickle

BERT_FILE := $(CHECKPOINTS_DIR)trained_bert.pth

SURPRISAL_DIR_BASE := $(CHECKPOINTS_DIR)surprisal/
SURPRISAL_RAW_DIR := $(SURPRISAL_DIR_BASE)/raw/
SURPRISAL_MERGE_FILE := $(SURPRISAL_DIR_BASE)/bert.pickle.gz2
UNIGRAM_FILE := $(SURPRISAL_DIR_BASE)/unigram.pickle.gz2

EMB_DIR_BASE := $(CHECKPOINTS_DIR)embeddings/
EMB_RAW_DIR := $(EMB_DIR_BASE)raw/
EMB_RAW_DONE := $(EMB_DIR_BASE)raw.done.txt
EMB_MERG_DIR := $(EMB_DIR_BASE)merged/
EMB_COV_DIR := $(EMB_DIR_BASE)covariances/
EMB_VAR_FILE := $(EMB_DIR_BASE)variances.pickle.bz2

POLYSEMY_VAR_FILE := $(RESULTS_DIR)polysemy_var.tsv
POLYSEMY_COV_FILE :=  $(RESULTS_DIR)polysemy_cov.tsv
SURPRISAL_FILE :=  $(RESULTS_DIR)surprisal.tsv
FULL_VAR_FILE := $(RESULTS_DIR)surprisal_var_polysemy.tsv
FULL_COV_FILE := $(RESULTS_DIR)surprisal_cov_polysemy.tsv
# CORR_VAR_FILE := $(RESULTS_DIR)surprisal_var_polysemy.tsv
# CORR_COV_FILE := $(RESULTS_DIR)surprisal_cov_polysemy.tsv
# PLOT_FILE := $(RESULTS_DIR)$(LANGUAGE)-plot_var
# PLOT_FILE_COV := $(RESULTS_DIR)$(LANGUAGE)-plot_cov


all: get_embeddings get_surprisal
	echo "Finished training" $(LANGUAGE)

merge_results: $(FULL_COV_FILE)

merge_surprisal: $(SURPRISAL_MERGE_FILE) $(SURPRISAL_FILE)
	echo "Merged surprisals" $(LANGUAGE)

get_surprisal: $(SURPRISAL_RAW_DIR)
	echo "Got surprisal" $(LANGUAGE)

train_surprisal: $(BERT_FILE)
	echo "Trained surprisal model" $(LANGUAGE)

get_polysemy: $(POLYSEMY_COV_FILE)

merge_embeddings: $(EMB_VAR_FILE)
	echo "Merged embeddings" $(LANGUAGE)

get_embeddings: $(EMB_RAW_DONE)
	echo "Got embeddings" $(LANGUAGE)

get_data: $(WIKI_WORDS_FILE)
	echo "Got data" $(LANGUAGE)

clean:
	rm -rf $(CHECKPOINTS_DIR)
	rm $(WIKI_SHUFFLED_FILE)
	rm $(WIKI_ANALYSIS_FILE)
	rm $(WIKI_TRAIN_FILE)
	rm $(WIKI_WORDS_FILE)

$(FULL_COV_FILE): $(POLYSEMY_COV_FILE) $(SURPRISAL_FILE)
	echo 'Get correlation file'
	mkdir -p $(RESULTS_DIR)
	python src/h04_analysis/merge_polysemy_surprisal.py \
		--surprisal-file $(SURPRISAL_FILE) \
		--polysemy-covariance-file $(POLYSEMY_COV_FILE) --polysemy-variance-file $(POLYSEMY_VAR_FILE) \
		--correlation-variance-file $(FULL_VAR_FILE) --correlation-covariance-file $(FULL_COV_FILE)

$(SURPRISAL_FILE): $(UNIGRAM_FILE) $(SURPRISAL_MERGE_FILE)
	mkdir -p $(RESULTS_DIR)
	python src/h04_analysis/get_surprisal.py \
		--surprisal-bert-file $(SURPRISAL_MERGE_FILE) --unigram-probs-file $(UNIGRAM_FILE) \
		--trained-bert-file $(BERT_FILE) --language $(LANGUAGE) \
		--surprisal-file $(SURPRISAL_FILE)

$(UNIGRAM_FILE): $(WIKI_ANALYSIS_FILE) $(WIKI_WORDS_FILE)
	echo 'Get unigram file' $(UNIGRAM_FILE)
	python src/h03_bert_surprisal/get_unigram_probs.py \
		--wikipedia-tokenized-file $(WIKI_ANALYSIS_FILE) --wikipedia-words-file $(WIKI_WORDS_FILE) \
		--unigram-probs-file $(UNIGRAM_FILE)

$(SURPRISAL_MERGE_FILE):
	echo 'Merge surprisal file' $(SURPRISAL_MERGE_FILE)
	python src/h03_bert_surprisal/merge_surprisal.py \
		--surprisal-bert-path $(SURPRISAL_RAW_DIR) --surprisal-bert-file $(SURPRISAL_MERGE_FILE)

$(POLYSEMY_COV_FILE): $(EMB_VAR_FILE)
	mkdir -p $(RESULTS_DIR)
	python src/h04_analysis/get_polysemy.py --language $(LANGUAGE) \
		--embeddings-covariance-path $(EMB_COV_DIR) --embeddings-variance-file $(EMB_VAR_FILE) \
		--polysemy-variance-file $(POLYSEMY_VAR_FILE) --polysemy-covariance-file $(POLYSEMY_COV_FILE)

$(EMB_VAR_FILE): $(EMB_MERG_DIR)
	echo 'Get variance file' $(EMB_VAR_FILE)
	mkdir -p $(EMB_COV_DIR)
	python src/h02_bert_embeddings/get_embeddings_covariances.py \
		--embeddings-merged-path $(EMB_MERG_DIR) --embeddings-covariance-path $(EMB_COV_DIR) \
		--embeddings-variance-file $(EMB_VAR_FILE)

$(EMB_MERG_DIR):
	echo 'Merge embeddings' $(EMB_MERG_DIR)
	mkdir -p $(EMB_MERG_DIR)
	python src/h02_bert_embeddings/merge_embeddings_per_word.py --dump-size 20 --n-chars-filesystem $(N_CHARS_FILESYSTEM) \
		--embeddings-raw-path $(EMB_RAW_DIR)  --embeddings-merged-path $(EMB_MERG_DIR)

# Get surprisal with trained Bert on MLM per word
$(SURPRISAL_RAW_DIR): $(BERT_FILE) $(WIKI_ANALYSIS_FILE) $(WIKI_WORDS_FILE)
	mkdir -p $(SURPRISAL_RAW_DIR)
	python src/h03_bert_surprisal/get_bert_surprisal.py --dump-size 5000 --batch-size 128 \
		--wikipedia-tokenized-file $(WIKI_ANALYSIS_FILE) --wikipedia-words-file $(WIKI_WORDS_FILE) \
		--trained-bert-file $(BERT_FILE) --surprisal-bert-path $(SURPRISAL_RAW_DIR)

# Train Bert on MLM per word
$(BERT_FILE): $(WIKI_TRAIN_FILE)
	mkdir -p $(CHECKPOINTS_DIR)
	python src/h03_bert_surprisal/train_bert_surprisal.py \
		--wikipedia-train-file $(WIKI_TRAIN_FILE) --trained-bert-file $(BERT_FILE) --batch-size 8

# Get Bert embeddings per word
$(EMB_RAW_DONE): | $(WIKI_ANALYSIS_FILE) $(WIKI_WORDS_FILE)
	mkdir -p $(EMB_RAW_DIR)
	python src/h02_bert_embeddings/get_bert_embeddings.py --dump-size 5000 --batch-size 128 \
		--wikipedia-tokenized-file $(WIKI_ANALYSIS_FILE) --wikipedia-words-file $(WIKI_WORDS_FILE) --embeddings-raw-path $(EMB_RAW_DIR)
	touch $(EMB_RAW_DONE)

$(WIKI_WORDS_FILE): | $(WIKI_ANALYSIS_FILE) $(WIKI_TRAIN_FILE)
	python src/h01_data/get_target_words.py --language $(LANGUAGE) \
		--wikipedia-tokenized-file $(WIKI_ANALYSIS_FILE) --wikipedia-train-file $(WIKI_TRAIN_FILE) \
		--wikipedia-words-file $(WIKI_WORDS_FILE)

# Split Data
$(WIKI_TRAIN_FILE): | $(WIKI_SHUFFLED_FILE)
	tail -n $(TRAIN_LINES) $(WIKI_SHUFFLED_FILE) > $(WIKI_TRAIN_FILE)

# Split Data
$(WIKI_ANALYSIS_FILE): | $(WIKI_SHUFFLED_FILE)
	head -n $(MAX_LINES) $(WIKI_SHUFFLED_FILE) > $(WIKI_ANALYSIS_FILE)

# Shuffle Data
$(WIKI_SHUFFLED_FILE): | $(WIKI_RAW_FILE)
	shuf $(WIKI_RAW_FILE) -n $(MAX_LINES_ALL) -o $(WIKI_SHUFFLED_FILE)
