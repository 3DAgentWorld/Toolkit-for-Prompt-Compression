CONFIG ?= config/example.json
DEVICE ?= cpu
MODELDIR ?= models/newsroom-P75/model-dirs/best_val_reward-7950
TESTSET ?= data/test-data/broadcast.jsonl
HC_OUTPUT ?= data/hc-outputs/hc.L11.google.jsonl

# TRAINING

.PHONY: train
train:
	python bin/train.py --verbose --config $(CONFIG) --device $(DEVICE)

# EVALUATING SCRL MODELS (predict + evaluate)

.PHONY: eval-google
eval-google:
	python bin/evaluate.py \
		--model-dir $(MODELDIR) \
		--device $(DEVICE) \
		--dataset data/test-data/google.jsonl


.PHONY: eval-duc2004
eval-duc2004:
	python bin/evaluate.py \
		--model-dir $(MODELDIR) \
		--device $(DEVICE) \
		--dataset data/test-data/duc2004.jsonl \
		--max-chars 75


.PHONY: eval-gigaword
eval-gigaword:
	python bin/evaluate.py \
		--model-dir $(MODELDIR) \
		--device $(DEVICE) \
		--dataset data/test-data/gigaword.jsonl \
		--pretokenized


.PHONY: eval-broadcast
eval-broadcast:
	python bin/evaluate.py \
		--model-dir $(MODELDIR) \
		--device $(DEVICE) \
		--dataset data/test-data/broadcast.jsonl \
		--pretokenized


.PHONY: eval-bnc
eval-bnc:
	python bin/evaluate.py \
		--model-dir $(MODELDIR) \
		--device $(DEVICE) \
		--dataset data/test-data/bnc.jsonl \
		--pretokenized


# EVALUATE HILL CLIMBING SEARCH

.PHONY: hc-eval-google
hc-eval-google:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/google.jsonl \
    	--outputs $(HC_OUTPUT)


.PHONY: hc-eval-duc2004
hc-eval-duc2004:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/duc2004.jsonl \
	    --outputs $(HC_OUTPUT)


.PHONY: hc-eval-gigaword
hc-eval-gigaword:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/gigaword.jsonl \
	    --outputs $(HC_OUTPUT)


.PHONY: hc-eval-broadcast
hc-eval-broadcast:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/broadcast.jsonl \
	    --outputs $(HC_OUTPUT)


.PHONY: hc-eval-bnc
hc-eval-bnc:
	python bin/evaluate_hc_output.py \
	    --dataset data/test-data/bnc.jsonl \
	    --outputs $(HC_OUTPUT)
