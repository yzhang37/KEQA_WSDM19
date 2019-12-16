#!/usr/bin/env bash
set -o errexit

# in this shell, no files needed to be downloaded.
# the files should exist at .downloads/

# check if SimpleQuestions dataset exist.
if [ ! -f ".downloads/data.zip" ]; then
    echo "Please download SimpleQuestions data, and put data.zip inside .downloads/."
    exit 1
fi
# check if KGembed.zip exist.
if [ ! -f ".downloads/KGembed.zip" ]; then
    echo "Please download Knowledge Graph Embedding data, and put KGembed.zip inside .downloads/."
    exit 1
fi

if [ -d "./data" ]; then
    echo "Existing ./data, unzip will pass."
    echo ""
else
    unzip .downloads/data.zip -d .
fi

echo "Preprocess the raw data"
python3.6 trim_names.py -f data/freebase-FB2M.txt -n data/FB5M.name.txt

echo "Create processed, augmented dataset...\n"
python3.6 augment_process_dataset.py -d data/


echo "Embed the Knowledge Graph:\n"
if [ -d "./KGembed" ]; then
    echo "Existing ./KGembed, unzip will pass."
    echo ""
else
    unzip .downloads/KGembed.zip -d .
    mv -f KGembed/* preprocess/
fi

#python3.6 transE_emb.py --learning_rate 0.003 --batch_size 3000 --eval_freq 50

echo "We could runn train_detection.py, train_entity.py, train_pred.py simultaneously"

echo "Head Entity Detection (HED) model, train and test the model..."
python3.6 train_detection.py --entity_detection_mode LSTM --fix_embed --gpu 0

echo "Entity representation learning..."
python3.6 train_entity.py --qa_mode GRU --fix_embed --gpu 0
python3.6 train_pred.py --qa_mode GRU --fix_embed --gpu 0

echo "We have to run train_detection.py, train_entity.py, train_pred.py first, before running test_main.py..."
python3.6 test_main.py --gpu 0
