#!/usr/bin/env bash
# Download every public dataset Ovi is built from, then rebuild the shipped model files.
#   bash scripts_fetch_data.sh
set -e
cd "$(dirname "$0")"
mkdir -p data/chat data/cornell data/facts
curl -sL -o data/gut.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip
(cd data && unzip -q -o gut.zip)
for f in ai botprofile computers conversations emotion food gossip greetings health history humor literature money movies politics psychology science sports trivia; do
  curl -sL -o data/chat/$f.yml https://raw.githubusercontent.com/gunthercox/chatterbot-corpus/master/chatterbot_corpus/data/english/$f.yml
done
curl -sL -o data/cornell/movie_lines.txt https://raw.githubusercontent.com/nguyentthai96/chatbots_seq2seq_tensorlayer/master/data/cornell_corpus/raw_data/movie_lines.txt
curl -sL -o data/facts/countries.csv https://raw.githubusercontent.com/dr5hn/countries-states-cities-database/master/csv/countries.csv
curl -sL -o data/facts/elements.csv https://raw.githubusercontent.com/utf/DataAnalyticsChemistry/master/datasets/periodic-table.csv
curl -sL -o data/facts/trivia.json https://raw.githubusercontent.com/lancejames221b/i2i/main/benchmarks/datasets/trivia_qa.json
python -m ovi.prep          # tokenizer + book tokens
python -m ovi.dataset       # ovi_pairs.bin
python -m ovi.fast build    # ovi_ngram.bin
echo "done. chat with: python -m ovi.brain"
