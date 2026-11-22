"""Download-free prep: build tokenizer + token arrays from data/ books and the seed chat."""
import glob, numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from .seed_chat import chat_corpus, USER, BOT, END

BOOKS = ["austen-*", "bryant-*", "burgess-*", "carroll-*", "chesterton-*", "edgeworth-*", "melville-*"]


def books_text():
    t = []
    for pat in BOOKS:
        for f in sorted(glob.glob(f"data/gutenberg/{pat}.txt")):
            t.append(open(f, encoding="latin-1").read())
    s = "\n".join(t)
    s = s.replace("\r", "")
    return "".join(ch for ch in s if ch == "\n" or 32 <= ord(ch) < 127)


def main(vocab=4096):
    books = books_text()
    chat = chat_corpus()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[USER, BOT, END],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([books, chat], tr)
    tok.save("data/tokenizer.json")
    np.save("data/books.npy", np.array(tok.encode(books).ids, dtype=np.uint16))
    print("books tokens:", len(tok.encode(books).ids), "chars:", len(books))


if __name__ == "__main__":
    main()
