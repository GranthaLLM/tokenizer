import pickle
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000

cnt = pickle.load(open("data/run_counts.pkl", "rb"))
done = pickle.load(open("data/teacher_labels.pkl", "rb"))
words = [w for w, _ in cnt.most_common() if len(w) > 3][:N]
words = [w for w in words if w not in done]
open("data/resolve_words.txt", "w", encoding="utf-8").write("\n".join(words) + "\n")
print(len(words), "words ->", "data/resolve_words.txt")
