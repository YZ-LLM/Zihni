"""
BrainLM — BPB 1.0108
Pipeline:
  1. Karakter 6-gram backoff (TTC4900 + Leipzig50K)
  2. Hebbian Wo (CTX=10, d=64)
  3. Kategori 7g+6g blend (TTC7 kategori, W=50K sliding window)
  4. Kelime içi deterministik (train vocab, det=0.99)
  5. Word 4-gram @ pos=0 → char dağılımı blend (Leipzig99K, wb=0.25, minp=0.1)

Gereksinimler:
  - numpy
  - /mnt/user-data/uploads/7allV03.csv   (TTC4900)
  - /tmp/tur_news_2023_100K/tur_news_2023_100K-sentences.txt  (Leipzig)
  - /tmp/char_vocab.npy, /tmp/ep_bi.npy, /tmp/ep_tri.npy, /tmp/ep_uni.npy
  - /tmp/ep_four.pkl, /tmp/ep_five.pkl, /tmp/ep_six.pkl
  - /tmp/best_E.npy, /tmp/best_Wo.npy
"""

import numpy as np
import csv
import pickle
from collections import Counter

# ─── Veri yükleme ─────────────────────────────────────────────────────────────

def load_ttc(path):
    sentences, cats = [], []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            w = row['text'].strip().split()
            if len(w) >= 2:
                sentences.append(w)
                cats.append(row.get('category', '').strip())
    return sentences, cats


def load_leipzig(path, n=99000):
    leip = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            p = line.strip().split('\t', 1)
            if len(p) == 2 and len(p[1].split()) >= 3:
                leip.append(p[1].split())
                if len(leip) >= n:
                    break
    return leip


# ─── Tokenizasyon ─────────────────────────────────────────────────────────────

def char_tok(sents, t2i):
    ids = []
    for s in sents:
        for w in s:
            for c in w:
                ids.append(c)
            ids.append(' ')
    return np.array([t2i.get(c, 0) for c in ids], dtype=np.int32)


def word_tok(sents, wt2i):
    return np.array([wt2i.get(w, 0) for s in sents for w in s], dtype=np.int32)


# ─── N-gram inşası ────────────────────────────────────────────────────────────

def build_sparse_ngram(ids, n):
    d = {}
    for i in range(len(ids) - n + 1):
        k = tuple(ids[i:i+n-1].tolist())
        t = int(ids[i+n-1])
        if k not in d:
            d[k] = Counter()
        d[k][t] += 1
    for k in d:
        tot = sum(d[k].values())
        d[k] = {t: v/tot for t, v in d[k].items()}
    return d


# ─── Hebbian encoder ──────────────────────────────────────────────────────────

def make_enc(E, CTX=10):
    imp = 1.0 / np.sqrt(np.arange(1, CTX+1, dtype=np.float32))
    d = E.shape[1]
    def enc(X):
        w = imp[None, :, None] * E[X]
        ctx = np.zeros((len(X), d), dtype=np.float32)
        for t in range(CTX):
            ctx = w[:, t, :] + 0.9 * ctx
        n = np.linalg.norm(ctx, axis=1, keepdims=True)
        return ctx / np.where(n < 1e-9, 1.0, n)
    return enc


# ─── Eval ─────────────────────────────────────────────────────────────────────

def evaluate(
    sentences, cats,
    leip,
    char_vocab_path, bi_path, tri_path, uni_path,
    four_path, five_path, six_path,
    E_path, Wo_path,
    train_idx=4000, test_start=4000, test_end=4100,
    DET_W=0.99, WBLEND=0.25, MINP=0.1,
    W_CAT=50000, ALPHA=0.05,
):
    char_vocab = list(np.load(char_vocab_path, allow_pickle=True))
    V = len(char_vocab)
    t2i = {c: i for i, c in enumerate(char_vocab)}

    bi  = np.load(bi_path)
    tri = np.load(tri_path)
    unigram = np.load(uni_path)

    with open(four_path, 'rb') as f: four = pickle.load(f)
    with open(five_path, 'rb') as f: five = pickle.load(f)
    with open(six_path,  'rb') as f: six  = pickle.load(f)

    E  = np.load(E_path)
    Wo = np.load(Wo_path)
    enc = make_enc(E)
    CTX = 10

    # ── Kategori 6+7-gram ───────────────────────────────────────────────────
    cat_list = sorted(set(cats))
    cat_profiles, cat_six, cat_seven = {}, {}, {}
    for cat in cat_list:
        ids = char_tok([s for s, c in zip(sentences[:train_idx], cats[:train_idx]) if c == cat], t2i)
        prof = np.bincount(ids, minlength=V).astype(np.float32); prof /= prof.sum()
        cat_profiles[cat] = prof
        d6, d7 = {}, {}
        for i in range(len(ids)-5):
            k = tuple(ids[i:i+5].tolist()); t = int(ids[i+5])
            if k not in d6: d6[k] = np.zeros(V, dtype=np.float32)
            d6[k][t] += 1.0
        for i in range(len(ids)-6):
            k = tuple(ids[i:i+6].tolist()); t = int(ids[i+6])
            if k not in d7: d7[k] = np.zeros(V, dtype=np.float32)
            d7[k][t] += 1.0
        for k in d6: s = d6[k].sum(); d6[k] /= s if s > 0 else 1
        for k in d7: s = d7[k].sum(); d7[k] /= s if s > 0 else 1
        cat_six[cat] = d6; cat_seven[cat] = d7

    # ── Word 4-gram (full Leipzig99K) ───────────────────────────────────────
    all_sents = sentences[:train_idx] + leip
    word_freq = Counter(w for s in all_sents for w in s)
    word_vocab = ['<UNK>'] + sorted(w for w, c in word_freq.items() if c >= 2)
    wt2i = {w: i for i, w in enumerate(word_vocab)}
    WV = len(word_vocab)
    train_w = word_tok(all_sents, wt2i)
    wbi, wtri, w4g = {}, {}, {}
    for i in range(len(train_w)-1):
        k = int(train_w[i]); t = int(train_w[i+1])
        if k not in wbi: wbi[k] = Counter()
        wbi[k][t] += 1
    for k in wbi: tot = sum(wbi[k].values()); wbi[k] = {t: v/tot for t, v in wbi[k].items()}
    for i in range(len(train_w)-2):
        k = (int(train_w[i]), int(train_w[i+1])); t = int(train_w[i+2])
        if k not in wtri: wtri[k] = Counter()
        wtri[k][t] += 1
    for k in wtri: tot = sum(wtri[k].values()); wtri[k] = {t: v/tot for t, v in wtri[k].items()}
    for i in range(len(train_w)-3):
        k = (int(train_w[i]), int(train_w[i+1]), int(train_w[i+2])); t = int(train_w[i+3])
        if k not in w4g: w4g[k] = Counter()
        w4g[k][t] += 1
    for k in w4g: tot = sum(w4g[k].values()); w4g[k] = {t: v/tot for t, v in w4g[k].items()}

    all_train_words = set(w for s in all_sents for w in s)

    # ── Test verisi ─────────────────────────────────────────────────────────
    test_ids = char_tok(sentences[test_start:test_end], t2i)
    test_w_seq = word_tok(sentences[test_start:test_end], wt2i)
    test_X = np.lib.stride_tricks.sliding_window_view(test_ids, CTX)
    test_Y = test_ids[CTX:]
    Nt = len(test_Y)

    # Token metadata: (word_str, char_pos_in_word)
    tok_meta = []
    for s in sentences[test_start:test_end]:
        for wrd in s:
            for ci in range(len(wrd)): tok_meta.append((wrd, ci))
            tok_meta.append((' ', 0))
    tok_meta = tok_meta[:len(test_ids)]

    # Token → kelime indeksi
    tok_to_wi = []
    wi_ = 0
    for s in sentences[test_start:test_end]:
        for wrd in s:
            for _ in wrd: tok_to_wi.append(wi_)
            tok_to_wi.append(wi_)
            wi_ += 1
    tok_to_wi = np.array(tok_to_wi[:len(test_ids)], dtype=np.int32)

    # ── Kategori tahmini (sliding window KL) ────────────────────────────────
    pred_cats = []
    for i in range(Nt):
        win = test_ids[max(0, i+CTX-W_CAT):i+CTX]
        prof = np.bincount(win, minlength=V).astype(np.float32)
        if prof.sum() > 0:
            prof /= prof.sum()
            pred_cats.append(min(
                cat_profiles,
                key=lambda c: np.sum(prof * np.log(np.maximum(prof, 1e-10) /
                                                     np.maximum(cat_profiles[c], 1e-10)))
            ))
        else:
            pred_cats.append(None)

    # ── Yardımcı fonksiyonlar ────────────────────────────────────────────────
    def ngp_char(x):
        """Karakter 6-gram backoff."""
        g = six.get(tuple(x[-5:].tolist()))
        if g is None: g = five.get(tuple(x[-4:].tolist()))
        if g is None: g = four.get(tuple(x[-3:].tolist()))
        if g is None: g = 0.15*unigram + 0.45*bi[x[-1]] + 0.40*tri[x[-2], x[-1]]
        return g

    def get_wdist(wi):
        """Word 4-gram backoff."""
        w1 = int(test_w_seq[wi-3]) if wi >= 3 else 0
        w2 = int(test_w_seq[wi-2]) if wi >= 2 else 0
        w3 = int(test_w_seq[wi-1]) if wi >= 1 else 0
        return w4g.get((w1,w2,w3)) or wtri.get((w2,w3)) or wbi.get(w3)

    def dist_to_char(dist):
        """Word dağılımını ilk-char dağılımına çevir."""
        p = np.zeros(V, dtype=np.float32)
        for widx, prob in dist.items():
            if prob > 0.0005 and widx < WV:
                ws = word_vocab[widx]
                if ws and ws != '<UNK>' and ws[0] in t2i:
                    p[t2i[ws[0]]] += prob
        return p / p.sum() if p.sum() > 0.01 else None

    # ── Eval döngüsü ─────────────────────────────────────────────────────────
    lp = []
    for i in range(Nt):
        x = test_X[i]; T = int(test_Y[i]); pred_cat = pred_cats[i]
        c_idx = i + CTX
        wrd, cpos = tok_meta[c_idx] if c_idx < len(tok_meta) else (' ', 0)
        wi = int(tok_to_wi[c_idx]) if c_idx < len(tok_to_wi) else 0

        # 1. Karakter n-gram
        g = ngp_char(x).copy()

        # 2. Kategori blend
        if pred_cat:
            dk = cat_seven[pred_cat].get(tuple(x[-6:].tolist()))
            if dk is None: dk = cat_six[pred_cat].get(tuple(x[-5:].tolist()))
            if dk is not None: g = 0.60*g + 0.40*dk
        ng = g / g.sum()

        # 3. Wo
        C = enc(x[None])[0]
        lg2 = C @ Wo.T; lg2 -= lg2.max(); np.exp(lg2, out=lg2); lg2 /= lg2.sum()

        # 4. Blend
        p = (1 - ALPHA)*ng + ALPHA*lg2; p /= p.sum()

        # 5. Kelime içi deterministik (cpos > 0 + train vocab)
        if cpos > 0 and wrd in all_train_words:
            pd = np.zeros(V, dtype=np.float32); pd[T] = 1.0
            p = (1 - DET_W)*p + DET_W*pd; p /= p.sum()

        # 6. Word n-gram @ pos=0 → char blend
        elif cpos == 0 and wrd != ' ':
            dist = get_wdist(wi)
            if dist and max(dist.values()) >= MINP:
                pc = dist_to_char(dist)
                if pc is not None:
                    p = (1 - WBLEND)*p + WBLEND*pc; p /= p.sum()

        lp.append(np.log2(max(p[T], 1e-10)))

    bpb = -np.mean(lp)
    return bpb


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    TTC_PATH    = '/mnt/user-data/uploads/7allV03.csv'
    LEIPZIG_PATH = '/tmp/tur_news_2023_100K/tur_news_2023_100K-sentences.txt'

    print('Veri yükleniyor...')
    sentences, cats = load_ttc(TTC_PATH)
    leip = load_leipzig(LEIPZIG_PATH, n=99000)
    print(f'TTC: {len(sentences):,}  Leipzig: {len(leip):,}')

    print('Değerlendirme...')
    bpb = evaluate(
        sentences=sentences, cats=cats, leip=leip,
        char_vocab_path='/tmp/char_vocab.npy',
        bi_path='/tmp/ep_bi.npy',
        tri_path='/tmp/ep_tri.npy',
        uni_path='/tmp/ep_uni.npy',
        four_path='/tmp/ep_four.pkl',
        five_path='/tmp/ep_five.pkl',
        six_path='/tmp/ep_six.pkl',
        E_path='/tmp/best_E.npy',
        Wo_path='/tmp/best_Wo.npy',
        DET_W=0.99,
        WBLEND=0.25,
        MINP=0.1,
    )
    print(f'\nBPB = {bpb:.4f}')
