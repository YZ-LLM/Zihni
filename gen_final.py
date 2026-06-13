import numpy as np, re, time
from collections import Counter, defaultdict

exec(open('/tmp/final_v2.py').read().split("# ── TEST")[0])
exec(open('/tmp/morph_gen_v4.py').read().split("if __name__ == '__main__':")[0])
gen_A = generate

TR_CHARS = set('abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ\'')
is_tr = lambda w: len(w)>=3 and all(c in TR_CHARS for c in w)

BAD_VERBS_NEG = {'değil','olma','çalışma','bulun','yapıl','edilme',
                 'olmadı','değildi','yoktu','gerek','lazım'}

def is_valid_fiil(r):
    if not r or len(r)<3: return False
    if any(r.endswith(x) for x in ['yor','iyor','uyor','üyor']): return False
    if r in BAD_VERBS_NEG: return False
    return True

ALL_YRD_V2 = {f for f in ALL_YRD if not any(x in f for x in
    ['iyor','uyor','üyor','ıyor','görüy','biliy','yapıy','oluy','diy','alıy'])}

CORPUS = [
    '/tmp/tur_news_2023_100K/tur_news_2023_100K-sentences.txt',
    '/tmp/tur_news_2020_1M/tur_news_2020_1M-sentences.txt',
]

t0=time.time()
fi_cnt=Counter(); dg_cnt=Counter()
S_DI=['ti','tu','tü','tı','di','du','dü','dı','miş','muş','müş','mış']
S_DG=['tiğini','tuğunu','tüğünü','tığını','diğini','duğunu','düğünü','dığını',
      'mesini','masını','eceğini','acağını']
fn=defaultdict(Counter); cooc=defaultdict(Counter); wc=Counter(); n=0

for path in CORPUS:
    with open(path) as f:
        for line in f:
            p=line.strip().split('\t',1)
            if len(p)<2: continue
            words=p[1].strip().split()
            if len(words)<3 or len(words)>18: continue
            n+=1
            wl=[w.lower().rstrip('.,;:!?"\'').lstrip('"\'') for w in words]
            nr=[]
            for w in wl:
                r=nkoku(w)
                if r and valid_nesne(r) and is_tr(r): wc[r]+=1; nr.append(r)
            for i,r in enumerate(nr):
                for j in range(max(0,i-5),min(len(nr),i+6)):
                    if i!=j: cooc[r][nr[j]]+=1
            for w in wl:
                for s in sorted(S_DI+S_DG,key=len,reverse=True):
                    if w.endswith(s) and len(w)-len(s)>=3:
                        r=w[:-len(s)]
                        if sum(1 for c in r if c in 'aeıiouüö')>=1 and is_tr(r):
                            if s in S_DI: fi_cnt[r]+=1
                            if s in S_DG: dg_cnt[r]+=1
                            break
            for i,w in enumerate(wl):
                r=fkoku(w)
                if r and is_tr(r) and is_valid_fiil(r):
                    if not ((fi_cnt.get(r,0)>=5 and r in dg_cnt and dg_cnt.get(r,0)>=2) or fi_cnt.get(r,0)>=50): continue
                    for j in range(max(0,i-5),i):
                        nj=nkoku(wl[j])
                        if nj and valid_nesne(nj) and is_tr(nj): fn[r][nj]+=1
            for i,w in enumerate(wl):
                if w not in ALL_YRD_V2: continue
                tip=YARDIMCI[w]
                for di in range(1,4):
                    if i-di<0: break
                    prev=wl[i-di]
                    if fkoku(prev): continue
                    pr=nkoku(prev)
                    if not pr or not valid_nesne(pr) or not is_tr(pr): continue
                    bk=pr+'_'+tip
                    for j in range(max(0,i-di-4),i-di):
                        nj=nkoku(wl[j])
                        if nj and valid_nesne(nj) and is_tr(nj): fn[bk][nj]+=1; fn[pr][nj]+=1
                    break

WL={r for r in fi_cnt if (fi_cnt[r]>=5 and r in dg_cnt and dg_cnt[r]>=2) or fi_cnt[r]>=50}
WL.update(['belirt','çek','al','ver','git','gel','bak','yap','bil','kal','geç','dur','çık'])
WL={r for r in WL if is_valid_fiil(r)}
LOW={r for r,c in wc.items() if c<5}
for f2 in list(fn.keys()):
    for nm in list(fn[f2].keys()):
        if nm in LOW: del fn[f2][nm]
total=max(sum(wc.values()),1)
print(f'Öğrenildi {time.time()-t0:.0f}s — {n:,} cümle')

def ppmi(a,b):
    cnt=cooc.get(a,{}).get(b,0)
    if cnt<3: return 0.0
    return max(0.0,float(np.log(cnt/total/(wc.get(a,1)/total*wc.get(b,1)/total+1e-10))))

def cluster(pr):
    c=Counter()
    for p in pr:
        for w,cnt in cooc.get(p,{}).items():
            if valid_nesne(w) and is_tr(w) and w not in LOW: c[w]+=cnt
    return {w for w,_ in c.most_common(60)}

def gen_B(pw, n_sents=3, seed=42):
    rng=np.random.default_rng(seed)
    subj=pw[0]
    pr=[nkoku(p.lower()) for p in pw]
    pr=[r for r in pr if r and valid_nesne(r) and is_tr(r)]
    cl=cluster(pr)
    fs=Counter()
    for p in pr:
        for f,cnt in cooc.get(p,{}).items():
            if f not in fn or sum(fn[f].values())<2 or not is_valid_fiil(f): continue
            fb=f.split('_')[0] if '_' in f else f
            if fb not in WL and '_' not in f: continue
            fs[f]+=cnt*(1+ppmi(p,f))
    if not fs:
        for f2,ns in fn.items():
            fb=f2.split('_')[0] if '_' in f2 else f2
            if fb not in WL and '_' not in f2: continue
            if sum(ns.values())>=3 and is_valid_fiil(fb): fs[f2]=sum(ns.values())
    used=set(r for r in pr if r); results=[]
    for i in range(n_sents):
        fc=[(f,s) for f,s in fs.most_common(20) if f not in used]
        if not fc: results.append(subj+' açıkladı.'); continue
        tf=fc[:10]; fp=np.array([s for _,s in tf],dtype=np.float32)+0.1; fp/=fp.sum()
        fiil=tf[int(rng.choice(len(tf),p=fp))][0]; used.add(fiil)
        fb=fiil.split('_')[0] if '_' in fiil else fiil
        ns=Counter()
        for ff in [fiil,fb]:
            for nm,cnt in fn.get(ff,{}).items():
                if nm in used or nm in LOW or not valid_nesne(nm) or not is_tr(nm): continue
                pv=sum(ppmi(p,nm) for p in pr)
                cb=3.0 if nm in cl else 0.5
                ns[nm]+=cnt*(1+pv)*cb
        nc=[(nm,s) for nm,s in ns.most_common(15) if nm not in used]
        if not nc:
            for nm in cl:
                if valid_nesne(nm) and nm not in used and nm not in LOW: nc.append((nm,float(wc.get(nm,1))))
            nc=sorted(nc,key=lambda x:-x[1])[:8]
        if not nc: results.append(subj+' '+inflect_vn(fiil)+' açıkladı.'); continue
        tn=nc[:8]; np_=np.array([s for _,s in tn],dtype=np.float32)+0.1; np_/=np_.sum()
        nesne=tn[int(rng.choice(len(tn),p=np_))][0]; used.add(nesne)
        results.append(subj+' '+inflect_noun(nesne)+' '+inflect_vn(fiil)+' '+HABER[i%len(HABER)]+'.')
    return '\n'.join(results)

TESTS=[
    (['galatasaray','şampiyonluk'],7),(['beşiktaş','transfer'],55),
    (['fenerbahçe','maç'],3),(['enflasyon','dolar'],123),
    (['faiz','merkez'],17),(['kanser','tedavi'],44),
    (['seçim','oy'],77),(['ukrayna','rusya'],88),
    (['yazılım','güvenlik'],205),(['film','yönetmen'],99),
    (['bitcoin','kripto'],33),(['grip','salgın'],77),
    (['muhalefet','anayasa'],91),(['samsung','yapay'],88),
    (['dizi','oyuncu'],66),
]
CAT_MAP={'galatasaray':'spor','beşiktaş':'spor','fenerbahçe':'spor',
         'enflasyon':'ekonomi','faiz':'ekonomi','bitcoin':'ekonomi',
         'kanser':'saglik','grip':'saglik','seçim':'siyaset',
         'muhalefet':'siyaset','ukrayna':'dunya','nato':'dunya',
         'yazılım':'teknoloji','samsung':'teknoloji','film':'kultur','dizi':'kultur'}
GOOD_MORPH=re.compile(
    r'(ını|ini|unu|ünü|nı|ni|nu|nü|sını|sini|sunu|sünü'
    r'|dığını|diğini|duğunu|düğünü|tığını|tiğini|tuğunu|tüğünü'
    r'| ettiğini| edildiğini| olduğunu| gördüğünü| alındığını)$')

t0=time.time(); b_ok=a_ok=grand=0
for pw,seed in TESTS:
    cat=CAT_MAP.get(pw[0],'siyaset')
    out_B=gen_B(pw,seed=seed)
    out_A=gen_A(pw,cat,seed=seed)
    def q(s):
        s=s.lower().rstrip('.')
        w=s.split()
        hm=bool(GOOD_MORPH.search(s))
        lr=sum(1 for x in w[1:] if len(nkoku(x) or '')>=4)
        nd=len(w)==len(set(w))
        nc=all(s.count(h)<=1 for h in ['açıkladı','belirtti','duyurdu'])
        tr=sum(1 for x in w if is_tr(x.rstrip('.')))>=len(w)*0.8
        return hm+(lr>=2)+nd+nc+tr
    bq=sum(q(s) for s in out_B.split('\n'))
    aq=sum(q(s) for s in out_A.split('\n'))
    b_ok+=bq; a_ok+=aq; grand+=15
    bi='✓' if bq>=12 else ('~' if bq>=8 else '✗')
    ai='✓' if aq>=12 else ('~' if aq>=8 else '✗')
    print(f'[{pw}]')
    print(f'  B({bi}): {out_B.replace(chr(10)," | ")}')
    print(f'  A({ai}): {out_A.replace(chr(10)," | ")}')
print(f'\nB:{b_ok/grand*100:.0f}% A:{a_ok/grand*100:.0f}% Fark:{abs(a_ok-b_ok)/grand*100:.0f}% Süre:{time.time()-t0:.1f}s')

# ── NESNE-ÖNCE MİMARİ ──────────────────────────────────────────────
def gen_B_v2(pw, n_sents=3, seed=42):
    """Önce nesne seç (cluster'dan), sonra o nesneyle uyumlu fiil seç"""
    rng=np.random.default_rng(seed)
    subj=pw[0]
    pr=[nkoku(p.lower()) for p in pw]
    pr=[r for r in pr if r and valid_nesne(r) and is_tr(r)]
    cl=cluster(pr)

    # Nesne adayları: cluster + fiil_nesne tablosunda olan
    nesne_scores=Counter()
    for nm in cl:
        if nm in LOW or not valid_nesne(nm) or not is_tr(nm): continue
        # Bu nesne için fiil var mı?
        has_verb=any(nm in fn.get(f,{}) for f in WL)
        # also check compound
        has_verb = has_verb or any(nm in fn.get(f+'_et',{}) or
                                    nm in fn.get(f+'_edil',{}) for f in WL)
        pv=sum(ppmi(p,nm) for p in pr)
        nesne_scores[nm]=wc.get(nm,1)*(1+pv)*(2.0 if has_verb else 0.8)

    used=set(r for r in pr if r); results=[]
    for i in range(n_sents):
        # Nesne seç
        nc=[(nm,s) for nm,s in nesne_scores.most_common(20) if nm not in used]
        if not nc: results.append(subj+' açıkladı.'); continue
        tn=nc[:10]; np_=np.array([s for _,s in tn],dtype=np.float32)+0.1; np_/=np_.sum()
        nesne=tn[int(rng.choice(len(tn),p=np_))][0]; used.add(nesne)

        # Bu nesneyle uyumlu fiil seç
        fs=Counter()
        for f,cnt in fn.items():
            if nesne not in cnt: continue
            if not is_valid_fiil(f.split('_')[0] if '_' in f else f): continue
            fb=f.split('_')[0] if '_' in f else f
            if fb not in WL and '_' not in f: continue
            # Prompt PMI bonusu
            pv=sum(ppmi(p,fb) for p in pr)
            fs[f]+=cnt[nesne]*(1+pv)

        if not fs:
            # Fallback: genel fiil
            for f,ns in fn.items():
                fb=f.split('_')[0] if '_' in f else f
                if fb not in WL and '_' not in f: continue
                if sum(ns.values())>=3 and is_valid_fiil(fb): fs[f]=sum(ns.values())

        fc=[(f,s) for f,s in fs.most_common(15) if f not in used]
        if not fc: results.append(subj+' '+inflect_noun(nesne)+' açıkladı.'); continue
        tf=fc[:8]; fp=np.array([s for _,s in tf],dtype=np.float32)+0.1; fp/=fp.sum()
        fiil=tf[int(rng.choice(len(tf),p=fp))][0]

        results.append(subj+' '+inflect_noun(nesne)+' '+inflect_vn(fiil)+' '+HABER[i%len(HABER)]+'.')
    return '\n'.join(results)

print()
print('='*65)
print('NESNE-ÖNCE MİMARİ')
print('='*65)
t0=time.time(); b_ok=a_ok=grand=0
for pw,seed in TESTS:
    cat=CAT_MAP.get(pw[0],'siyaset')
    out_B=gen_B_v2(pw,seed=seed)
    out_A=gen_A(pw,cat,seed=seed)
    bq=sum(q(s) for s in out_B.split('\n'))
    aq=sum(q(s) for s in out_A.split('\n'))
    b_ok+=bq; a_ok+=aq; grand+=15
    bi='✓' if bq>=12 else ('~' if bq>=8 else '✗')
    ai='✓' if aq>=12 else ('~' if aq>=8 else '✗')
    print(f'[{pw}]')
    print(f'  B({bi}): {out_B.replace(chr(10)," | ")}')
    print(f'  A({ai}): {out_A.replace(chr(10)," | ")}')
print(f'\nB:{b_ok/grand*100:.0f}% A:{a_ok/grand*100:.0f}% Fark:{abs(a_ok-b_ok)/grand*100:.0f}% Süre:{time.time()-t0:.1f}s')
