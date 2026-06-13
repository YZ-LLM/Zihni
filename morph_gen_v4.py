"""
MorphGen v4 — 4 atomik fix:
1. galibiyet+ini = galibiyetini (t istisnası — sert ünsüz korunur bazı kelimelerde)
2. Haber fiili VN_ACC fiiline göre seç (duyur→duyurdu çakışmasın)
3. Intransitive fiiller V_PAST kullanır, VN_ACC değil
4. Prompt nesnesi frame uyumlu havuzda önce aranır
"""
import numpy as np, time

# ── MORFOLOJİ ──────────────────────────────────────────────────────────────
YUMUSAMA = {'ç':'c','k':'ğ','p':'b','t':'d'}

# Sorun 1 FIX: Bu kelimeler t/k/p/ç yumuşatılmaz (isim sonu sessiz korunur)
NO_SOFTEN = {
    'kayıt','araç','yurt','kanıt','robot','format','limit','tablet',
    'market','pilot','sepet','karat','kalp','harp','sant','kart',
    # t korunanlar — istatistiksel (Türkçe'de galibiyet, yenilgi gibi)
    'galibiyet','yenilgi','mağlubiyet','başarı','zafer',
    'hakikat','niyet','itimat','sıfat','hayat','hizmet',
    'devlet','millet','saadet','selamet','silahet',
    'internet','tablet','futbol','basketbol',
}

def soften(word):
    if len(word) < 2: return word
    last = word[-1].lower()
    if last not in YUMUSAMA: return word
    vowels = sum(1 for c in word if c in 'aeıiouüö')
    if vowels <= 1: return word
    if word.lower() in NO_SOFTEN: return word
    return word[:-1] + YUMUSAMA[last]

def last_vowel(w):
    for c in reversed(w.lower()):
        if c in 'aeıiouüö': return c
    return 'a'

def is_vowel(c): return c.lower() in 'aeıiouüö'

def harmony(w):
    v = last_vowel(w)
    if v in 'aı': return 'A'
    if v in 'ou': return 'O'
    if v in 'ei': return 'E'
    return 'U'

SERT_U = set('çfhkpsşt')
EK = {
    'ACC':         {'A':'ı','O':'u','E':'i','U':'ü'},
    'DAT':         {'A':'a','O':'a','E':'e','U':'e'},
    'LOC':         {'A':'da','O':'da','E':'de','U':'de'},
    'ABL':         {'A':'dan','O':'dan','E':'den','U':'den'},
    'GEN':         {'A':'ın','O':'un','E':'in','U':'ün'},
    'P3S_DAT':     {'A':'ına','O':'una','E':'ine','U':'üne'},
    'P3S_DATV':    {'A':'sına','O':'suna','E':'sine','U':'süne'},
    'P3S_ACC':     {'A':'ını','O':'unu','E':'ini','U':'ünü'},
    'P3S_ACCV':    {'A':'sını','O':'sunu','E':'sini','U':'sünü'},
    'PLU_ACC':     {'A':'ları','O':'ları','E':'leri','U':'leri'},
    'PLU_P3S_ACC': {'A':'larını','O':'larını','E':'lerini','U':'lerini'},
    'PLU_P3S_DAT': {'A':'larına','O':'larına','E':'lerine','U':'lerine'},
    'PLU_ABL':     {'A':'lardan','O':'lardan','E':'lerden','U':'lerden'},
}

def v_vn_acc(root):
    h = harmony(root)
    suf = {'A':'tığını','O':'tuğunu','E':'tiğini','U':'tüğünü'} \
          if root[-1].lower() in SERT_U \
          else {'A':'dığını','O':'duğunu','E':'diğini','U':'düğünü'}
    return root + suf[h]

def v_past(root):
    h = harmony(root)
    suf = {'A':'tı','O':'tu','E':'ti','U':'tü'} \
          if root[-1].lower() in SERT_U \
          else {'A':'dı','O':'du','E':'di','U':'dü'}
    return root + suf[h]

def v_vn_nec(root):
    h = harmony(root)
    return root + {'A':'masını','O':'masını','E':'mesini','U':'mesini'}[h]

def inflect(root, case):
    soft = soften(root) if case in ('ACC','DAT','LOC','ABL','GEN',
                                     'P3S_DAT','P3S_ACC') else root
    h = harmony(root); ev = is_vowel(root[-1])
    last = root[-1].lower(); sert = last in SERT_U
    if case=='ACC':
        return soft+('y' if ev else '')+EK['ACC'][h]
    if case=='DAT':
        s = soften(root)
        return s+('y' if ev else '')+EK['DAT'][h]
    if case=='LOC':
        s = soften(root)
        suf = ('ta' if sert and not ev else EK['LOC'][h])
        return s+suf
    if case=='ABL':
        s = soften(root)
        suf = ('tan' if sert and not ev else EK['ABL'][h])
        return s+suf
    if case=='GEN':
        s = soften(root)
        return s+('n' if ev else '')+EK['GEN'][h]
    if case=='P3S_DAT':
        s = soften(root)
        return s+(EK['P3S_DATV'][h] if ev else EK['P3S_DAT'][h])
    if case=='P3S_ACC':
        s = soften(root)
        return s+(EK['P3S_ACCV'][h] if ev else EK['P3S_ACC'][h])
    if case=='PLU_ACC':      return root+EK['PLU_ACC'][h]
    if case=='PLU_P3S_ACC':  return root+EK['PLU_P3S_ACC'][h]
    if case=='PLU_P3S_DAT':  return root+EK['PLU_P3S_DAT'][h]
    if case=='PLU_ABL':      return root+EK['PLU_ABL'][h]
    return root

def inflect_phrase(expr, case):
    parts = expr.strip().split()
    if len(parts)==1:
        r=parts[0]
        if case=='VN_ACC': return v_vn_acc(r)
        if case=='VN_NEC': return v_vn_nec(r)
        if case=='V_PAST': return v_past(r)
        return inflect(r, case)
    last=parts[-1]; pre=' '.join(parts[:-1])
    if case=='VN_ACC': return pre+' '+v_vn_acc(last)
    if case=='VN_NEC': return pre+' '+v_vn_nec(last)
    if case=='V_PAST': return pre+' '+v_past(last)
    return pre+' '+inflect(last, case)

# ── FİİL-NESNE UYUMLULUK ─────────────────────────────────────────────────────
# Sorun 3 FIX: is_intransitive → V_PAST kullan, VN_ACC değil
# format: (fiil_kök, nesne_case, [nesneler], [haber_fiiller], is_intransitive)
VERB_FRAMES = {
    'spor': [
        ('kazan',       'P3S_ACC',['şampiyonluk','kupa','maç','galibiyet','puan'],
                        ['belirtti','açıkladı'], False),
        ('kaybet',      'P3S_ACC',['maç','kupa','puan','galibiyet','liderlik'],
                        ['belirtti','açıkladı'], False),
        ('transfer et', 'P3S_ACC',['oyuncu','yıldız','teknik direktör','forvet'],
                        ['duyurdu','açıkladı'], False),
        ('imzala',      'ACC',    ['sözleşme','anlaşma','kontrat','uzatma'],
                        ['duyurdu','belirtti'], False),
        ('elde et',     'P3S_ACC',['galibiyet','kupa','puan','zafer'],
                        ['açıkladı','belirtti'], False),
        ('hazırlan',    'DAT',    ['maç','sezon','kupa','final'],
                        ['belirtti','açıkladı'], True),   # intransitive → V_PAST
        ('devam et',    'DAT',    ['şampiyonluk','mücadele','sezon'],
                        ['belirtti','söyledi'], True),    # intransitive
    ],
    'ekonomi': [
        ('artır',       'P3S_ACC',['faiz','yatırım','ihracat','bütçe','gelir'],
                        ['açıkladı','belirtti'], False),
        ('gerçekleş',   'LOC',    ['piyasa','büyüme','artış'],
                        ['açıklandı','bildirildi'], True),  # intransitive
        ('ulaş',        'DAT',    ['hedef','oran','seviye','büyüme','rekor'],
                        ['açıkladı','bildirdi'], True),     # intransitive
        ('kaydet',      'P3S_ACC',['büyüme','artış','gelişme','rekor'],
                        ['açıkladı','belirtti'], False),
        ('belirle',     'P3S_ACC',['hedef','oran','politika','strateji'],
                        ['açıkladı','duyurdu'], False),
        ('açıkla',      'P3S_ACC',['veri','oran','rakam','karar','hedef'],
                        ['belirtti','bildirdi'], False),
        ('azalt',       'P3S_ACC',['enflasyon','faiz','bütçe','borç'],
                        ['açıkladı','belirtti'], False),
    ],
    'saglik': [
        ('tedavi et',   'ACC',    ['hasta','hastalık','kanser','enfeksiyon'],
                        ['belirtti','açıkladı'], False),
        ('önle',        'P3S_ACC',['hastalık','salgın','risk','enfeksiyon'],
                        ['uyardı','belirtti'], False),
        ('geliştir',    'P3S_ACC',['ilaç','aşı','tedavi','yöntem'],
                        ['duyurdu','açıkladı'], False),
        ('bul',         'P3S_ACC',['ilaç','aşı','çözüm','tedavi'],
                        ['açıkladı','bildirdi'], False),
        ('uygula',      'P3S_ACC',['tedavi','ilaç','yöntem','aşı'],
                        ['belirtti','açıkladı'], False),
        ('öner',        'P3S_ACC',['tedavi','ilaç','yöntem','önlem'],
                        ['vurguladı','belirtti'], False),
    ],
    'siyaset': [
        ('kazan',       'P3S_ACC',['seçim','oy','referandum','zafer'],
                        ['açıkladı','belirtti'], False),
        ('reddet',      'P3S_ACC',['teklif','talep','iddia','karar','yasa'],
                        ['açıkladı','belirtti'], False),
        ('kabul et',    'P3S_ACC',['karar','yasa','teklif','öneri','değişiklik'],
                        ['açıkladı','duyurdu'], False),
        ('sun',         'P3S_ACC',['yasa','teklif','plan','program','rapor'],
                        ['açıkladı','belirtti'], False),
        ('destekle',    'P3S_ACC',['karar','yasa','teklif','aday','girişim'],
                        ['belirtti','açıkladı'], False),
        ('müzakere et', 'P3S_ACC',['karar','reform','anlaşma','teklif'],
                        ['belirtti','açıkladı'], False),
    ],
    'dunya': [
        ('müzakere et', 'P3S_ACC',['anlaşma','barış','ateşkes','kriz'],
                        ['açıkladı','belirtti'], False),
        ('kına',        'P3S_ACC',['saldırı','karar','eylem','hamle','tutum'],
                        ['açıkladı','bildirdi'], False),
        ('destekle',    'P3S_ACC',['karar','girişim','operasyon','çaba'],
                        ['belirtti','açıkladı'], False),
        ('imzala',      'P3S_ACC',['anlaşma','antlaşma','bildiri','protokol'],
                        ['duyurdu','açıkladı'], False),
        ('çekil',       'ABL',    ['anlaşma','görüşme','süreç','müzakere'],
                        ['açıkladı','bildirdi'], True),     # intransitive
        ('bildir',      'P3S_ACC',['karar','açıklama','önlem','strateji'],
                        ['açıkladı','belirtti'], False),
    ],
    'teknoloji': [
        ('geliştir',    'P3S_ACC',['yazılım','uygulama','teknoloji','sistem','araç'],
                        ['duyurdu','açıkladı'], False),
        ('piyasaya sür','P3S_ACC',['ürün','yazılım','uygulama','cihaz','model'],
                        ['duyurdu','açıkladı'], False),
        ('tanıt',       'P3S_ACC',['ürün','sistem','özellik','platform','model'],
                        ['açıkladı','belirtti'], False),
        ('entegre et',  'P3S_ACC',['sistem','yazılım','platform','teknoloji'],
                        ['açıkladı','belirtti'], False),
        ('güncelle',    'P3S_ACC',['sistem','uygulama','yazılım','platform'],
                        ['duyurdu','açıkladı'], False),
    ],
    'kultur': [
        ('çek',         'P3S_ACC',['film','dizi','belgesel','klip','video'],
                        ['söyledi','açıkladı'], False),
        ('yayınla',     'P3S_ACC',['albüm','film','dizi','kitap','single'],
                        ['duyurdu','açıkladı'], False),
        ('tamamla',     'P3S_ACC',['proje','film','albüm','çalışma','yapım'],
                        ['söyledi','belirtti'], False),
        ('sun',         'P3S_ACC',['proje','sergi','program','film'],
                        ['açıkladı','söyledi'], False),
        ('hazırla',     'P3S_ACC',['albüm','proje','film','program'],
                        ['söyledi','belirtti'], False),
    ],
}

# Sorun 2 FIX: Haber fiili VN_ACC fiiline göre seç
VERB_HABER_CONFLICT = {
    # Eğer VN_ACC'nin son kökü bu ise, bu haber fiillerini KULLANMA
    'duyur': ['duyurdu'],
    'bildir': ['bildirdi'],
    'açıkla': ['açıkladı'],
    'belirt': ['belirtti'],
    'söyle': ['söyledi'],
}

def pick_haber_verb(verb_root, haber_verbs, sent_i):
    """Sorun 2: VN_ACC fiiliyle çakışmayan haber fiilini seç"""
    # verb_root'un son kökü ne?
    root_key = verb_root.split()[-1][:6]  # ilk 6 karakter
    forbidden = set()
    for k, fv in VERB_HABER_CONFLICT.items():
        if root_key.startswith(k[:5]):
            forbidden.update(fv)
    available = [h for h in haber_verbs if h not in forbidden]
    if not available:
        available = haber_verbs  # fallback
    return available[sent_i % len(available)]

# Sorun 4 FIX: Prompt nesnesini önce frame havuzunda ara
def find_prompt_noun(prompt_words, noun_list, noun_case):
    """Prompt kelimesi doğrudan frame'in nesne listesindeyse kullan"""
    prompt_lower = {p.lower() for p in prompt_words[1:]}  # özneden sonrası
    # Tam eşleşme
    for pw in prompt_lower:
        if pw in noun_list:
            return pw
    # Kısmi eşleşme (kök benzerliği)
    for pw in prompt_lower:
        for noun in noun_list:
            if len(pw)>=4 and len(noun)>=4:
                if pw[:4]==noun[:4] or noun.startswith(pw[:5]):
                    return noun
    return None

def generate_sentence(prompt_words, category, frame_idx, used_nouns, rng, sent_i):
    frames = VERB_FRAMES.get(category, VERB_FRAMES['siyaset'])
    fi = frame_idx % len(frames)
    verb_root, noun_case, nouns, haber_verbs, is_intrans = frames[fi]

    subj = prompt_words[0]

    # Sorun 4 FIX: önce prompt nesnesini ara
    chosen = find_prompt_noun(prompt_words, nouns, noun_case)
    if chosen is None or chosen in used_nouns:
        available = [n for n in nouns if n not in used_nouns]
        if not available: available = nouns
        chosen = available[int(rng.integers(0, len(available)))]
    used_nouns.add(chosen)

    noun_inflected = inflect(chosen, noun_case)

    # Sorun 3 FIX: intransitive → V_PAST, transitive → VN_ACC
    if is_intrans:
        verb_part = inflect_phrase(verb_root, 'V_PAST')
        # Sorun 2 FIX: intransitive'de haber fiili ekleme, zaten tam cümle
        return f'{subj} {noun_inflected} {verb_part}.'
    else:
        verb_vn = inflect_phrase(verb_root, 'VN_ACC')
        # Sorun 2 FIX: çakışmayan haber fiili seç
        hv = pick_haber_verb(verb_root, haber_verbs, sent_i)
        return f'{subj} {noun_inflected} {verb_vn} {hv}.'

def generate(prompt_words, category, n_sents=3, seed=42):
    rng = np.random.default_rng(seed)
    frames = VERB_FRAMES.get(category, VERB_FRAMES['siyaset'])
    used_nouns = set(w.lower() for w in prompt_words)
    start = int(rng.integers(0, len(frames)))
    results = []
    for i in range(n_sents):
        fi = (start + i) % len(frames)
        s = generate_sentence(prompt_words, category, fi, used_nouns, rng, i)
        results.append(s)
    return '\n'.join(results)

if __name__ == '__main__':
    # Morfoloji testleri
    print('[MORFOLOJİ FIX TESTİ]')
    for root, case in [('galibiyet','P3S_ACC'),('şampiyonluk','P3S_ACC'),
                        ('hastalık','DAT'),('maç','P3S_ACC'),
                        ('karar','P3S_ACC'),('anlaşma','P3S_ACC')]:
        print(f'  {root}+{case} → {inflect(root,case)}')
    print()

    tests_fixed = [
        (['galatasaray','şampiyonluk'], 'spor',      7),
        (['beşiktaş','transfer'],       'spor',     55),
        (['fenerbahçe','maç'],          'spor',      3),
        (['enflasyon','dolar'],         'ekonomi', 123),
        (['faiz','merkez'],             'ekonomi',  17),
        (['ameliyat','doktor'],         'saglik',   33),
        (['kanser','tedavi'],           'saglik',   44),
        (['seçim','oy'],               'siyaset',  77),
        (['muhalefet','meclis'],       'siyaset',  91),
        (['ukrayna','rusya'],          'dunya',    88),
        (['nato','savaş'],             'dunya',    66),
        (['yazılım','güvenlik'],       'teknoloji',205),
        (['yapay','zeka'],             'teknoloji', 78),
        (['film','yönetmen'],          'kultur',   99),
        (['müzik','konser'],           'kultur',   55),
        # Yeni promptlar
        (['trabzonspor','lig'],        'spor',     11),
        (['bitcoin','kripto'],         'ekonomi',  33),
        (['grip','salgın'],            'saglik',   77),
        (['erdoğan','meclis'],         'siyaset',  42),
        (['fransa','nato'],            'dunya',    19),
        (['samsung','yapay'],          'teknoloji', 88),
        (['dizi','oyuncu'],            'kultur',   66),
        (['biden','çin'],              'dunya',    55),
        (['istanbul','ekonomi'],       'ekonomi',  22),
        (['pfizer','aşı'],            'saglik',   11),
    ]

    t0 = time.time()
    for pw, cat, seed in tests_fixed:
        print(f'[{cat.upper()}] {pw}')
        print(generate(pw, cat, seed=seed))
        print()
    print(f'Toplam: {time.time()-t0:.3f}s')
