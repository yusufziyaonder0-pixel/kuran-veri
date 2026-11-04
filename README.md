# kuran-veri

Kur’ân-ı Kerîm Android uygulamasının (Marmara ve Diyanet mushafları) **internetten indirilen veri paketleri**.
Uygulama bu dosyaları `data/` klasöründen okur; `data/manifest.json` her dosyanın sürümünü (`h`), boyutunu ve güncellenme zamanını (`t`) tutar.
Kişisel, ticari olmayan kullanım içindir.

## Kaynaklar
- **Quranpedia.net** (https://quranpedia.net) — Nesefî, *Medâriku’t-tenzîl*; on kıraat farkları ve râvîlerin âyet kayıtları; müteşâbih âyetler; Vâhidî, *Esbâbü’n-nüzûl*; garîb kelime manaları. Veri dökümü sürümü: **2026-11-03** (Quranpedia veri lisansı gereği kaynak ve sürüm belirtilmiştir).
- **Tanzil Projesi** (https://tanzil.net) — Osmanlı/imlâ Kur’ân metni (CC BY 3.0, değiştirilmeden), Celâleyn tefsiri.
- **Quranic Arabic Corpus** (https://corpus.quran.com, GNU GPL) — kelime kökleri ve morfoloji.
- **OpenITI** (Şâmile nüshası) — Râgıb el-İsfahânî, *el-Müfredât fî garîbi’l-Kur’ân*.
- İbn Fâris, *Mekâyîsü’l-luga* (nibrascode/translate-jobs) — kök anlamlarına kaynak.
- Bağlamsal kelime manaları (`mana.json`), kök açıklamaları (`mana-kok.json`) ve esbâb-ı nüzûl tercümeleri (`nuzul-tr.json`) uygulama için hazırlanmıştır.

Günümüz eserleri (mealler, Ömer Çelik tefsiri, İ‘râb kitapları, Quran.com kelime mealleri) telifli olduğundan burada **yoktur**; yalnızca uygulamanın içinde bulunur.
