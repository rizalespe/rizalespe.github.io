# Konteks Produk: KidVital — untuk Dipakai di Claude Code

> Tempel seluruh isi dokumen ini sebagai pesan pertama ke Claude Code, lalu tambahkan instruksi akhir seperti: *"Buatkan satu file HTML single-page yang menjelaskan produk ini berdasarkan konteks di atas, siap dibuka di browser."* Claude Code akan langsung menulis kode HTML/CSS/JS-nya secara nyata (bukan sekadar mockup visual seperti di Stitch), sehingga hasilnya file HTML asli yang bisa langsung dibuka atau di-hosting.

---

## 1. Ringkasan Produk

**Nama produk:** KidVital (nama sementara, bisa diganti)
**Kategori:** Aplikasi mobile kesehatan anak (health-tech untuk keluarga)
**Tagline:** Respons cepat untuk kesehatan anak Anda.

**Deskripsi singkat:** KidVital adalah aplikasi yang menghubungkan orang tua, dokter spesialis anak, dan rumah sakit dalam satu platform untuk memantau kondisi kesehatan anak secara real-time, mendeteksi dini kondisi yang memerlukan penanganan cepat, dan mengedukasi anak lewat gamifikasi dengan maskot karakter.

**Masalah yang diselesaikan:**
1. Orang tua sering terlambat menyadari kondisi darurat anak karena tidak ada pemantauan berkelanjutan di rumah.
2. Data kesehatan anak tersebar di banyak tempat (buku KIA, memori orang tua, rekam medis RS berbeda-beda) dan tidak terhubung langsung ke dokter.
3. Edukasi kesehatan untuk anak masih membosankan, satu arah, dan tidak melibatkan anak sebagai partisipan aktif.

**Status produk:** Konsep produk, tahap riset & prototipe (belum live/production).

---

## 2. Tiga Pilar Utama Produk

1. **Monitoring Real-Time via Bluetooth**
   Integrasi dengan alat kesehatan rumahan yang sudah beredar di pasaran (oksimeter, termometer) lewat protokol Bluetooth LE standar (GATT profile), sehingga data suhu, saturasi oksigen, dan detak jantung anak otomatis masuk ke aplikasi tanpa input manual.

2. **Deteksi Dini & Jalur Cepat ke Tenaga Medis**
   Sistem mendeteksi saat data vital anak berada di luar ambang batas normal sesuai usia, lalu memberi notifikasi ke orang tua dengan rekomendasi tindakan dan opsi satu-tap untuk konsultasi ke dokter/rumah sakit mitra. Dilengkapi asisten AI berbasis LLM untuk triase awal lewat percakapan (dengan disclaimer jelas sebagai alat bantu skrining, bukan diagnosis).

3. **Edukasi & Gamifikasi Anak lewat Maskot "Vito"**
   Maskot karakter (rubah kecil ramah) memandu anak menyelesaikan "misi kesehatan harian" (minum air, cek suhu, latihan napas) dengan sistem reward/badge, sekaligus jadi indikator kepatuhan (adherence) yang terlihat oleh dokter.

---

## 3. Fitur Teknologi AI yang Digunakan

| Fitur | Teknologi | Catatan |
|---|---|---|
| Chatbot triase gejala | LLM + RAG (menjangkarkan jawaban ke pedoman medis tervalidasi) + rule klinis dari dokter | Alat bantu skrining, bukan pengganti dokter |
| Skrining foto ruam/kulit | Vision AI (model fondasi/API pihak ketiga) | Akurat lintas warna kulit, bisa on-device untuk privasi |
| Skrining suara batuk/napas | Model klasifikasi audio | Cukup pakai mikrofon HP, tanpa alat tambahan |
| Percakapan maskot dengan anak | LLM dengan filter konten ramah anak yang ketat | Perlu lapisan keamanan konten ekstra |

---

## 4. Stakeholder / Target Pengguna

| Peran | Kebutuhan Utama |
|---|---|
| **Orang tua** (25-40 tahun) | Rasa tenang, notifikasi jelas, tindakan cepat saat darurat |
| **Anak** (4-10 tahun) | Interaksi menyenangkan lewat maskot dan misi bermain |
| **Dokter spesialis anak** | Dashboard ringkas berisi data vital & riwayat pasien real-time |
| **Rumah sakit/klinik** | Integrasi rekam medis, booking, potensi interoperabilitas dengan SATUSEHAT |

---

## 5. Tim Pengembang

- Dokter Spesialis Anak — validasi konten medis & keamanan klinis
- Tim IT & AI — pengembangan aplikasi, integrasi Bluetooth, model AI
- Tim Bisnis — strategi produk, kemitraan rumah sakit, model bisnis

---

## 6. Gaya Visual & Sistem Warna

**Nuansa keseluruhan:** modern, hangat, tepercaya, minim-stimulasi untuk bagian data/medis, tapi playful dan ceria khusus di bagian yang membahas mode anak/maskot.

**Palet warna:**
- Latar utama (netral hangat): `#F7F3EC`
- Teks utama: `#2B2B2B`
- Warna primer/brand (biru tenang, dipakai untuk judul & CTA): `#3D7BC7`
- Warna sekunder (card/section): `#E8F0FA`
- Warna aksen playful (khusus bagian maskot/anak): kuning `#FFC94D`, hijau `#6FCF97`, biru langit `#56CCF2`
- Warna alert/kritis (dipakai sangat terbatas): `#E4572E` (bahaya), `#2E9E6D` (status aman/normal)
- Mode gelap opsional: latar `#1C1C1E`, aksen warna tetap sama

**Tipografi:** sans-serif modern, judul besar dan tegas (headline-style ala pitch deck), body text mudah dibaca.

**Gaya layout:** bento grid untuk bagian berisi banyak poin (fitur, statistik, tim), rounded corners 16-24px, soft shadow tipis, banyak white space.

---

## 7. Struktur Konten Halaman yang Diinginkan

Halaman HTML sebaiknya berbentuk **single-page scroll** dengan section-section berikut secara berurutan:

1. **Hero/Cover** — judul besar tagline, badge "Konsep Produk — Tahap Riset & Prototipe", CTA scroll ke bawah.
2. **Masalah yang Diselesaikan** — 3 poin masalah dalam layout bento grid.
3. **Solusi Kami** — 3 pilar produk berdampingan (lihat bagian 2 di atas).
4. **Cara Kerja** — alur 4 langkah horizontal: Alat Bluetooth kirim data → AI deteksi anomali → Notifikasi ke orang tua & dokter → Tindakan cepat lewat chat/konsultasi.
5. **Teknologi AI yang Digunakan** — 3 kapabilitas AI (lihat bagian 3), dengan catatan etik "alat bantu skrining, bukan pengganti dokter" ditampilkan jelas.
6. **Dibangun untuk Semua Pihak** — 4 kartu persona (lihat bagian 4).
7. **Tim di Balik Produk** — 3 kelompok tim (lihat bagian 5).
8. **Penutup & Ajakan Bertindak** — CTA "Jadwalkan Diskusi" dan info kontak placeholder.

Tambahkan navigasi titik (dot navigation) di sisi kanan layar untuk lompat antar section, dengan animasi fade-in halus saat tiap section muncul ketika di-scroll (subtle, bukan berlebihan).

---

## 8. Instruksi Teknis untuk Claude Code

- Buat sebagai **satu file HTML mandiri** (inline CSS & JS dalam file yang sama, tanpa dependency eksternal selain mungkin Google Fonts) agar mudah dibagikan/dibuka langsung dari browser tanpa server.
- Pastikan **responsif** (tetap enak dilihat di HP maupun laptop), karena kemungkinan akan dibuka lewat link yang dikirim ke berbagai perangkat.
- Gunakan HTML semantik dan pastikan kontras warna teks vs latar memenuhi standar aksesibilitas dasar (WCAG AA), karena ini berkaitan dengan konteks medis dan kredibilitas.
- Bahasa konten: Bahasa Indonesia, profesional namun tetap hangat — cocok dibaca dokter, tim rumah sakit, maupun investor.
- Ini adalah halaman **presentasi/penjelasan konsep**, bukan aplikasi fungsional — tidak perlu backend, database, atau logika bisnis nyata. Boleh gunakan data/ilustrasi placeholder di bagian yang membutuhkan gambar (mis. mockup layar aplikasi, ilustrasi maskot) — cukup gunakan bentuk/ikon SVG sederhana atau placeholder visual yang relevan, tidak perlu gambar asli.