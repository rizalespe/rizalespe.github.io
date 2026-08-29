# Sigil — tanda tangan digital untuk tim internal

Tool statis (tanpa backend) untuk generate key, sign, dan verify berkas
pakai OpenPGP. Semua kriptografi jalan di browser lewat `openpgp.js` —
private key tidak pernah meninggalkan device pengguna.

## Deploy ke GitHub Pages

1. Buat folder `sign/` di repo `rizalespe.github.io`, lalu salin seluruh
   isi folder ini ke sana (`index.html`, `keygen.html`, `sign.html`,
   `verify.html`, `css/`, `js/`, `keys/`).
2. Commit & push.
3. Buka `https://rizalespe.github.io/sign/` — selesai, tidak ada build
   step.

## Menambahkan anggota tim baru

1. Anggota baru buka halaman **Buat Kunci**, generate key pair,
   simpan private key-nya sendiri baik-baik (jangan pernah commit ke
   repo).
2. Download public key-nya (`nama-public.asc`), taruh di folder
   `keys/`.
3. Tambahkan satu entri ke `keys/manifest.json`:

```json
[
  {
    "name": "Rizal Setya Perdana",
    "file": "rizal-public.asc",
    "fingerprint": "AAAA BBBB CCCC DDDD EEEE FFFF 0000 1111 2222 3333"
  }
]
```

   Fingerprint bisa dilihat lewat `gpg --show-keys nama-public.asc`
   atau situs [openpgpjs.org](https://openpgpjs.org) kalau mau cek manual.

4. Commit & push. Public key langsung muncul di halaman Beranda dan
   di dropdown halaman Verify.

## Mencabut key (revocation)

Kalau seseorang keluar dari tim atau key-nya bocor: hapus entrinya
dari `keys/manifest.json` (dan hapus file `.asc`-nya kalau mau).
Verifikasi lama terhadap berkas yang sudah ditandatangani key itu
tetap bisa dilakukan manual, tapi key itu tidak lagi muncul sebagai
opsi resmi di tool ini.

## Batasan yang perlu disadari tim

- **Bukan e-signature bersertifikat.** Ini cocok untuk kebutuhan
  internal (verifikasi dokumen, rilis, konfigurasi, dsb), bukan
  pengganti tanda tangan elektronik tersertifikasi PSrE yang punya
  kekuatan hukum formal di Indonesia.
- **Tidak ada timestamp authority.** Sigil tidak membuktikan *kapan*
  persisnya sebuah berkas ditandatangani secara independen — hanya
  bahwa ia ditandatangani oleh pemegang key tersebut. Kalau butuh
  bukti waktu yang kuat, tambahkan integrasi RFC 3161 TSA di iterasi
  berikutnya (butuh proxy kecil karena CORS, misal lewat Cloudflare
  Worker gratis).
- **Passphrase hilang = key hilang.** Tidak ada mekanisme recovery.
  Sarankan tim pakai password manager.
- **File privasi:** kalau berkas yang ditandatangani bersifat
  rahasia, ingat bahwa proses ini memang tidak mengunggah berkas ke
  mana pun — tapi tetap pastikan device yang dipakai aman (browser
  ekstensi mencurigakan dsb bisa membaca clipboard/DOM).

## Struktur folder

```
sign/
├── index.html       # beranda + daftar public key tim
├── keygen.html       # generate key pair
├── sign.html         # sign berkas apa saja (detached signature)
├── sign-pdf.html      # tempel gambar ttd + segel ke PDF, lalu sign
├── verify.html       # verify berkas + signature
├── css/style.css
├── js/seal.js
└── keys/
    ├── manifest.json # daftar {name, file, fingerprint}
    └── *.asc         # public key masing-masing anggota
```

## Tentang halaman "Sign PDF"

Halaman ini khusus untuk PDF yang butuh tampilan tanda tangan visual
(gambar ttd + segel kecil berisi nama, fingerprint, dan waktu). Alurnya:

1. Gambar PNG tanda tangan ditempel ke PDF di posisi yang dipilih.
2. PDF hasil tempelan itu (bukan PDF asli) yang ditandatangani secara
   kriptografis — jadi kalau stempel visualnya diubah setelah ini,
   verifikasi akan gagal juga.
3. Gambar tanda tangan itu sendiri **hanya dekorasi**. Bukti keaslian
   yang sesungguhnya tetap file `.sig.asc` yang menyertainya —
   verifikasi selalu lewat halaman Verify, bukan dengan melihat
   gambarnya secara visual.
