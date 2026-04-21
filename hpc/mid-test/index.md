---
layout: hpc
title: UTS Praktikum HPC — Optimalisasi Sistem Prediksi Anomali Maritim Indonesia
---

# UTS Praktikum High Performance Computing
## "Optimalisasi Sistem Prediksi Anomali Maritim Indonesia"

> **Tipe:** Take-Home Exam  
> **Durasi:** 7 hari 
> **Infrastruktur:** Komputer pribadi masing-masing mahasiswa 
---

## Latar Belakang

Sebuah lembaga riset kelautan di Indonesia sedang mengembangkan **sistem pemantauan gelombang untuk peringatan dini tsunami**. Mereka memiliki kode prototipe yang sudah berjalan, namun performanya sangat lambat dan tidak memenuhi syarat *real-time*.

Anda ditugaskan sebagai **Spesialis HPC** untuk mengoptimasi tiga modul utama dalam sistem tersebut. Setiap modul memiliki bug dan bottleneck yang sengaja ditanam — tugas Anda adalah mengidentifikasi, memperbaiki, dan mengukur dampak optimisasinya.

---

## Struktur Soal

| Bagian | Modul | Topik | Bobot |
|--------|-------|-------|-------|
| [Bagian 1](bagian1) | Analisis Statistik | OpenMP — Race Condition & Scheduling | 35% |
| [Bagian 2](bagian2) | Filter Sinyal | SIMD & Vektorisasi | 30% |
| [Bagian 3](bagian3) | Simulasi Perambatan | CUDA GPU Programming | 35% |

---

## File Kode yang Disediakan

Unduh masing-masing starter code berikut:
- [`wave_stats.c`](code/wave_stats.c) — Modul OpenMP (Bagian 1)
- [`signal_filter.c`](code/signal_filter.c) — Modul SIMD (Bagian 2)
- [`wave_propagation.cu`](code/wave_propagation.cu) — Modul CUDA (Bagian 3)

---

## Format Laporan

Laporan dikumpulkan dalam **satu file PDF** dengan struktur berikut:

1. **Halaman Judul** — Nama, NIM, Tanggal
2. **Bagian 1** — Analisis + Solusi + Evaluasi OpenMP
3. **Bagian 2** — Analisis + Solusi + Evaluasi SIMD
4. **Bagian 3** — Analisis + Solusi + Evaluasi CUDA
5. **Kesimpulan Umum** — Trade-off kompleksitas vs performa

Lihat [template laporan](template_laporan) untuk panduan lengkap.

---

## Infrastruktur & Kompilasi

Gunakan komputer pribadi masing-masing untuk mengerjakan soal ini. Pastikan tools berikut sudah terinstal:

### Bagian 1 & 2 (OpenMP & SIMD)
```bash
# Pastikan GCC terinstal dengan dukungan OpenMP
gcc -O2 -fopenmp -fopt-info-vec wave_stats.c -o wave_stats
```

### Bagian 3 (CUDA)
```bash
# Pastikan NVIDIA CUDA Toolkit terinstal
nvcc -O2 wave_propagation.cu -o wave_propagation
nvprof ./wave_propagation
```

> **Catatan:** Jika komputer Anda tidak memiliki GPU NVIDIA, gunakan [Google Colab](https://colab.research.google.com/) dengan runtime GPU untuk mengerjakan Bagian 3.

---

## Aturan Pengerjaan

- Dikerjakan **secara individu**
- Dilarang berbagi kode solusi dengan sesama mahasiswa
- Laporan harus mencantumkan **output terminal asli** (screenshot atau copy-paste)
- Semua grafik harus dibuat dari data eksperimen nyata, bukan estimasi
- Plagiarisme akan berakibat **nilai 0** untuk seluruh ujian

---

## Deadline & Pengumpulan

Pengumpulan melalui platform yang ditentukan dosen. Pastikan file laporan diberi nama:

```
UTS_HPC_[NIM]_[NamaLengkap].pdf
```

---

*Selamat mengerjakan! Ingat: sistem peringatan dini tsunami yang andal bergantung pada performa komputasi yang Anda optimalkan.*
