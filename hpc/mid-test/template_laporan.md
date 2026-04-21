---
layout: hpc
title: Template Laporan UTS
---

[← Kembali ke Halaman Utama](./)

# Template Laporan UTS Praktikum HPC

> Gunakan template ini sebagai panduan struktur laporan Anda.  
> Hapus semua teks dalam kurung siku `[...]` dan ganti dengan konten Anda sendiri.  
> Laporan dikumpulkan dalam format **PDF**.

---

## Halaman Judul

**Ujian Tengah Semester Praktikum**  
**High Performance Computing**

| | |
|---|---|
| **Nama Lengkap** | [Nama Anda] |
| **NIM** | [NIM Anda] |
| **Program Studi** | [Prodi Anda] |
| **Tanggal Pengumpulan** | [DD/MM/YYYY] |
| **Spesifikasi Komputer** | [CPU, jumlah core, RAM, OS] |
| **GPU (jika ada)** | [Nama/spesifikasi GPU yang digunakan] |

---

## Bagian 1: Modul Analisis Statistik (OpenMP)

### 1.1 Analisis Masalah

#### Race Condition pada `sum` dan `max_val`

[Jelaskan mengapa race condition terjadi. Sertakan output terminal yang menunjukkan nilai tidak konsisten antar run.]

**Bukti inkonsistensi (output terminal):**
```
Run 1: Rata-rata = [nilai]
Run 2: Rata-rata = [nilai]
Run 3: Rata-rata = [nilai]
Run 4: Rata-rata = [nilai]
Run 5: Rata-rata = [nilai]
```

#### False Sharing pada `partial[]`

[Jelaskan konsep false sharing dan hubungannya dengan ukuran cache line. Sertakan diagram atau ilustrasi jika membantu.]

---

### 1.2 Solusi yang Diimplementasikan

#### Perbaikan Race Condition (klausa `reduction`)

```c
// Tempelkan kode perbaikan Anda di sini
```

[Jelaskan mengapa klausa `reduction` menyelesaikan race condition tanpa overhead besar]

#### Perbaikan `max_val`

```c
// Tempelkan kode perbaikan Anda di sini
```

[Jelaskan pilihan pendekatan Anda: `critical`, `atomic`, atau `reduction` manual]

#### Perbaikan False Sharing (padding)

```c
// Tempelkan kode perbaikan Anda di sini
```

---

### 1.3 Evaluasi Kinerja — Scheduling

**Spesifikasi Komputer:** [isi spesifikasi]  
**OMP_NUM_THREADS:** [isi jumlah thread]

| Strategi Scheduling | Waktu (detik) | Speedup vs Serial | Efisiensi (%) |
|--------------------|---------------|-------------------|---------------|
| Serial (tanpa OpenMP) | | 1.00× | — |
| `schedule(static)` | | | |
| `schedule(dynamic, 1)` | | | |
| `schedule(dynamic, 1000)` | | | |
| `schedule(guided)` | | | |

**Grafik Perbandingan:**

[Tempelkan grafik bar chart atau line chart di sini]

**Perhitungan Speedup & Efisiensi (konfigurasi terbaik):**

$$S = \frac{T_{serial}}{T_{parallel}} = \frac{[\text{nilai}]}{[\text{nilai}]} = [\text{nilai}]\times$$

$$E = \frac{S}{p} \times 100\% = \frac{[\text{nilai}]}{[\text{jumlah thread}]} \times 100\% = [\text{nilai}]\%$$

**Analisis:** [Jelaskan mengapa strategi terbaik bekerja lebih baik. Hubungkan dengan karakteristik beban kerja heterogen.]

---

## Bagian 2: Modul Filter Sinyal (SIMD & Vektorisasi)

### 2.1 Analisis Masalah

#### Laporan Kompiler (Sebelum Perbaikan)

```
# Output dari: gcc -O2 -fopt-info-vec-missed signal_filter.c -o signal_filter
[tempelkan output lengkap di sini]
```

#### Penjelasan Pointer Aliasing

[Jelaskan mengapa pointer aliasing mencegah vektorisasi. Berikan contoh skenario konkret di mana overlap pointer bisa menghasilkan hasil salah.]

#### Penjelasan Memory Alignment

[Jelaskan mengapa unaligned access lebih lambat untuk instruksi SIMD/AVX]

---

### 2.2 Solusi yang Diimplementasikan

#### Versi `__restrict__`

```c
// Tempelkan kode perbaikan Anda di sini
```

**Laporan kompiler setelah perbaikan:**
```
[tempelkan output -fopt-info-vec-optimized di sini]
```

#### Versi `#pragma omp simd`

```c
// Tempelkan kode perbaikan Anda di sini
```

#### Versi Aligned Memory

```c
// Tempelkan kode perbaikan Anda di sini
```

---

### 2.3 Benchmarking

| Versi | Flag Kompilasi | Divektorisasi? | Waktu (detik) | Speedup |
|-------|----------------|----------------|---------------|---------|
| Baseline | `-O2` | Tidak | | 1.0× |
| + `__restrict__` | `-O2` | | | |
| + AVX2 | `-O3 -mavx2` | | | |
| + Aligned + AVX2 | `-O3 -mavx2` | | | |
| + `#pragma omp simd` | `-O2 -fopenmp-simd` | | | |

**Speedup teoritis vs aktual:**

- Teoritis (AVX2, 8 float/cycle): **8×**
- Aktual yang dicapai: **[nilai]×**
- Gap: **[nilai]×**

**Faktor penyebab gap:**
1. [Faktor 1 — jelaskan]
2. [Faktor 2 — jelaskan]
3. [Faktor 3 — jelaskan]

---

## Bagian 3: Modul Simulasi Perambatan (CUDA)

### 3.1 Perbaikan Global Thread Index

**Analisis Kesalahan:**

[Jelaskan secara matematis berapa sel yang diproses vs yang seharusnya diproses]

Dengan `blockDim = (16, 16)` dan kode yang salah:
- Thread yang aktif: hanya `threadIdx.x` ∈ [0, 15] dan `threadIdx.y` ∈ [0, 15]
- Jumlah sel yang diproses: **[hitung]** dari total **[hitung]** sel

**Kode Perbaikan:**

```cuda
// Tempelkan kode perbaikan Anda di sini
```

---

### 3.2 Perbaikan Warp Divergence

**Penjelasan Warp Divergence:**

[Jelaskan konsep warp, SIMT, dan bagaimana divergence muncul]

**Analisis `classify_wave_height`:**

[Estimasi persentase waktu yang terbuang akibat divergence]

**Kode Perbaikan (no-divergence):**

```cuda
// Tempelkan kode perbaikan Anda di sini
```

**Perbandingan `nvprof`:**

| Metrik | Versi Divergent | Versi Optimized | Improvement |
|--------|----------------|-----------------|-------------|
| Waktu kernel (ms) | | | |
| Warp efficiency (%) | | | |

---

### 3.3 Optimisasi Transfer Data

**Output `nvprof` Kode Awal:**
```
[Tempelkan output nvprof di sini]
```

**Analisis Bottleneck:**
- % waktu untuk kernel: [nilai]%
- % waktu untuk transfer: [nilai]%
- % waktu untuk cudaMalloc: [nilai]%

**Kode Perbaikan (transfer optimal):**

```cuda
// Tempelkan kode perbaikan Anda di sini
```

**Output `nvprof` Kode Optimal:**
```
[Tempelkan output nvprof di sini]
```

**Tabel Perbandingan:**

| Metrik | Versi Awal | Versi Optimal | Improvement |
|--------|-----------|---------------|-------------|
| Total waktu eksekusi (s) | | | |
| Waktu kernel GPU (s) | | | |
| Waktu transfer data (s) | | | |
| Jumlah `cudaMalloc` | 2000× | | |

---

## Kesimpulan Umum

### Ringkasan Pencapaian

| Modul | Bug Utama | Teknik Perbaikan | Speedup Dicapai |
|-------|-----------|------------------|-----------------|
| OpenMP | Race condition, false sharing | `reduction`, padding, `dynamic` scheduling | [nilai]× |
| SIMD | Pointer aliasing, unaligned access | `__restrict__`, `#pragma omp simd`, `posix_memalign` | [nilai]× |
| CUDA | Wrong index, warp divergence, transfer overhead | Index fix, predication, batch transfer | [nilai]× |

### Trade-off yang Ditemukan

[Diskusikan trade-off yang Anda temukan selama pengerjaan. Contoh pertanyaan yang bisa dibahas:]

- Apakah `schedule(dynamic)` selalu lebih baik dari `schedule(static)`?
- Kapan overhead sinkronisasi (critical section) lebih mahal dari manfaat parallelism-nya?
- Sampai kapan memindahkan komputasi ke GPU menguntungkan vs overhead transfer datanya?
- Bagaimana memilih antara kompleksitas kode yang tinggi vs peningkatan performa yang marginal?

### Insight Pribadi

[Tuliskan satu atau dua paragraf tentang pengalaman Anda mengerjakan ujian ini. Apa yang paling menantang? Apa yang paling menarik?]

---

## Referensi

[Daftar semua sumber yang Anda gunakan: dokumentasi OpenMP, CUDA Programming Guide, paper, dsb.]

1. OpenMP API Specification 5.2 — https://www.openmp.org/spec-html/5.2/
2. CUDA C++ Programming Guide — https://docs.nvidia.com/cuda/cuda-c-programming-guide/
3. GCC Vectorization Documentation — https://gcc.gnu.org/projects/tree-ssa/vectorization.html
4. [Sumber tambahan Anda]

---

*File ini: `UTS_HPC_[NIM]_[NamaLengkap].pdf`*
