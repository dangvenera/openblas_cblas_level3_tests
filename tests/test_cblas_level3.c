/*
 * Interface tests for OpenBLAS CBLAS Level 3 functions.
 * Language: C99.
 * Scope: CBLAS BLAS Level 3 routines: GEMM, SYMM, HEMM, SYRK, HERK,
 *        SYR2K, HER2K, TRMM, TRSM for supported S/D/C/Z data types.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <cblas.h>

#if defined(__GNUC__)
void openblas_set_num_threads(int num_threads) __attribute__((weak));
#else
void openblas_set_num_threads(int num_threads);
#endif

static int checks = 0;
static int fails = 0;
static int current_threads = 1;

static void set_threads_safe(int n) {
#if defined(__GNUC__)
    if (openblas_set_num_threads) openblas_set_num_threads(n);
#else
    openblas_set_num_threads(n);
#endif
    current_threads = n;
}

static void ok_real(const char *name, const double *got, const double *exp, int len, double eps) {
    int local_fails = 0;
    checks++;
    for (int i = 0; i < len; ++i) {
        double diff = fabs(got[i] - exp[i]);
        if (diff > eps) {
            if (local_fails < 3) {
                printf("[FAIL] threads=%d %s: index %d got %.12g expected %.12g diff %.3g\n",
                       current_threads, name, i, got[i], exp[i], diff);
            }
            local_fails++;
        }
    }
    if (local_fails) {
        fails++;
    } else {
        printf("[PASS] threads=%d %s\n", current_threads, name);
    }
}

static void ok_float(const char *name, const float *got, const float *exp, int len, float eps) {
    double *g = (double*)calloc((size_t)len, sizeof(double));
    double *e = (double*)calloc((size_t)len, sizeof(double));
    for (int i = 0; i < len; ++i) { g[i] = got[i]; e[i] = exp[i]; }
    ok_real(name, g, e, len, eps);
    free(g); free(e);
}

static void ok_cfloat(const char *name, const float complex *got, const float complex *exp, int len, float eps) {
    int local_fails = 0;
    checks++;
    for (int i = 0; i < len; ++i) {
        float diff = cabsf(got[i] - exp[i]);
        if (diff > eps) {
            if (local_fails < 3) {
                printf("[FAIL] threads=%d %s: index %d got %.6g%+.6gi expected %.6g%+.6gi diff %.3g\n",
                       current_threads, name, i, crealf(got[i]), cimagf(got[i]),
                       crealf(exp[i]), cimagf(exp[i]), diff);
            }
            local_fails++;
        }
    }
    if (local_fails) fails++; else printf("[PASS] threads=%d %s\n", current_threads, name);
}

static void ok_cdouble(const char *name, const double complex *got, const double complex *exp, int len, double eps) {
    int local_fails = 0;
    checks++;
    for (int i = 0; i < len; ++i) {
        double diff = cabs(got[i] - exp[i]);
        if (diff > eps) {
            if (local_fails < 3) {
                printf("[FAIL] threads=%d %s: index %d got %.12g%+.12gi expected %.12g%+.12gi diff %.3g\n",
                       current_threads, name, i, creal(got[i]), cimag(got[i]),
                       creal(exp[i]), cimag(exp[i]), diff);
            }
            local_fails++;
        }
    }
    if (local_fails) fails++; else printf("[PASS] threads=%d %s\n", current_threads, name);
}

static void ok_float_tri(const char *name, const float *got, const float *exp, int n, CBLAS_UPLO uplo, float eps) {
    int local_fails = 0;
    checks++;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if ((uplo == CblasUpper && i <= j) || (uplo == CblasLower && i >= j)) {
                int k = i*n + j;
                float diff = fabsf(got[k] - exp[k]);
                if (diff > eps) {
                    if (local_fails < 3) printf("[FAIL] threads=%d %s: (%d,%d) got %.6g expected %.6g\n", current_threads, name, i, j, got[k], exp[k]);
                    local_fails++;
                }
            }
        }
    }
    if (local_fails) fails++; else printf("[PASS] threads=%d %s\n", current_threads, name);
}

static void ok_double_tri(const char *name, const double *got, const double *exp, int n, CBLAS_UPLO uplo, double eps) {
    int local_fails = 0;
    checks++;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if ((uplo == CblasUpper && i <= j) || (uplo == CblasLower && i >= j)) {
                int k = i*n + j;
                double diff = fabs(got[k] - exp[k]);
                if (diff > eps) {
                    if (local_fails < 3) printf("[FAIL] threads=%d %s: (%d,%d) got %.12g expected %.12g\n", current_threads, name, i, j, got[k], exp[k]);
                    local_fails++;
                }
            }
        }
    }
    if (local_fails) fails++; else printf("[PASS] threads=%d %s\n", current_threads, name);
}

static void ok_cfloat_tri(const char *name, const float complex *got, const float complex *exp, int n, CBLAS_UPLO uplo, float eps) {
    int local_fails = 0;
    checks++;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if ((uplo == CblasUpper && i <= j) || (uplo == CblasLower && i >= j)) {
                int k = i*n + j;
                float diff = cabsf(got[k] - exp[k]);
                if (diff > eps) {
                    if (local_fails < 3) printf("[FAIL] threads=%d %s: (%d,%d) got %.6g%+.6gi expected %.6g%+.6gi\n", current_threads, name, i, j, crealf(got[k]), cimagf(got[k]), crealf(exp[k]), cimagf(exp[k]));
                    local_fails++;
                }
            }
        }
    }
    if (local_fails) fails++; else printf("[PASS] threads=%d %s\n", current_threads, name);
}

static void ok_cdouble_tri(const char *name, const double complex *got, const double complex *exp, int n, CBLAS_UPLO uplo, double eps) {
    int local_fails = 0;
    checks++;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if ((uplo == CblasUpper && i <= j) || (uplo == CblasLower && i >= j)) {
                int k = i*n + j;
                double diff = cabs(got[k] - exp[k]);
                if (diff > eps) {
                    if (local_fails < 3) printf("[FAIL] threads=%d %s: (%d,%d) got %.12g%+.12gi expected %.12g%+.12gi\n", current_threads, name, i, j, creal(got[k]), cimag(got[k]), creal(exp[k]), cimag(exp[k]));
                    local_fails++;
                }
            }
        }
    }
    if (local_fails) fails++; else printf("[PASS] threads=%d %s\n", current_threads, name);
}

static void test_gemm_real(void) {
    float Af[6] = {1,2,3, 4,5,6};
    float Bf[6] = {7,8, 9,10, 11,12};
    float Cf[4] = {1,1,1,1};
    float Ef[4] = {58.5f, 64.5f, 139.5f, 154.5f};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 3, 1.0f, Af, 3, Bf, 2, 0.5f, Cf, 2);
    ok_float("cblas_sgemm RowMajor NoTrans/NoTrans alpha/beta", Cf, Ef, 4, 1e-4f);

    double Ad[6] = {1,2,3, 4,5,6};
    double Bd[6] = {7,8, 9,10, 11,12};
    double Cd[4] = {1,1,1,1};
    double Ed[4] = {58.5, 64.5, 139.5, 154.5};
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 3, 1.0, Ad, 3, Bd, 2, 0.5, Cd, 2);
    ok_real("cblas_dgemm RowMajor NoTrans/NoTrans alpha/beta", Cd, Ed, 4, 1e-10);

    double Ac[4] = {1,3, 2,4}; /* column-major storage of [[1,2],[3,4]] */
    double Bc[4] = {5,7, 6,8}; /* column-major storage of [[5,6],[7,8]] */
    double Cc[4] = {0,0,0,0};
    double Ec_col[4] = {19,43, 22,50};
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, Ac, 2, Bc, 2, 0.0, Cc, 2);
    ok_real("cblas_dgemm ColMajor layout", Cc, Ec_col, 4, 1e-10);
}

static void test_gemm_complex(void) {
    float complex A[4] = {1+1*I, 2-1*I, 3+0*I, 4+2*I};
    float complex B[4] = {2+0*I, 0+1*I, 1-1*I, 3+0*I};
    float complex C[4] = {1+0*I,1+0*I,1+0*I,1+0*I};
    float complex E[4] = {4-1*I, 6-2*I, 13-2*I, 13+9*I};
    float complex alpha = 1.0f + 0.0f*I, beta = 1.0f + 0.0f*I;
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, &alpha, A, 2, B, 2, &beta, C, 2);
    ok_cfloat("cblas_cgemm complex float", C, E, 4, 1e-4f);

    double complex Az[4] = {1+1*I, 2-1*I, 3+0*I, 4+2*I};
    double complex Bz[4] = {2+0*I, 0+1*I, 1-1*I, 3+0*I};
    double complex Cz[4] = {1+0*I,1+0*I,1+0*I,1+0*I};
    double complex Ez[4] = {4-1*I, 6-2*I, 13-2*I, 13+9*I};
    double complex alphaz = 1.0 + 0.0*I, betaz = 1.0 + 0.0*I;
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, &alphaz, Az, 2, Bz, 2, &betaz, Cz, 2);
    ok_cdouble("cblas_zgemm complex double", Cz, Ez, 4, 1e-10);
}

static void test_symm(void) {
    float A[4] = {2,1, 1,3};
    float B[4] = {1,2, 3,4};
    float C[4] = {0,0,0,0};
    float E[4] = {5,8, 10,14};
    cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, 2, 2, 1.0f, A, 2, B, 2, 0.0f, C, 2);
    ok_float("cblas_ssymm Side=Left Uplo=Upper", C, E, 4, 1e-4f);

    double Ad[4] = {2,1, 1,3};
    double Bd[4] = {1,2, 3,4};
    double Cd[4] = {0,0,0,0};
    double Ed[4] = {4,7, 10,15};
    cblas_dsymm(CblasRowMajor, CblasRight, CblasLower, 2, 2, 1.0, Ad, 2, Bd, 2, 0.0, Cd, 2);
    ok_real("cblas_dsymm Side=Right Uplo=Lower", Cd, Ed, 4, 1e-10);

    float complex Ac[4] = {2+0*I, 1+1*I, 1+1*I, 3+0*I};
    float complex Bc[4] = {1+0*I,2+0*I,3+0*I,4+0*I};
    float complex Cc[4] = {0,0,0,0};
    float complex Ec[4] = {5+3*I,8+4*I,10+1*I,14+2*I};
    float complex alpha = 1+0*I, beta = 0+0*I;
    cblas_csymm(CblasRowMajor, CblasLeft, CblasUpper, 2, 2, &alpha, Ac, 2, Bc, 2, &beta, Cc, 2);
    ok_cfloat("cblas_csymm complex symmetric", Cc, Ec, 4, 1e-4f);

    double complex Az[4] = {2+0*I, 1+1*I, 1+1*I, 3+0*I};
    double complex Bz[4] = {1+0*I,2+0*I,3+0*I,4+0*I};
    double complex Cz[4] = {0,0,0,0};
    double complex Ez[4] = {5+3*I,8+4*I,10+1*I,14+2*I};
    double complex alphaz = 1+0*I, betaz = 0+0*I;
    cblas_zsymm(CblasRowMajor, CblasLeft, CblasUpper, 2, 2, &alphaz, Az, 2, Bz, 2, &betaz, Cz, 2);
    ok_cdouble("cblas_zsymm complex symmetric", Cz, Ez, 4, 1e-10);
}

static void test_hemm(void) {
    float complex A[4] = {2+0*I, 1+1*I, 1-1*I, 3+0*I};
    float complex B[4] = {1+0*I,2+0*I,3+0*I,4+0*I};
    float complex C[4] = {0,0,0,0};
    float complex E[4] = {5+3*I,8+4*I,10-1*I,14-2*I};
    float complex alpha = 1+0*I, beta = 0+0*I;
    cblas_chemm(CblasRowMajor, CblasLeft, CblasUpper, 2, 2, &alpha, A, 2, B, 2, &beta, C, 2);
    ok_cfloat("cblas_chemm Hermitian complex float", C, E, 4, 1e-4f);

    double complex Az[4] = {2+0*I, 1+1*I, 1-1*I, 3+0*I};
    double complex Bz[4] = {1+0*I,2+0*I,3+0*I,4+0*I};
    double complex Cz[4] = {0,0,0,0};
    double complex Ez[4] = {5+3*I,8+4*I,10-1*I,14-2*I};
    double complex alphaz = 1+0*I, betaz = 0+0*I;
    cblas_zhemm(CblasRowMajor, CblasLeft, CblasUpper, 2, 2, &alphaz, Az, 2, Bz, 2, &betaz, Cz, 2);
    ok_cdouble("cblas_zhemm Hermitian complex double", Cz, Ez, 4, 1e-10);
}

static void test_syrk_herk(void) {
    float A[6] = {1,2,3, 4,5,6};
    float C[4] = {0,0,0,0};
    float E[4] = {14,32,32,77};
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 3, 1.0f, A, 3, 0.0f, C, 2);
    ok_float_tri("cblas_ssyrk Uplo=Upper NoTrans", C, E, 2, CblasUpper, 1e-4f);

    double Ad[6] = {1,2,3, 4,5,6};
    double Cd[4] = {0,0,0,0};
    double Ed[4] = {14,32,32,77};
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, 2, 3, 1.0, Ad, 3, 0.0, Cd, 2);
    ok_double_tri("cblas_dsyrk Uplo=Lower NoTrans", Cd, Ed, 2, CblasLower, 1e-10);

    float complex Ac[4] = {1+1*I,2+0*I,3+0*I,4-I};
    float complex Cc[4] = {0,0,0,0};
    float complex Ec[4] = {4+2*I, 11+1*I, 11+1*I, 24-8*I};
    float complex alpha = 1+0*I, beta = 0+0*I;
    cblas_csyrk(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2, &alpha, Ac, 2, &beta, Cc, 2);
    ok_cfloat_tri("cblas_csyrk complex symmetric rank-k", Cc, Ec, 2, CblasUpper, 1e-4f);

    double complex Az[4] = {1+1*I,2+0*I,3+0*I,4-I};
    double complex Cz[4] = {0,0,0,0};
    double complex Ez[4] = {4+2*I, 11+1*I, 11+1*I, 24-8*I};
    double complex alphaz = 1+0*I, betaz = 0+0*I;
    cblas_zsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2, &alphaz, Az, 2, &betaz, Cz, 2);
    ok_cdouble_tri("cblas_zsyrk complex symmetric rank-k", Cz, Ez, 2, CblasUpper, 1e-10);

    float complex Ah[4] = {1+1*I,2+0*I,3+0*I,4-I};
    float complex Ch[4] = {0,0,0,0};
    float complex Eh[4] = {6+0*I, 11+5*I, 11-5*I, 26+0*I};
    cblas_cherk(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2, 1.0f, Ah, 2, 0.0f, Ch, 2);
    ok_cfloat_tri("cblas_cherk Hermitian rank-k", Ch, Eh, 2, CblasUpper, 1e-4f);

    double complex Azh[4] = {1+1*I,2+0*I,3+0*I,4-I};
    double complex Czh[4] = {0,0,0,0};
    double complex Ezh[4] = {6+0*I, 11+5*I, 11-5*I, 26+0*I};
    cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2, 1.0, Azh, 2, 0.0, Czh, 2);
    ok_cdouble_tri("cblas_zherk Hermitian rank-k", Czh, Ezh, 2, CblasUpper, 1e-10);
}

static void test_syr2k_her2k(void) {
    float A[4] = {1,2,3,4};
    float B[4] = {5,6,7,8};
    float C[4] = {0,0,0,0};
    float E[4] = {34,62,62,106};
    cblas_ssyr2k(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2, 1.0f, A, 2, B, 2, 0.0f, C, 2);
    ok_float_tri("cblas_ssyr2k Uplo=Upper", C, E, 2, CblasUpper, 1e-4f);

    double Ad[4] = {1,2,3,4};
    double Bd[4] = {5,6,7,8};
    double Cd[4] = {0,0,0,0};
    double Ed[4] = {34,62,62,106};
    cblas_dsyr2k(CblasRowMajor, CblasLower, CblasNoTrans, 2, 2, 1.0, Ad, 2, Bd, 2, 0.0, Cd, 2);
    ok_double_tri("cblas_dsyr2k Uplo=Lower", Cd, Ed, 2, CblasLower, 1e-10);

    float complex Ac[4] = {1+I,2,3,4-I};
    float complex Bc[4] = {5,6+I,7-I,8};
    float complex Cc[4] = {0,0,0,0};
    float complex Ec[4];
    for (int i=0;i<2;i++) for (int j=0;j<2;j++) {
        float complex s = 0;
        for (int k=0;k<2;k++) s += Ac[i*2+k]*Bc[j*2+k] + Bc[i*2+k]*Ac[j*2+k];
        Ec[i*2+j]=s;
    }
    float complex alpha = 1+0*I, beta = 0;
    cblas_csyr2k(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2, &alpha, Ac, 2, Bc, 2, &beta, Cc, 2);
    ok_cfloat_tri("cblas_csyr2k complex symmetric rank-2k", Cc, Ec, 2, CblasUpper, 1e-4f);

    double complex Az[4] = {1+I,2,3,4-I};
    double complex Bz[4] = {5,6+I,7-I,8};
    double complex Cz[4] = {0,0,0,0};
    double complex Ez[4];
    for (int i=0;i<2;i++) for (int j=0;j<2;j++) {
        double complex s = 0;
        for (int k=0;k<2;k++) s += Az[i*2+k]*Bz[j*2+k] + Bz[i*2+k]*Az[j*2+k];
        Ez[i*2+j]=s;
    }
    double complex alphaz = 1+0*I, betaz = 0;
    cblas_zsyr2k(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2, &alphaz, Az, 2, Bz, 2, &betaz, Cz, 2);
    ok_cdouble_tri("cblas_zsyr2k complex symmetric rank-2k", Cz, Ez, 2, CblasUpper, 1e-10);

    float complex Ch[4] = {0,0,0,0};
    float complex Eh[4];
    float complex ah = 1.0f + 0.5f*I;
    for (int i=0;i<2;i++) for (int j=0;j<2;j++) {
        float complex s1 = 0, s2 = 0;
        for (int k=0;k<2;k++) { s1 += Ac[i*2+k] * conjf(Bc[j*2+k]); s2 += Bc[i*2+k] * conjf(Ac[j*2+k]); }
        Eh[i*2+j] = ah*s1 + conjf(ah)*s2;
    }
    cblas_cher2k(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2, &ah, Ac, 2, Bc, 2, 0.0f, Ch, 2);
    ok_cfloat_tri("cblas_cher2k Hermitian rank-2k", Ch, Eh, 2, CblasUpper, 1e-4f);

    double complex Czh[4] = {0,0,0,0};
    double complex Ezh[4];
    double complex azh = 1.0 + 0.5*I;
    for (int i=0;i<2;i++) for (int j=0;j<2;j++) {
        double complex s1 = 0, s2 = 0;
        for (int k=0;k<2;k++) { s1 += Az[i*2+k] * conj(Bz[j*2+k]); s2 += Bz[i*2+k] * conj(Az[j*2+k]); }
        Ezh[i*2+j] = azh*s1 + conj(azh)*s2;
    }
    cblas_zher2k(CblasRowMajor, CblasUpper, CblasNoTrans, 2, 2, &azh, Az, 2, Bz, 2, 0.0, Czh, 2);
    ok_cdouble_tri("cblas_zher2k Hermitian rank-2k", Czh, Ezh, 2, CblasUpper, 1e-10);
}

static void test_trmm_trsm_real(void) {
    float Af[4] = {2,0, 0,4};
    float Bf[4] = {1,2, 3,4};
    float Etrmmf[4] = {2,4, 12,16};
    cblas_strmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 2, 2, 1.0f, Af, 2, Bf, 2);
    ok_float("cblas_strmm triangular matrix multiply", Bf, Etrmmf, 4, 1e-4f);

    float Bsf[4] = {2,4, 12,16};
    float Etrsmf[4] = {1,2,3,4};
    cblas_strsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 2, 2, 1.0f, Af, 2, Bsf, 2);
    ok_float("cblas_strsm triangular solve", Bsf, Etrsmf, 4, 1e-4f);

    double Ad[4] = {2,0, 0,4};
    double Bd[4] = {1,2, 3,4};
    double Etrmmd[4] = {2,4, 12,16};
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 2, 2, 1.0, Ad, 2, Bd, 2);
    ok_real("cblas_dtrmm Uplo=Lower", Bd, Etrmmd, 4, 1e-10);

    double Bsd[4] = {2,4, 12,16};
    double Etrsmd[4] = {1,2,3,4};
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 2, 2, 1.0, Ad, 2, Bsd, 2);
    ok_real("cblas_dtrsm Uplo=Lower", Bsd, Etrsmd, 4, 1e-10);
}

static void test_trmm_trsm_complex(void) {
    float complex A[4] = {2+0*I,0,0,4+0*I};
    float complex B[4] = {1,2,3,4};
    float complex E[4] = {2,4,12,16};
    float complex alpha = 1+0*I;
    cblas_ctrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, 2, 2, &alpha, A, 2, B, 2);
    ok_cfloat("cblas_ctrmm ConjTrans complex", B, E, 4, 1e-4f);

    float complex Bs[4] = {2,4,12,16};
    float complex Es[4] = {1,2,3,4};
    cblas_ctrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit, 2, 2, &alpha, A, 2, Bs, 2);
    ok_cfloat("cblas_ctrsm ConjTrans complex", Bs, Es, 4, 1e-4f);

    double complex Az[4] = {2+0*I,0,0,4+0*I};
    double complex Bz[4] = {1,2,3,4};
    double complex Ez[4] = {2,8,6,16};
    double complex alphaz = 1+0*I;
    cblas_ztrmm(CblasRowMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, 2, 2, &alphaz, Az, 2, Bz, 2);
    ok_cdouble("cblas_ztrmm Side=Right Trans", Bz, Ez, 4, 1e-10);

    double complex Bsz[4] = {2,4,12,16};
    double complex Esz[4] = {1,1,6,4};
    cblas_ztrsm(CblasRowMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, 2, 2, &alphaz, Az, 2, Bsz, 2);
    ok_cdouble("cblas_ztrsm Side=Right Trans", Bsz, Esz, 4, 1e-10);

    double complex Bunit[4] = {1,2,3,4};
    double complex Eunit[4] = {1,2,3,4};
    cblas_ztrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit, 2, 2, &alphaz, Az, 2, Bunit, 2);
    ok_cdouble("cblas_ztrsm Diag=Unit", Bunit, Eunit, 4, 1e-10);
}

static void run_all_tests_for_thread_count(int nthreads) {
    set_threads_safe(nthreads);
    printf("\n=== Running CBLAS Level 3 interface tests, threads=%d ===\n", nthreads);
    test_gemm_real();
    test_gemm_complex();
    test_symm();
    test_hemm();
    test_syrk_herk();
    test_syr2k_her2k();
    test_trmm_trsm_real();
    test_trmm_trsm_complex();
}

int main(void) {
    printf("CBLAS Level 3 interface test suite for OpenBLAS\n");
    printf("Coverage: GEMM, SYMM, HEMM, SYRK, HERK, SYR2K, HER2K, TRMM, TRSM; S/D/C/Z where supported.\n");
    run_all_tests_for_thread_count(1);
    run_all_tests_for_thread_count(2);
    printf("\nTotal checks: %d\n", checks);
    printf("Failures: %d\n", fails);
    if (fails == 0) {
        printf("RESULT: ALL TESTS PASSED\n");
        return 0;
    }
    printf("RESULT: TESTS FAILED AS EXPECTED FOR BROKEN OR INCOMPATIBLE LIBRARY\n");
    return 1;
}
