# OpenBLAS CBLAS Level 3 Interface Tests

Проект содержит интерфейсные тесты на C99 для проверки части функционала OpenBLAS: CBLAS Level 3.

## Что проверяется

Тестируются функции:

- `gemm`: `cblas_sgemm`, `cblas_dgemm`, `cblas_cgemm`, `cblas_zgemm`
- `symm`: `cblas_ssymm`, `cblas_dsymm`, `cblas_csymm`, `cblas_zsymm`
- `hemm`: `cblas_chemm`, `cblas_zhemm`
- `syrk`: `cblas_ssyrk`, `cblas_dsyrk`, `cblas_csyrk`, `cblas_zsyrk`
- `herk`: `cblas_cherk`, `cblas_zherk`
- `syr2k`: `cblas_ssyr2k`, `cblas_dsyr2k`, `cblas_csyr2k`, `cblas_zsyr2k`
- `her2k`: `cblas_cher2k`, `cblas_zher2k`
- `trmm`: `cblas_strmm`, `cblas_dtrmm`, `cblas_ctrmm`, `cblas_ztrmm`
- `trsm`: `cblas_strsm`, `cblas_dtrsm`, `cblas_ctrsm`, `cblas_ztrsm`

Используются типы `float`, `double`, `float complex`, `double complex`, а также разные значения параметров `layout`, `side`, `uplo`, `trans`, `diag`, `alpha`, `beta`.

## Сборка OpenBLAS из исходников

```bash
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
make -j4
make PREFIX=$HOME/openblas install
```

## Сборка и запуск тестов с OpenBLAS

```bash
cd openblas_cblas_level3_tests
make clean
make OPENBLAS_PREFIX=$HOME/openblas test
```

Ожидаемый итог:

```text
RESULT: ALL TESTS PASSED
Covered CBLAS Level 3 functions: 30 / 30
```

## Дополнительное задание: неправильная библиотека

```bash
make clean
make OPENBLAS_PREFIX=$HOME/openblas wrong
```

Ожидаемый итог: часть тестов должна упасть, потому что библиотека `wrong_lib/wrong_cblas_level3.c` имеет совместимые имена функций, но не выполняет операции BLAS.


