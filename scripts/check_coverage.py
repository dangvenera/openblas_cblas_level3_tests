#!/usr/bin/env python3
import re
import sys
from pathlib import Path

required = [
    'cblas_sgemm', 'cblas_dgemm', 'cblas_cgemm', 'cblas_zgemm',
    'cblas_ssymm', 'cblas_dsymm', 'cblas_csymm', 'cblas_zsymm',
    'cblas_chemm', 'cblas_zhemm',
    'cblas_ssyrk', 'cblas_dsyrk', 'cblas_csyrk', 'cblas_zsyrk',
    'cblas_cherk', 'cblas_zherk',
    'cblas_ssyr2k', 'cblas_dsyr2k', 'cblas_csyr2k', 'cblas_zsyr2k',
    'cblas_cher2k', 'cblas_zher2k',
    'cblas_strmm', 'cblas_dtrmm', 'cblas_ctrmm', 'cblas_ztrmm',
    'cblas_strsm', 'cblas_dtrsm', 'cblas_ctrsm', 'cblas_ztrsm',
]

if len(sys.argv) != 2:
    print('Usage: python3 scripts/check_coverage.py build/openblas_test_result.txt')
    sys.exit(2)

text = Path(sys.argv[1]).read_text(encoding='utf-8', errors='ignore')
missing = [name for name in required if name not in text]
threads = sorted(set(re.findall(r'threads=(\d+)', text)))

print('\n=== COVERAGE REPORT ===')
print(f'Covered CBLAS Level 3 functions: {len(required) - len(missing)} / {len(required)}')
print('Threads used:', ', '.join(threads) if threads else 'not found')
if missing:
    print('Missing functions:')
    for name in missing:
        print('  -', name)
    sys.exit(1)
print('All required function names were executed by the interface tests.')
