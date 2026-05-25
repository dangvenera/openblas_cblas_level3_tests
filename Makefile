CC ?= gcc
AR ?= ar
CFLAGS ?= -std=c99 -O2 -Wall -Wextra
# Set this to the OpenBLAS install prefix, for example:
#   make OPENBLAS_PREFIX=$$HOME/openblas
OPENBLAS_PREFIX ?= $(HOME)/openblas
OPENBLAS_INC ?= $(OPENBLAS_PREFIX)/include
OPENBLAS_LIB ?= $(OPENBLAS_PREFIX)/lib

TEST_SRC = tests/test_cblas_level3.c
WRONG_SRC = wrong_lib/wrong_cblas_level3.c
BUILD_DIR = build

.PHONY: all test wrong test_wrong clean

all: $(BUILD_DIR)/test_openblas $(BUILD_DIR)/test_wrong

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/test_openblas: $(TEST_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(OPENBLAS_INC) $< -L$(OPENBLAS_LIB) -lopenblas -lm -o $@

$(BUILD_DIR)/wrong_cblas_level3.o: $(WRONG_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(OPENBLAS_INC) -c $< -o $@

$(BUILD_DIR)/libwrongcblas.a: $(BUILD_DIR)/wrong_cblas_level3.o
	$(AR) rcs $@ $<

$(BUILD_DIR)/test_wrong: $(TEST_SRC) $(BUILD_DIR)/libwrongcblas.a | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(OPENBLAS_INC) $(TEST_SRC) -L$(BUILD_DIR) -lwrongcblas -lm -o $@

test: $(BUILD_DIR)/test_openblas
	$(BUILD_DIR)/test_openblas | tee $(BUILD_DIR)/openblas_test_result.txt
	python3 scripts/check_coverage.py $(BUILD_DIR)/openblas_test_result.txt

wrong: $(BUILD_DIR)/test_wrong
	-$(BUILD_DIR)/test_wrong | tee $(BUILD_DIR)/wrong_library_test_result.txt
	python3 scripts/check_coverage.py $(BUILD_DIR)/wrong_library_test_result.txt || true

clean:
	rm -rf $(BUILD_DIR)
