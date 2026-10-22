/* SPDX-License-Identifier: Apache-2.0 */
/* The C quick-start program of README.md, as a file a user (or a test) can compile directly.
 *
 * Build and run it against an installed package:
 *
 *   export PKG_CONFIG_PATH=$(python -c "import eldpc, os; print(os.path.join(os.path.dirname(eldpc.__file__), 'lib', 'pkgconfig'))")
 *   cc example.c $(pkg-config --cflags --libs eldpc) \
 *      -Wl,-rpath,$(pkg-config --variable=libdir eldpc) -o example
 *   ./example
 *
 * It decodes one syndrome of a 3-detector, 5-mechanism problem and prints the correction, the
 * observable prediction and the BP iteration count.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "eldpc.h"

/* H = [[1,1,0,0,1],[0,1,1,0,0],[0,0,1,1,1]] and O = [[0,1,0,1,1]] in CSC. */
static const uint32_t h_colptr[6] = {0, 1, 3, 5, 6, 8};
static const uint32_t h_rowind[8] = {0, 0, 1, 1, 2, 2, 0, 2};
static const uint32_t o_colptr[6] = {0, 0, 1, 1, 2, 3};
static const uint32_t o_rowind[3] = {0, 0, 0};
static const double priors[5] = {0.05, 0.10, 0.02, 0.20, 0.01};

int main(void) {
    eldpc_problem p;
    memset(&p, 0, sizeof p);
    p.abi_version = ELDPC_API_VERSION;
    p.num_detectors = 3;
    p.num_mechanisms = 5;
    p.num_observables = 1;
    p.h_colptr = h_colptr;
    p.h_rowind = h_rowind;
    p.o_colptr = o_colptr;
    p.o_rowind = o_rowind;
    p.priors = priors;

    char err[512];
    eldpc_decoder *dec = eldpc_create(
        "bp_method=ms;decoder=bposd;max_iter=3;ms_scaling_factor=0.3125;"
        "osd_method=osd0;osd_order=0;schedule=parallel", &p, err, sizeof err);
    if (dec == NULL) {
        fprintf(stderr, "create: %s\n", err);
        return 1;
    }

    const uint8_t syndrome[3] = {1, 0, 0};
    uint8_t correction[5], observables[1];
    uint32_t status = 0;
    if (eldpc_decode(dec, syndrome, correction, observables, &status) != ELDPC_OK) return 1;

    printf("correction %u%u%u%u%u, observable %u, %u iterations\n",
           correction[0], correction[1], correction[2], correction[3], correction[4],
           observables[0], (unsigned)(status >> ELDPC_ST_ITER_SHIFT));
    eldpc_destroy(dec);
    return 0;
}
