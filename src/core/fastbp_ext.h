/*
 * fastbp_ext.h -- extensions of the fastbp GPU decoder libraries beyond the decoder ABI (decoder_abi.h).
 *
 * Every function here is OPTIONAL in the sense of decoder_abi.h's dbench_decoder_build_info: it is not one of
 * the ABI's required symbols (the ABI version stays 1), and a caller resolves it by name at run time (ctypes, the loader's symbol lookup); a
 * missing symbol means "not supported".  Names carry the stable prefix fastbp_ (not the slot's name, not
 * dbench_), so callers do not rename them from one library version to the next.
 */
#ifndef FASTBP_EXT_H
#define FASTBP_EXT_H

#include <stdint.h>

#include "decoder_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Flags of fastbp_keep_warm(). */
#define FASTBP_KEEP_WARM_GPU 1u  /* enqueue one empty 1-block kernel on the handle's stream */

/* Keep an idle handle's GPU out of its idle clock state between decodes.  Contract:
 * - The call enqueues one empty kernel on the handle's stream and returns: about 1.5 us of host time; no
 *   synchronisation, no allocation, no thread, no I/O.
 * - Call it every <= 0.5 ms while the handle is idle (sparser calls let the clocks drop again).  The last call
 *   is best placed shortly before the next decode.
 * - It never changes any result: the outputs of every later decode are bit-identical to a run without it (the
 *   kernel touches no memory and no state of the handle changes).
 * - Like every call on a handle, not concurrently with another call on the same handle.
 * flags = 0 does nothing.  Returns DBENCH_OK; DBENCH_EINVAL for a NULL handle or any flag bit other than
 * FASTBP_KEEP_WARM_GPU; DBENCH_EINTERNAL when the kernel launch failed (the handle stays usable: the next decode
 * reports any sticky device error itself). */
DBENCH_EXPORT int fastbp_keep_warm(dbench_decoder *h, uint32_t flags);

#ifdef __cplusplus
}
#endif

#endif /* FASTBP_EXT_H */
