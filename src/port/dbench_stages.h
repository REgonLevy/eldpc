/* SPDX-License-Identifier: Apache-2.0 */
/* Stage marks of the decoder-bench measurement build, as seen by the shipped library.
 *
 * The core sources mark the boundaries of their decode stages with the macros below. In the
 * reference construction's measurement build those macros expand to clock reads and two extra
 * exports; a shipped library carries neither. This header provides the shipped definitions: the
 * mark ids exist (the source text names them) and every macro is a no-op, so the object code is
 * that of the unmarked source and the library exports no dbench_stage_* symbol.
 */
#ifndef DBENCH_STAGES_H
#define DBENCH_STAGES_H

/* Mark ids, in decode order (the measurement build's names). */
#define DBENCH_STAGE_ENTRY 0
#define DBENCH_STAGE_PREP 1
#define DBENCH_STAGE_BP_BEGIN 2
#define DBENCH_STAGE_BP_END 3
#define DBENCH_STAGE_POSTERIOR 4
#define DBENCH_STAGE_ORDER_INIT 5
#define DBENCH_STAGE_WALK 6
#define DBENCH_STAGE_BACKSUB 7
#define DBENCH_STAGE_OSD_END 8
#define DBENCH_STAGE_OBS 9
#define DBENCH_STAGE_RET 10
#define DBENCH_STAGE_NMARK 16
#define DBENCH_STAGE_NACC 4
#define DBENCH_STAGE_ACC_ORDER 0

#define DBENCH_STAGE_BEGIN() ((void)0)
#define DBENCH_STAGE(id) ((void)0)
#define DBENCH_STAGE_ACC_BEGIN(i) ((void)0)
#define DBENCH_STAGE_ACC_END(i) ((void)0)

#endif /* DBENCH_STAGES_H */
