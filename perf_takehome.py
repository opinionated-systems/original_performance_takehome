# moonshotai/Kimi-K2.6

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def get_slot_writes(self, slot):
        engine, args = slot[0], slot[1]
        if engine == 'debug':
            return set()
        op = args[0]
        if engine == 'alu':
            return {args[1]}
        elif engine == 'valu':
            if op == 'vbroadcast':
                return set(range(args[1], args[1] + VLEN))
            elif op == 'multiply_add':
                return set(range(args[1], args[1] + VLEN))
            else:
                return set(range(args[1], args[1] + VLEN))
        elif engine == 'load':
            if op == 'load':
                return {args[1]}
            elif op == 'load_offset':
                return {args[1] + args[3]}
            elif op == 'vload':
                return set(range(args[1], args[1] + VLEN))
            elif op == 'const':
                return {args[1]}
        elif engine == 'store':
            return set()
        elif engine == 'flow':
            if op == 'select':
                return {args[1]}
            elif op == 'vselect':
                return set(range(args[1], args[1] + VLEN))
            elif op == 'add_imm':
                return {args[1]}
        return set()

    def get_slot_reads(self, slot):
        engine, args = slot[0], slot[1]
        if engine == 'debug':
            return set()
        op = args[0]
        if engine == 'alu':
            return {args[2], args[3]}
        elif engine == 'valu':
            if op == 'vbroadcast':
                return {args[2]}
            elif op == 'multiply_add':
                reads = set()
                for vec in [args[2], args[3], args[4]]:
                    reads.update(range(vec, vec + VLEN))
                return reads
            else:
                reads = set()
                for vec in [args[2], args[3]]:
                    reads.update(range(vec, vec + VLEN))
                return reads
        elif engine == 'load':
            if op == 'load':
                return {args[2]}
            elif op == 'load_offset':
                return {args[2] + args[3]}
            elif op == 'vload':
                return {args[2]}
            elif op == 'const':
                return set()
        elif engine == 'store':
            if op == 'store':
                return {args[1], args[2]}
            elif op == 'vstore':
                reads = {args[1]}
                reads.update(range(args[2], args[2] + VLEN))
                return reads
        elif engine == 'flow':
            if op == 'select':
                return {args[2], args[3], args[4]}
            elif op == 'vselect':
                reads = set()
                for vec in [args[2], args[3], args[4]]:
                    reads.update(range(vec, vec + VLEN))
                return reads
            elif op == 'add_imm':
                return {args[2]}
        return set()

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        reg_last_read = {}
        reg_last_write = {}
        last_mem_write = -1
        last_mem_read = -1

        cycles = []

        for i, slot in enumerate(slots):
            engine = slot[0]
            slot_writes = self.get_slot_writes(slot)
            slot_reads = self.get_slot_reads(slot)

            is_mem_read = engine == 'load' and slot[1][0] in ('load', 'load_offset', 'vload')
            is_mem_write = engine == 'store'

            earliest_cycle = 0

            for r in slot_reads:
                if r in reg_last_write:
                    earliest_cycle = max(earliest_cycle, reg_last_write[r] + 1)

            for w in slot_writes:
                if w in reg_last_write:
                    earliest_cycle = max(earliest_cycle, reg_last_write[w] + 1)

            for w in slot_writes:
                if w in reg_last_read:
                    earliest_cycle = max(earliest_cycle, reg_last_read[w])

            if is_mem_read:
                earliest_cycle = max(earliest_cycle, last_mem_write + 1)
            if is_mem_write:
                earliest_cycle = max(earliest_cycle, last_mem_write + 1, last_mem_read)

            c = earliest_cycle
            while True:
                if c == len(cycles):
                    cycles.append(defaultdict(list))

                if len(cycles[c][engine]) < SLOT_LIMITS[engine]:
                    cycles[c][engine].append(slot[1])

                    for r in slot_reads:
                        reg_last_read[r] = max(reg_last_read.get(r, 0), c)
                    for w in slot_writes:
                        reg_last_write[w] = c

                    if is_mem_read:
                        last_mem_read = max(last_mem_read, c)
                    if is_mem_write:
                        last_mem_write = max(last_mem_write, c)

                    break
                c += 1

        instrs = []
        for c in cycles:
            if c:
                instrs.append(dict(c))
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_vliw(self, slots):
        instrs = self.build(slots, vliw=True)
        self.instrs.extend(instrs)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        if self.scratch_ptr > SCRATCH_SIZE:
            raise RuntimeError(f'Out of scratch space: {self.scratch_ptr} > {SCRATCH_SIZE}')
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add('load', ('const', addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int,
        c_flow3=4, c_flow4=4, c_flow5=11,
        alu_chunks=4
    ):
        tmp1 = self.alloc_scratch('tmp1')

        # OPTIMIZATION: Only allocate the pointer variables we actually use
        # Memory layout: [0]=rounds, [1]=n_nodes, [2]=batch_size, [3]=forest_height,
        #                [4]=forest_values_p, [5]=inp_indices_p, [6]=inp_values_p
        # We skip allocating unused variables: rounds, n_nodes, batch_size, forest_height
        used_vars = ['forest_values_p', 'inp_indices_p', 'inp_values_p']
        for v in used_vars:
            self.alloc_scratch(v, 1)

        needed_consts = [0, 1, 2, 3, 4, 5, 6, 8, 9, 16, 19, 24] + list(range(7, 31))
        needed_consts = list(set(needed_consts))
        consts = {}
        for i in needed_consts:
            addr = self.alloc_scratch(f'c{i}')
            consts[i] = addr
            self.const_map[i] = addr

        hc_vals = [0x7ED55D16, 0xC761C23C, 0xFD7046C5, 0xB55A4F09, 4097, 33, 0xE9F8CC1D, 0xACCF6200, 16896]
        hash_consts = {}
        for val in hc_vals:
            addr = self.alloc_scratch(f'hc_{val & 0xFFFF}')
            self.const_map[val] = addr
            hash_consts[val] = addr
        hash_consts.update({9: consts[9], 16: consts[16], 19: consts[19]})

        num_chunks = (batch_size + VLEN - 1) // VLEN

        chunk_regs = []
        for c in range(num_chunks):
            regs = {
                'idx':  self.alloc_scratch(f'idx_{c}', VLEN),
                'val':  self.alloc_scratch(f'val_{c}', VLEN),
                'nv':   self.alloc_scratch(f'nv_{c}', VLEN),
                'tmp':  self.alloc_scratch(f'tmp_{c}', VLEN),
            }
            chunk_regs.append(regs)

        tmp_vecs = [self.alloc_scratch(f'tmp_vec_{i}', VLEN) for i in range(4)]

        vec_consts = {}
        for i in [0, 1, 2, 3, 4, 5, 6, 9, 16, 19]:
            vec_consts[i] = self.alloc_scratch(f'v{i}', VLEN)

        vec_4097 = self.alloc_scratch('v_4097', VLEN)
        vec_33 = self.alloc_scratch('v_33', VLEN)

        vec_hash = {}
        for val in hc_vals:
            if val not in (4097, 33):
                vec_hash[val] = self.alloc_scratch(f'vh_{val & 0xFFFF}', VLEN)

        PRELOADED_NODES = 31
        vec_nodes = {}
        for i in range(PRELOADED_NODES):
            vec_nodes[i] = self.alloc_scratch(f'vn_{i}', VLEN)

        # PEER OPTIMIZATION: Precompute diff_2_1 = vec_nodes[2] - vec_nodes[1]
        diff_2_1 = self.alloc_scratch('diff_2_1', VLEN)
        
        ALU_CHUNKS = min(num_chunks, alu_chunks)

        C_FLOW3 = min(num_chunks, c_flow3)
        C_LOAD3 = min(num_chunks - C_FLOW3, 27)
        C_LOAD3_END = C_FLOW3 + C_LOAD3

        C_FLOW4 = min(num_chunks, c_flow4)
        C_LOAD4 = min(num_chunks - C_FLOW4, 29)
        C_LOAD4_END = C_FLOW4 + C_LOAD4

        # ---- INITIALIZATION ----
        init_slots = []

        # OPTIMIZATION: Only load the pointer values we need (positions 4, 5, 6)
        for i, v in enumerate(used_vars):
            mem_pos = 4 + i
            init_slots.append(('load', ('const', tmp1, mem_pos)))
            init_slots.append(('load', ('load', self.scratch[v], tmp1)))

        for i in needed_consts:
            init_slots.append(('load', ('const', consts[i], i)))
        for val in hc_vals:
            init_slots.append(('load', ('const', hash_consts[val], val)))

        for i in [0, 1, 2, 3, 4, 5, 6, 9, 16, 19]:
            init_slots.append(('valu', ('vbroadcast', vec_consts[i], consts[i])))
        init_slots.append(('valu', ('vbroadcast', vec_4097, hash_consts[4097])))
        init_slots.append(('valu', ('vbroadcast', vec_33, hash_consts[33])))
        for val in hc_vals:
            if val not in (4097, 33):
                init_slots.append(('valu', ('vbroadcast', vec_hash[val], hash_consts[val])))

        for i in range(4):
            addr_reg = chunk_regs[i % 4]['tmp']
            init_slots.append(('alu', ('+', addr_reg, self.scratch['forest_values_p'], consts[i*8])))

        self.add_vliw(init_slots)

        # EXP 6: Interleave vloads and broadcasts for better scheduling overlap
        all_load_broadcast_slots = []
        for i in range(4):
            addr_reg = chunk_regs[i % min(num_chunks, 4)]['tmp']
            all_load_broadcast_slots.append(('load', ('vload', tmp_vecs[0], addr_reg)))
            for j in range(8):
                node_idx = i * 8 + j
                if node_idx < PRELOADED_NODES:
                    all_load_broadcast_slots.append(('valu', ('vbroadcast', vec_nodes[node_idx], tmp_vecs[0] + j)))
        self.add_vliw(all_load_broadcast_slots)

        # PEER OPTIMIZATION: Precompute diff_2_1 after loading vec_nodes
        self.add_vliw([('valu', ('-', diff_2_1, vec_nodes[2], vec_nodes[1]))])
        self.add('flow', ('pause',))

        scalar_reg = self.alloc_scratch('scalar_reg')
        addr_reg = self.alloc_scratch('addr_reg')

        body = []
        fvp = self.scratch['forest_values_p']

        ptr_idx = self.alloc_scratch('ptr_idx')
        ptr_val = self.alloc_scratch('ptr_val')
        ptr_idx_st = self.alloc_scratch('ptr_idx_st')
        ptr_val_st = self.alloc_scratch('ptr_val_st')

        body.append(('alu', ('+', ptr_idx, self.scratch['inp_indices_p'], consts[0])))
        body.append(('alu', ('+', ptr_val, self.scratch['inp_values_p'], consts[0])))
        body.append(('alu', ('+', ptr_idx_st, self.scratch['inp_indices_p'], consts[0])))
        body.append(('alu', ('+', ptr_val_st, self.scratch['inp_values_p'], consts[0])))

        for chunk in range(num_chunks):
            body.append(('load', ('vload', chunk_regs[chunk]['idx'], ptr_idx)))
            body.append(('load', ('vload', chunk_regs[chunk]['val'], ptr_val)))
            if chunk < num_chunks - 1:
                body.append(('alu', ('+', ptr_idx, ptr_idx, consts[8])))
                body.append(('alu', ('+', ptr_val, ptr_val, consts[8])))

        wrap_round = forest_height + 1

        def gen_hash_alu(chunk_idx, node_val_is_scalar=False, node_val_scalar=None):
            regs = chunk_regs[chunk_idx]
            for vi in range(VLEN):
                val = regs['val'] + vi
                nv  = regs['nv'] + vi
                tmp = regs['tmp'] + vi
                if node_val_is_scalar:
                    body.append(('alu', ('^', val, val, node_val_scalar)))
                else:
                    body.append(('alu', ('^', val, val, nv)))
                body.append(('alu', ('*', tmp, val, hash_consts[4097])))
                body.append(('alu', ('+', val, tmp, hash_consts[0x7ED55D16])))
                body.append(('alu', ('^', nv, val, hash_consts[0xC761C23C])))
                body.append(('alu', ('>>', tmp, val, hash_consts[19])))
                body.append(('alu', ('^', val, nv, tmp)))
                body.append(('alu', ('*', nv, val, hash_consts[33])))
                body.append(('alu', ('+', nv, nv, hash_consts[0xE9F8CC1D])))
                body.append(('alu', ('*', tmp, val, hash_consts[16896])))
                body.append(('alu', ('+', tmp, tmp, hash_consts[0xACCF6200])))
                body.append(('alu', ('^', val, nv, tmp)))
                body.append(('alu', ('*', tmp, val, hash_consts[9])))
                body.append(('alu', ('+', val, tmp, hash_consts[0xFD7046C5])))
                body.append(('alu', ('^', nv, val, hash_consts[0xB55A4F09])))
                body.append(('alu', ('>>', tmp, val, hash_consts[16])))
                body.append(('alu', ('^', val, nv, tmp)))

        def gen_hash_valu(chunk_idx, node_val_is_vec=False, node_val_vec=None):
            regs = chunk_regs[chunk_idx]
            if node_val_is_vec:
                body.append(('valu', ('^', regs['val'], regs['val'], node_val_vec)))
            else:
                body.append(('valu', ('^', regs['val'], regs['val'], regs['nv'])))
            body.append(('valu', ('multiply_add', regs['val'], regs['val'], vec_4097, vec_hash[0x7ED55D16])))
            body.append(('valu', ('^', regs['nv'], regs['val'], vec_hash[0xC761C23C])))
            body.append(('valu', ('>>', regs['tmp'], regs['val'], vec_consts[19])))
            body.append(('valu', ('^', regs['val'], regs['nv'], regs['tmp'])))
            body.append(('valu', ('multiply_add', regs['nv'], regs['val'], vec_33, vec_hash[0xE9F8CC1D])))
            body.append(('valu', ('multiply_add', regs['tmp'], regs['val'], vec_hash[16896], vec_hash[0xACCF6200])))
            body.append(('valu', ('^', regs['val'], regs['nv'], regs['tmp'])))
            body.append(('valu', ('multiply_add', regs['val'], regs['val'], vec_consts[9], vec_hash[0xFD7046C5])))
            body.append(('valu', ('^', regs['nv'], regs['val'], vec_hash[0xB55A4F09])))
            body.append(('valu', ('>>', regs['tmp'], regs['val'], vec_consts[16])))
            body.append(('valu', ('^', regs['val'], regs['nv'], regs['tmp'])))

        def gen_idx_update_alu(chunk_idx, wrap=False):
            regs = chunk_regs[chunk_idx]
            for vi in range(VLEN):
                val = regs['val'] + vi
                nv  = regs['nv'] + vi
                idx = regs['idx'] + vi
                tmp = regs['tmp'] + vi
                body.append(('alu', ('&', nv, val, consts[1])))
                body.append(('alu', ('*', tmp, idx, consts[2])))
                body.append(('alu', ('+', idx, tmp, consts[1])))
                body.append(('alu', ('+', idx, idx, nv)))
                if wrap:
                    body.append(('alu', ('&', idx, idx, consts[0])))

        def gen_idx_update_valu(chunk_idx, wrap=False):
            regs = chunk_regs[chunk_idx]
            body.append(('valu', ('&', regs['nv'], regs['val'], vec_consts[1])))
            body.append(('valu', ('multiply_add', regs['idx'], regs['idx'], vec_consts[2], vec_consts[1])))
            body.append(('valu', ('+', regs['idx'], regs['idx'], regs['nv'])))
            if wrap:
                body.append(('valu', ('&', regs['idx'], regs['idx'], vec_consts[0])))

        # PEER OPTIMIZATION: Use pure ALU arithmetic for Round 1 (not FLOW select)
        # This uses precomputed diff_2_1 = vec_nodes[2] - vec_nodes[1]
        # nv = (idx - 1) * diff_2_1 + vec_nodes[1]
        def gen_node_val_r1_alu(chunk_idx):
            regs = chunk_regs[chunk_idx]
            for vi in range(VLEN):
                idx = regs['idx'] + vi
                nv  = regs['nv'] + vi
                tmp = regs['tmp'] + vi
                body.append(('alu', ('-', tmp, idx, consts[1])))
                body.append(('alu', ('*', tmp, tmp, diff_2_1 + vi)))
                body.append(('alu', ('+', nv, tmp, vec_nodes[1] + vi)))

        def gen_node_val_r1_valu(chunk_idx):
            regs = chunk_regs[chunk_idx]
            body.append(('valu', ('==', regs['tmp'], regs['idx'], vec_consts[1])))
            body.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[1], vec_nodes[2])))

        # PEER OPTIMIZATION: Use pure ALU accumulation for Round 2 (not FLOW select)
        def gen_node_val_r2_alu(chunk_idx):
            regs = chunk_regs[chunk_idx]
            for vi in range(VLEN):
                idx = regs['idx'] + vi
                nv  = regs['nv'] + vi
                tmp = regs['tmp'] + vi
                body.append(('alu', ('==', tmp, idx, consts[3])))
                body.append(('alu', ('*', nv, tmp, vec_nodes[3])))
                for node in [4, 5, 6]:
                    body.append(('alu', ('==', tmp, idx, consts[node])))
                    body.append(('alu', ('*', tmp, tmp, vec_nodes[node])))
                    body.append(('alu', ('+', nv, nv, tmp)))

        def gen_node_val_r2_valu(chunk_idx):
            regs = chunk_regs[chunk_idx]
            body.append(('valu', ('==', regs['tmp'], regs['idx'], vec_consts[3])))
            body.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[3], vec_nodes[4])))
            body.append(('valu', ('==', regs['tmp'], regs['idx'], vec_consts[5])))
            body.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[5], regs['nv'])))
            body.append(('valu', ('==', regs['tmp'], regs['idx'], vec_consts[6])))
            body.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[6], regs['nv'])))

        def gen_node_val_flow(chunk_idx, start_node, num_nodes):
            regs = chunk_regs[chunk_idx]
            body.append(('valu', ('+', regs['nv'], vec_nodes[start_node], vec_consts[0])))
            for i in range(1, num_nodes):
                node_idx = start_node + i
                tmp_vec = tmp_vecs[(chunk_idx * num_nodes + i) % 4]
                body.append(('valu', ('vbroadcast', tmp_vec, consts[node_idx])))
                body.append(('valu', ('==', regs['tmp'], regs['idx'], tmp_vec)))
                body.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[node_idx], regs['nv'])))

        def gen_node_val_alu(chunk_idx, start_node, num_nodes):
            regs = chunk_regs[chunk_idx]
            for vi in range(VLEN):
                idx = regs['idx'] + vi
                nv  = regs['nv'] + vi
                tmp = regs['tmp'] + vi
                body.append(('alu', ('==', tmp, idx, consts[start_node])))
                body.append(('alu', ('*', nv, tmp, vec_nodes[start_node])))
                for i in range(1, num_nodes):
                    node_idx = start_node + i
                    body.append(('alu', ('==', tmp, idx, consts[node_idx])))
                    body.append(('alu', ('*', tmp, tmp, vec_nodes[node_idx])))
                    body.append(('alu', ('+', nv, nv, tmp)))

        # ---- Main round loop ----
        r = 0
        while r < rounds:
            if r == 0 or r == wrap_round:
                for chunk in range(ALU_CHUNKS, num_chunks):
                    regs = chunk_regs[chunk]
                    gen_hash_valu(chunk, node_val_is_vec=True, node_val_vec=vec_nodes[0])
                    body.append(('valu', ('&', regs['nv'], regs['val'], vec_consts[1])))
                    body.append(('valu', ('+', regs['idx'], regs['nv'], vec_consts[1])))
                for chunk in range(min(ALU_CHUNKS, num_chunks)):
                    regs = chunk_regs[chunk]
                    gen_hash_alu(chunk, node_val_is_scalar=True, node_val_scalar=vec_nodes[0])
                    for vi in range(VLEN):
                        val = regs['val'] + vi
                        nv  = regs['nv'] + vi
                        idx = regs['idx'] + vi
                        body.append(('alu', ('&', nv, val, consts[1])))
                        body.append(('alu', ('+', idx, nv, consts[1])))
                r += 1

            elif r == 1 or r == wrap_round + 1:
                for chunk in range(ALU_CHUNKS, num_chunks):
                    gen_node_val_r1_valu(chunk)
                    gen_hash_valu(chunk)
                    gen_idx_update_valu(chunk)
                for chunk in range(min(ALU_CHUNKS, num_chunks)):
                    gen_node_val_r1_alu(chunk)
                    gen_hash_alu(chunk)
                    gen_idx_update_alu(chunk)
                r += 1

            elif r == 2 or r == wrap_round + 2:
                for chunk in range(ALU_CHUNKS, num_chunks):
                    regs = chunk_regs[chunk]
                    gen_node_val_r2_valu(chunk)
                    gen_hash_valu(chunk)
                    gen_idx_update_valu(chunk)
                    if C_FLOW3 <= chunk < C_LOAD3_END:
                        for vi in range(VLEN):
                            body.append(('alu', ('+', regs['tmp'] + vi, fvp, regs['idx'] + vi)))
                for chunk in range(min(ALU_CHUNKS, num_chunks)):
                    gen_node_val_r2_alu(chunk)
                    gen_hash_alu(chunk)
                    gen_idx_update_alu(chunk)
                    if C_FLOW3 <= chunk < C_LOAD3_END:
                        regs = chunk_regs[chunk]
                        for vi in range(VLEN):
                            body.append(('alu', ('+', regs['tmp'] + vi, fvp, regs['idx'] + vi)))
                r += 1

            elif r == 3 or r == wrap_round + 3:
                c_flow = C_FLOW3
                c_load = C_LOAD3

                for chunk in range(c_flow, c_flow + c_load):
                    regs = chunk_regs[chunk]
                    for vi in range(VLEN):
                        body.append(('load', ('load', regs['nv'] + vi, regs['tmp'] + vi)))

                for chunk in range(num_chunks):
                    if chunk < c_flow:
                        gen_node_val_flow(chunk, 7, 8)
                    elif chunk < c_flow + c_load:
                        pass
                    else:
                        gen_node_val_alu(chunk, 7, 8)

                    if chunk >= num_chunks - ALU_CHUNKS:
                        gen_hash_alu(chunk)
                        gen_idx_update_alu(chunk)
                    else:
                        gen_hash_valu(chunk)
                        gen_idx_update_valu(chunk)

                    if C_FLOW4 <= chunk < C_LOAD4_END:
                        regs = chunk_regs[chunk]
                        for vi in range(VLEN):
                            body.append(('alu', ('+', regs['tmp'] + vi, fvp, regs['idx'] + vi)))
                r += 1

            elif r == 4 or r == wrap_round + 4:
                c_flow = C_FLOW4
                c_load = C_LOAD4
                is_last_round = (r == wrap_round + 4)

                for chunk in range(c_flow, c_flow + c_load):
                    regs = chunk_regs[chunk]
                    for vi in range(VLEN):
                        body.append(('load', ('load', regs['nv'] + vi, regs['tmp'] + vi)))

                for chunk in range(num_chunks):
                    if chunk < c_flow:
                        gen_node_val_flow(chunk, 15, 16)
                    elif chunk < c_flow + c_load:
                        pass
                    else:
                        gen_node_val_alu(chunk, 15, 16)

                    if chunk >= num_chunks - ALU_CHUNKS:
                        gen_hash_alu(chunk)
                        gen_idx_update_alu(chunk)
                    else:
                        gen_hash_valu(chunk)
                        gen_idx_update_valu(chunk)

                    if r == 4 and r + 1 < rounds and r + 1 != wrap_round:
                        regs = chunk_regs[chunk]
                        for vi in range(VLEN):
                            body.append(('alu', ('+', regs['tmp'] + vi, fvp, regs['idx'] + vi)))

                    # Store results interleaved with computation
                    if is_last_round:
                        regs = chunk_regs[chunk]
                        body.append(('store', ('vstore', ptr_idx_st, regs['idx'])))
                        body.append(('store', ('vstore', ptr_val_st, regs['val'])))
                        if chunk < num_chunks - 1:
                            body.append(('alu', ('+', ptr_idx_st, ptr_idx_st, consts[8])))
                            body.append(('alu', ('+', ptr_val_st, ptr_val_st, consts[8])))
                r += 1

            elif r == 5 or r == wrap_round + 5:
                c_flow = min(num_chunks, c_flow5)
                c_alu = min(num_chunks - c_flow, 1)
                c_load = num_chunks - c_flow - c_alu

                for chunk in range(c_flow):
                    regs = chunk_regs[chunk]
                    body.append(('valu', ('&', regs['nv'], regs['nv'], vec_consts[0])))

                for chunk in range(c_flow, c_flow + c_alu):
                    regs = chunk_regs[chunk]
                    for vi in range(VLEN):
                        body.append(('alu', ('&', regs['nv'] + vi, regs['nv'] + vi, consts[0])))

                for chunk in range(c_flow + c_alu, num_chunks):
                    regs = chunk_regs[chunk]
                    for vi in range(VLEN):
                        body.append(('load', ('load', regs['nv'] + vi, regs['tmp'] + vi)))

                body.append(('alu', ('+', scalar_reg, consts[24], consts[7])))
                body.append(('alu', ('+', addr_reg, consts[24], consts[7])))
                body.append(('alu', ('+', addr_reg, fvp, addr_reg)))

                for i in range(32):
                    if i % 8 == 0:
                        body.append(('load', ('vload', tmp_vecs[0], addr_reg)))
                        body.append(('alu', ('+', addr_reg, addr_reg, consts[8])))

                    body.append(('valu', ('vbroadcast', tmp_vecs[1], tmp_vecs[0] + (i % 8))))
                    body.append(('valu', ('vbroadcast', tmp_vecs[2], scalar_reg)))

                    for chunk in range(c_flow):
                        regs = chunk_regs[chunk]
                        body.append(('valu', ('==', regs['tmp'], regs['idx'], tmp_vecs[2])))
                        body.append(('flow', ('vselect', regs['nv'], regs['tmp'], tmp_vecs[1], regs['nv'])))

                    for chunk in range(c_flow, c_flow + c_alu):
                        regs = chunk_regs[chunk]
                        for vi in range(VLEN):
                            idx = regs['idx'] + vi
                            nv  = regs['nv'] + vi
                            tmp = regs['tmp'] + vi
                            body.append(('alu', ('==', tmp, idx, scalar_reg)))
                            body.append(('alu', ('*', tmp, tmp, tmp_vecs[1] + vi)))
                            body.append(('alu', ('+', nv, nv, tmp)))

                    body.append(('alu', ('+', scalar_reg, scalar_reg, consts[1])))

                for chunk in range(num_chunks):
                    if chunk >= num_chunks - ALU_CHUNKS:
                        gen_hash_alu(chunk)
                        gen_idx_update_alu(chunk)
                    else:
                        gen_hash_valu(chunk)
                        gen_idx_update_valu(chunk)

                    if r == 5 and r + 1 < rounds and r + 1 != wrap_round:
                        regs = chunk_regs[chunk]
                        for vi in range(VLEN):
                            body.append(('alu', ('+', regs['tmp'] + vi, fvp, regs['idx'] + vi)))
                r += 1

            else:
                # General scattered round handler
                do_wrap = (r == wrap_round - 1)
                is_last_scattered = (r == rounds - 1) or (r == wrap_round - 1 and wrap_round > rounds - 1)

                for chunk in range(num_chunks):
                    regs = chunk_regs[chunk]
                    for vi in range(VLEN):
                        body.append(('load', ('load', regs['nv'] + vi, regs['tmp'] + vi)))

                for chunk in range(num_chunks):
                    regs = chunk_regs[chunk]
                    if chunk >= num_chunks - ALU_CHUNKS:
                        gen_hash_alu(chunk)
                        if not do_wrap:
                            gen_idx_update_alu(chunk)
                            if not is_last_scattered and r + 1 < rounds and r + 1 != wrap_round:
                                for vi in range(VLEN):
                                    body.append(('alu', ('+', regs['tmp'] + vi, fvp, regs['idx'] + vi)))
                    else:
                        gen_hash_valu(chunk)
                        if not do_wrap:
                            gen_idx_update_valu(chunk)
                            if not is_last_scattered and r + 1 < rounds and r + 1 != wrap_round:
                                for vi in range(VLEN):
                                    body.append(('alu', ('+', regs['tmp'] + vi, fvp, regs['idx'] + vi)))

                r += 1

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)

        self.instrs.append({'flow': [('pause',)]})


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f'{forest_height=}, {rounds=}, {batch_size=}')
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : ref_mem[6] + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : ref_mem[6] + len(inp.values)]
        ), f'Incorrect result on round {i}'
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_values_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : ref_mem[5] + len(inp.indices)])

    print('CYCLES: ', machine.cycle)
    print('Speedup over baseline: ', BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == '__main__':
    unittest.main()
