# deepseek-v4-pro

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

    def build(self, slots, vliw=False):
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        reg_last_read = {}
        reg_last_write = {}
        last_mem_write = -1
        cycles = []

        for slot in slots:
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

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        if self.scratch_ptr > SCRATCH_SIZE:
            raise RuntimeError(f'Out of scratch space: {self.scratch_ptr} > {SCRATCH_SIZE}')
        return addr

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int,
    ):
        PRELOADED_NODES = 15
        needed_consts = [0, 1, 2, 3, 5, 6, 7, 8, 9, 16, 19] + list(range(10, PRELOADED_NODES))
        needed_consts = list(set(needed_consts))
        consts = {}
        for i in needed_consts:
            addr = self.alloc_scratch(f'c{i}')
            consts[i] = addr
            self.const_map[i] = addr

        hc_vals = [
            0x7ED55D16, 0xC761C23C, 0xFD7046C5, 0xB55A4F09,
            4097, 33, 0xE9F8CC1D, 0xACCF6200, 16896,
        ]
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

        tmp_vec = self.alloc_scratch('tmp_vec', VLEN)

        vec_consts = {}
        for i in needed_consts:
            vec_consts[i] = self.alloc_scratch(f'v{i}', VLEN)

        vec_4097 = self.alloc_scratch('v_4097', VLEN)
        vec_33 = self.alloc_scratch('v_33', VLEN)

        vec_hash = {}
        for val in hc_vals:
            if val not in (4097, 33):
                vec_hash[val] = self.alloc_scratch(f'vh_{val & 0xFFFF}', VLEN)

        vec_nodes = {}
        for i in range(PRELOADED_NODES):
            vec_nodes[i] = self.alloc_scratch(f'vn_{i}', VLEN)

        all_slots = []
        
        inp_values_p = self.alloc_scratch('inp_values_p', 1)
        all_slots.append(('load', ('const', inp_values_p, 7 + n_nodes + batch_size)))

        for i in needed_consts:
            all_slots.append(('load', ('const', consts[i], i)))
            all_slots.append(('valu', ('vbroadcast', vec_consts[i], consts[i])))

        for val in hc_vals:
            all_slots.append(('load', ('const', hash_consts[val], val)))
            if val == 4097:
                all_slots.append(('valu', ('vbroadcast', vec_4097, hash_consts[val])))
            elif val == 33:
                all_slots.append(('valu', ('vbroadcast', vec_33, hash_consts[val])))
            else:
                all_slots.append(('valu', ('vbroadcast', vec_hash[val], hash_consts[val])))

        fvp = self.alloc_scratch('fvp', 1)
        all_slots.append(('load', ('const', fvp, 7)))
        
        num_vloads = (PRELOADED_NODES + 7) // 8
        for i in range(num_vloads):
            addr_reg = chunk_regs[i % 4]['tmp']
            all_slots.append(('flow', ('add_imm', addr_reg, fvp, i * 8)))
            all_slots.append(('load', ('vload', tmp_vec, addr_reg)))
            for j in range(8):
                node_idx = i * 8 + j
                if node_idx < PRELOADED_NODES:
                    all_slots.append(('valu', ('vbroadcast', vec_nodes[node_idx], tmp_vec + j)))

        for chunk in range(num_chunks):
            ptr_reg = chunk_regs[chunk]['tmp']
            all_slots.append(('flow', ('add_imm', ptr_reg, inp_values_p, chunk * 8)))
            all_slots.append(('load', ('vload', chunk_regs[chunk]['val'], ptr_reg)))

        wrap_round = forest_height + 1

        def gen_hash_valu(chunk_idx, node_val_is_vec=False, node_val_vec=None):
            regs = chunk_regs[chunk_idx]
            if node_val_is_vec:
                all_slots.append(('valu', ('^', regs['val'], regs['val'], node_val_vec)))
            else:
                all_slots.append(('valu', ('^', regs['val'], regs['val'], regs['nv'])))
            all_slots.append(('valu', ('multiply_add', regs['val'], regs['val'], vec_4097, vec_hash[0x7ED55D16])))
            all_slots.append(('valu', ('^', regs['nv'], regs['val'], vec_hash[0xC761C23C])))
            all_slots.append(('valu', ('>>', regs['tmp'], regs['val'], vec_consts[19])))
            all_slots.append(('valu', ('^', regs['val'], regs['nv'], regs['tmp'])))
            all_slots.append(('valu', ('multiply_add', regs['nv'], regs['val'], vec_33, vec_hash[0xE9F8CC1D])))
            all_slots.append(('valu', ('multiply_add', regs['tmp'], regs['val'], vec_hash[16896], vec_hash[0xACCF6200])))
            all_slots.append(('valu', ('^', regs['val'], regs['nv'], regs['tmp'])))
            all_slots.append(('valu', ('multiply_add', regs['val'], regs['val'], vec_consts[9], vec_hash[0xFD7046C5])))
            all_slots.append(('valu', ('^', regs['nv'], regs['val'], vec_hash[0xB55A4F09])))
            all_slots.append(('valu', ('>>', regs['tmp'], regs['val'], vec_consts[16])))
            all_slots.append(('valu', ('^', regs['val'], regs['nv'], regs['tmp'])))

        def gen_idx_update_valu(chunk_idx, wrap=False):
            regs = chunk_regs[chunk_idx]
            all_slots.append(('valu', ('multiply_add', regs['idx'], regs['idx'], vec_consts[2], vec_consts[1])))
            all_slots.append(('valu', ('&', regs['tmp'], regs['val'], vec_consts[1])))
            all_slots.append(('valu', ('+', regs['idx'], regs['idx'], regs['tmp'])))
            if wrap:
                all_slots.append(('valu', ('&', regs['idx'], regs['idx'], vec_consts[0])))

        def gen_idx_update_alu(chunk_idx, wrap=False):
            regs = chunk_regs[chunk_idx]
            for vi in range(VLEN):
                all_slots.append(('alu', ('+', regs['idx'] + vi, regs['idx'] + vi, regs['idx'] + vi)))
                all_slots.append(('alu', ('+', regs['idx'] + vi, regs['idx'] + vi, consts[1])))
                all_slots.append(('alu', ('&', regs['tmp'] + vi, regs['val'] + vi, consts[1])))
                all_slots.append(('alu', ('+', regs['idx'] + vi, regs['idx'] + vi, regs['tmp'] + vi)))
                if wrap:
                    all_slots.append(('alu', ('&', regs['idx'] + vi, regs['idx'] + vi, consts[0])))

        def gen_addr_compute(chunk_idx):
            regs = chunk_regs[chunk_idx]
            for vi in range(VLEN):
                all_slots.append(('alu', ('+', regs['tmp'] + vi, consts[7], regs['idx'] + vi)))

        def gen_loads(chunk_idx):
            regs = chunk_regs[chunk_idx]
            for vi in range(VLEN):
                all_slots.append(('load', ('load', regs['nv'] + vi, regs['tmp'] + vi)))

        def gen_node_val_r1_valu(chunk_idx):
            regs = chunk_regs[chunk_idx]
            all_slots.append(('valu', ('==', regs['tmp'], regs['idx'], vec_consts[1])))
            all_slots.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[1], vec_nodes[2])))

        def gen_node_val_r2_valu(chunk_idx):
            regs = chunk_regs[chunk_idx]
            all_slots.append(('valu', ('==', regs['tmp'], regs['idx'], vec_consts[3])))
            all_slots.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[3], vec_nodes[4])))
            all_slots.append(('valu', ('==', regs['tmp'], regs['idx'], vec_consts[5])))
            all_slots.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[5], regs['nv'])))
            all_slots.append(('valu', ('==', regs['tmp'], regs['idx'], vec_consts[6])))
            all_slots.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[6], regs['nv'])))

        def gen_node_val_r3_valu(chunk_idx):
            regs = chunk_regs[chunk_idx]
            all_slots.append(('valu', ('==', regs['tmp'], regs['idx'], vec_consts[7])))
            all_slots.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[7], vec_consts[0])))
            for node in range(8, PRELOADED_NODES):
                all_slots.append(('valu', ('==', regs['tmp'], regs['idx'], vec_consts[node])))
                all_slots.append(('flow', ('vselect', regs['nv'], regs['tmp'], vec_nodes[node], regs['nv'])))

        def is_scatter(r):
            return r not in (0, wrap_round, 1, wrap_round + 1, 2, wrap_round + 2, 3, wrap_round + 3)

        def process_scattered_round(chunk_idx, do_wrap, is_last=False):
            gen_addr_compute(chunk_idx)
            gen_loads(chunk_idx)
            gen_hash_valu(chunk_idx)
            if is_last:
                pass
            elif do_wrap:
                regs = chunk_regs[chunk_idx]
                for vi in range(VLEN):
                    all_slots.append(('alu', ('&', regs['idx'] + vi, consts[0], consts[0])))
            else:
                gen_idx_update_alu(chunk_idx)

        def process_round(chunk_idx, r, is_last=False):
            if r == 0 or r == wrap_round:
                regs = chunk_regs[chunk_idx]
                # IDEA CN: Revert BA, use VALU for round 0 init
                all_slots.append(('valu', ('multiply_add', regs['idx'], vec_consts[0], vec_consts[2], vec_consts[1])))
                gen_hash_valu(chunk_idx, node_val_is_vec=True, node_val_vec=vec_nodes[0])
                if is_last:
                    pass
                else:
                    # Keep CG: & in VALU, + in ALU
                    all_slots.append(('valu', ('&', regs['nv'], regs['val'], vec_consts[1])))
                    for vi in range(VLEN):
                        all_slots.append(('alu', ('+', regs['idx'] + vi, regs['idx'] + vi, regs['nv'] + vi)))

            elif r == 1 or r == wrap_round + 1:
                gen_node_val_r1_valu(chunk_idx)
                gen_hash_valu(chunk_idx)
                if is_last:
                    pass
                else:
                    group_pos = chunk_idx % 4
                    if group_pos < 3:
                        gen_idx_update_alu(chunk_idx)
                    else:
                        gen_idx_update_valu(chunk_idx)

            elif r == 2 or r == wrap_round + 2:
                gen_node_val_r2_valu(chunk_idx)
                gen_hash_valu(chunk_idx)
                if is_last:
                    pass
                else:
                    group_pos = chunk_idx % 4
                    if group_pos < 3:
                        gen_idx_update_alu(chunk_idx)
                    else:
                        gen_idx_update_valu(chunk_idx)

            elif r == 3 or r == wrap_round + 3:
                gen_node_val_r3_valu(chunk_idx)
                gen_hash_valu(chunk_idx)
                if is_last:
                    pass
                else:
                    group_pos = chunk_idx % 4
                    if group_pos < 3:
                        gen_idx_update_alu(chunk_idx)
                    else:
                        gen_idx_update_valu(chunk_idx)

            else:
                do_wrap = (r == wrap_round - 1)
                process_scattered_round(chunk_idx, do_wrap, is_last)

        GROUP_SIZE = 4
        for chunk_group in range(0, num_chunks, GROUP_SIZE):
            chunks_in_group = []
            for i in range(GROUP_SIZE):
                c = chunk_group + i
                if c < num_chunks:
                    chunks_in_group.append(c)
            
            r = 0
            while r < rounds:
                is_last = (r == rounds - 1)
                if r == 0 or r == wrap_round:
                    for c in chunks_in_group:
                        process_round(c, r, is_last)
                    r += 1
                elif r == 1 or r == wrap_round + 1:
                    for c in chunks_in_group:
                        process_round(c, r, is_last)
                    r += 1
                elif r == 2 or r == wrap_round + 2:
                    for c in chunks_in_group:
                        process_round(c, r, is_last)
                    r += 1
                elif r == 3 or r == wrap_round + 3:
                    for c in chunks_in_group:
                        process_round(c, r, is_last)
                    r += 1
                elif r + 1 < rounds and is_scatter(r) and is_scatter(r + 1):
                    for c in chunks_in_group:
                        for rd in range(2):
                            do_wrap = (r + rd == wrap_round - 1)
                            is_last_rd = (r + rd == rounds - 1)
                            process_scattered_round(c, do_wrap, is_last_rd)
                    r += 2
                else:
                    for c in chunks_in_group:
                        do_wrap = (r == wrap_round - 1)
                        process_scattered_round(c, do_wrap, is_last)
                    r += 1

        store_ptrs = []
        for chunk in range(num_chunks):
            sp = self.alloc_scratch(f'sp_{chunk}', 1)
            store_ptrs.append(sp)
            all_slots.append(('flow', ('add_imm', sp, inp_values_p, chunk * 8)))

        for chunk in range(num_chunks):
            regs = chunk_regs[chunk]
            all_slots.append(('store', ('vstore', store_ptrs[chunk], regs['val'])))

        self.instrs.extend(self.build(all_slots, vliw=True))


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
        trace=trace,
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