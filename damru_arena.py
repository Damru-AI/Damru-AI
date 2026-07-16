#!/usr/bin/env python3
'''
DAMRU ARENA -- Autonomous Self-Improvement Gym (self-play + execution-graded).

The real fire: Damru manufactures its OWN training data, unsupervised, and only
keeps rows it can PROVE correct by running code -- no human labels, no reliance
on another AI to judge truth. This is the self-learning flywheel that feeds the
20M-row corpus with HARD, VERIFIED examples.

Loop:  generate problem -> student solves -> GRADE BY EXECUTION -> adapt
       difficulty (curriculum) -> emit verified row -> repeat.

Five beast features:
  A. Self-Curriculum Generator -- invents fresh math/code problems, difficulty-scaled.
  B. Execution Grader          -- code is run in a sandbox; math is checked by a
                                  safe AST evaluator. Correctness is PROVEN.
  C. Adaptive Difficulty (Elo) -- holds the student at the edge of its ability
                                  (fastest learning), like AlphaZero self-play.
  D. Verified Data Emitter     -- only execution-verified solutions become
                                  training rows (reasoning-traces schema).
  E. Scorecard + Anti-Gaming   -- Elo, solve-rate, per-domain stats; de-dupes
                                  so the corpus never fills with the same row.

Dependency-free (Python standard library only).

Offline self-test:  python3 damru_arena.py
'''
from __future__ import annotations
import os
import re
import ast
import sys
import time
import random
import operator
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================ config
@dataclass
class ArenaConfig:
    rounds: int = 50
    difficulty_start: int = 2
    max_difficulty: int = 8
    window: int = 6
    target_lo: float = 0.5
    target_hi: float = 0.8
    seed: int = 7
    exec_timeout: float = 6.0
    domains: Tuple[str, ...] = ('math', 'code')

    @classmethod
    def from_env(cls) -> 'ArenaConfig':
        g = os.environ.get

        def _i(n, d):
            try:
                return int(g(n, str(d)))
            except Exception:
                return d

        def _f(n, d):
            try:
                return float(g(n, str(d)))
            except Exception:
                return d

        return cls(
            rounds=_i('ARENA_ROUNDS', 50),
            difficulty_start=_i('ARENA_DIFF_START', 2),
            max_difficulty=_i('ARENA_MAX_DIFF', 8),
            window=_i('ARENA_WINDOW', 6),
            target_lo=_f('ARENA_TARGET_LO', 0.5),
            target_hi=_f('ARENA_TARGET_HI', 0.8),
            seed=_i('ARENA_SEED', 7),
            exec_timeout=_f('ARENA_EXEC_TIMEOUT', 6.0),
        )


# ============================================================ safe math evaluator
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}


def safe_eval(expr: str):
    '''Evaluate a pure integer arithmetic expression with NO eval() and NO
    names/calls/attributes. Raises ValueError for anything unexpected.'''
    tree = ast.parse(expr, mode='eval').body
    return _ev(tree)


def _ev(n):
    if isinstance(n, ast.BinOp):
        op = _BIN_OPS.get(type(n.op))
        if op is None:
            raise ValueError('operator not allowed')
        a = _ev(n.left)
        b = _ev(n.right)
        if type(n.op) is ast.Pow and (abs(b) > 6 or abs(a) > 1000):
            raise ValueError('exponent too large')
        if type(n.op) in (ast.FloorDiv, ast.Mod) and b == 0:
            raise ValueError('division by zero')
        return op(a, b)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
        v = _ev(n.operand)
        return v if isinstance(n.op, ast.UAdd) else -v
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    raise ValueError('unsupported expression node')


# ============================================================ code runner
def _run_python(code: str, timeout: float = 6.0) -> Dict[str, Any]:
    '''Run code in an isolated subprocess (hard timeout); fall back to a plain
    in-process exec if spawning is unavailable.'''
    try:
        fd, path = tempfile.mkstemp(suffix='.py')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(code)
            proc = subprocess.run([sys.executable, '-I', '-S', path],
                                  capture_output=True, text=True, timeout=timeout)
            return {'ok': proc.returncode == 0, 'out': proc.stdout, 'err': proc.stderr}
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except subprocess.TimeoutExpired:
        return {'ok': False, 'out': '', 'err': 'timeout'}
    except Exception:
        import io
        import contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, '<arena>', 'exec'), {'__name__': '__main__'})
            return {'ok': True, 'out': buf.getvalue(), 'err': ''}
        except Exception as ex:
            return {'ok': False, 'out': buf.getvalue(), 'err': type(ex).__name__ + ': ' + str(ex)}


_CODE_RE = re.compile(r'```(?:python)?\s*([\s\S]*?)```', re.I)


def _extract_code_block(text: str) -> str:
    m = _CODE_RE.search(text or '')
    return (m.group(1) if m else (text or '')).strip()


# ============================================================ problem bank
# (difficulty, func, prompt, tests, reference_solution)
_CODE_BANK: List[Tuple[int, str, str, List[str], str]] = [
    (1, 'add', 'Write a function add(a, b) that returns their sum.',
     ['assert add(2, 3) == 5', 'assert add(-4, 4) == 0', 'assert add(100, 250) == 350'],
     'def add(a, b):\n    return a + b'),
    (2, 'reverse_str', 'Write reverse_str(s) that returns the string reversed.',
     ["assert reverse_str('abc') == 'cba'", "assert reverse_str('') == ''",
      "assert reverse_str('racecar') == 'racecar'"],
     'def reverse_str(s):\n    return s[::-1]'),
    (3, 'is_prime', 'Write is_prime(n) returning True iff n is a prime number.',
     ['assert is_prime(2) == True', 'assert is_prime(15) == False',
      'assert is_prime(97) == True', 'assert is_prime(1) == False'],
     'def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True'),
    (4, 'nth_fib', 'Write nth_fib(k) returning the k-th Fibonacci number (fib(0)=0, fib(1)=1).',
     ['assert nth_fib(0) == 0', 'assert nth_fib(1) == 1', 'assert nth_fib(10) == 55'],
     'def nth_fib(k):\n    a, b = 0, 1\n    for _ in range(k):\n        a, b = b, a + b\n    return a'),
    (5, 'gcd', 'Write gcd(a, b) returning the greatest common divisor.',
     ['assert gcd(12, 18) == 6', 'assert gcd(17, 5) == 1', 'assert gcd(100, 80) == 20'],
     'def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a'),
]
_REF_BY_NAME = {e[1]: e[4] for e in _CODE_BANK}


# ============================================================ problems + curriculum
@dataclass
class Problem:
    pid: int
    domain: str
    prompt: str
    difficulty: int
    signature: str
    expected: Optional[str] = None
    tests: Optional[List[str]] = None
    func: Optional[str] = None


class Curriculum:
    '''Generates fresh, difficulty-scaled problems on demand.'''

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)
        self._n = 0

    def _pid(self) -> int:
        self._n += 1
        return self._n

    def gen(self, domain: str, difficulty: int) -> Problem:
        if domain == 'math':
            return self._math(difficulty)
        return self._code(difficulty)

    def _math(self, d: int) -> Problem:
        d = max(1, min(d, 8))
        n_ops = 1 + d // 2
        hi = 6 + 2 * d
        parts = [str(self.rng.randint(1, hi))]
        for _ in range(n_ops):
            parts.append(self.rng.choice(['+', '-', '*']))
            parts.append(str(self.rng.randint(1, hi)))
        expr = ' '.join(parts)
        expected = str(safe_eval(expr))
        return Problem(pid=self._pid(), domain='math', difficulty=d,
                       prompt='Compute the exact value of: ' + expr,
                       expected=expected, signature='math:' + expr)

    def _code(self, d: int) -> Problem:
        d = max(1, min(d, 5))
        cands = [e for e in _CODE_BANK if e[0] <= d] or _CODE_BANK[:1]
        entry = cands[-1] if self.rng.random() < 0.6 else self.rng.choice(cands)
        diff, name, prompt, tests, _ref = entry
        return Problem(pid=self._pid(), domain='code', difficulty=diff,
                       prompt=prompt, tests=list(tests), func=name,
                       signature='code:' + name)


# ============================================================ grader
class Grader:
    '''Grades a candidate solution by EXECUTION / exact value -- never by vibes.'''

    def __init__(self, exec_timeout: float = 6.0):
        self.exec_timeout = exec_timeout

    def grade(self, problem: Problem, solution: str) -> Dict[str, Any]:
        if problem.domain == 'math':
            return self._grade_math(problem, solution)
        return self._grade_code(problem, solution)

    def _grade_math(self, p: Problem, sol: str) -> Dict[str, Any]:
        cand = (sol or '').strip()
        nums = re.findall(r'-?\d+', cand)
        got = nums[-1] if nums else cand
        passed = got == str(p.expected)
        return {'passed': passed, 'score': 1.0 if passed else 0.0,
                'detail': 'expected ' + str(p.expected) + ' got ' + str(got)}

    def _grade_code(self, p: Problem, sol: str) -> Dict[str, Any]:
        code = _extract_code_block(sol)
        if p.func and ('def ' + p.func) not in code:
            return {'passed': False, 'score': 0.0, 'detail': 'missing function ' + p.func}
        harness = code + '\n' + '\n'.join(p.tests or []) + "\nprint('GRADE_OK')\n"
        res = _run_python(harness, self.exec_timeout)
        passed = bool(res['ok']) and 'GRADE_OK' in (res['out'] or '')
        return {'passed': passed, 'score': 1.0 if passed else 0.0,
                'detail': (res['err'] or '')[:160]}


# ============================================================ difficulty + elo
class Difficulty:
    '''Curriculum controller -- keeps the rolling solve-rate inside a target band
    so problems track the edge of the student ability.'''

    def __init__(self, start: int = 2, lo: float = 0.5, hi: float = 0.8,
                 mn: int = 1, mx: int = 8, window: int = 6):
        self.d = start
        self.lo = lo
        self.hi = hi
        self.mn = mn
        self.mx = mx
        self.window = window
        self.hist: List[int] = []

    def record(self, passed: bool) -> int:
        self.hist.append(1 if passed else 0)
        self.hist = self.hist[-self.window:]
        if len(self.hist) >= self.window:
            rate = sum(self.hist) / len(self.hist)
            if rate > self.hi and self.d < self.mx:
                self.d += 1
                self.hist = []
            elif rate < self.lo and self.d > self.mn:
                self.d -= 1
                self.hist = []
        return self.d


def _elo_update(m: float, p: float, score: float, k: float = 32.0) -> float:
    expected = 1.0 / (1.0 + 10 ** ((p - m) / 400.0))
    return m + k * (score - expected)


# ============================================================ engine
class ArenaEngine:
    '''Runs the self-play loop and emits only execution-verified training rows.'''

    def __init__(self, student_fn: Callable[[Problem], str],
                 emit_sink: Optional[Callable[[dict], None]] = None,
                 cfg: Optional[ArenaConfig] = None):
        self.student = student_fn
        self.emit = emit_sink
        self.cfg = cfg or ArenaConfig.from_env()
        self.cur = Curriculum(seed=self.cfg.seed)
        self.grader = Grader(exec_timeout=self.cfg.exec_timeout)
        self.diff = Difficulty(start=self.cfg.difficulty_start, lo=self.cfg.target_lo,
                               hi=self.cfg.target_hi, mn=1, mx=self.cfg.max_difficulty,
                               window=self.cfg.window)
        self.model_elo = 1000.0
        self._seen = set()
        self.rows: List[dict] = []

    def _row(self, p: Problem, sol: str) -> dict:
        answer = _extract_code_block(sol) if p.domain == 'code' else str(p.expected)
        return {'question': p.prompt, 'answer': answer,
                'reasoning': 'self-play @ difficulty ' + str(p.difficulty) +
                             '; execution-graded PASS',
                'domain': p.domain, 'difficulty': p.difficulty, 'agreement': 1.0,
                'teachers': 'damru-arena', 'kind': 'self_play_verified',
                'verified': True}

    def run(self, rounds: Optional[int] = None) -> Dict[str, Any]:
        rounds = rounds or self.cfg.rounds
        t0 = time.time()
        solved = 0
        emitted = 0
        per_domain: Dict[str, Dict[str, int]] = {}
        for i in range(rounds):
            domain = self.cfg.domains[i % len(self.cfg.domains)]
            p = self.cur.gen(domain, self.diff.d)
            try:
                sol = self.student(p)
            except Exception:
                sol = ''
            g = self.grader.grade(p, sol)
            problem_elo = 800.0 + p.difficulty * 80.0
            self.model_elo = _elo_update(self.model_elo, problem_elo, g['score'])
            self.diff.record(g['passed'])
            per_domain.setdefault(domain, {'n': 0, 'solved': 0})
            per_domain[domain]['n'] += 1
            if g['passed']:
                solved += 1
                per_domain[domain]['solved'] += 1
                if p.signature not in self._seen:
                    self._seen.add(p.signature)
                    row = self._row(p, sol)
                    self.rows.append(row)
                    emitted += 1
                    if self.emit is not None:
                        try:
                            self.emit(row)
                        except Exception:
                            pass
        return {'rounds': rounds, 'solved': solved,
                'solve_rate': round(solved / max(1, rounds), 3),
                'rows_emitted': emitted, 'unique_rows': len(self._seen),
                'final_difficulty': self.diff.d, 'model_elo': round(self.model_elo, 1),
                'per_domain': per_domain, 'elapsed': round(time.time() - t0, 3)}


# ============================================================ offline self-test
def _selftest() -> int:
    print('== DAMRU ARENA self-test ==\n')
    ok = True

    # 1. safe evaluator
    t1 = (safe_eval('2 + 3 * 4') == 14 and safe_eval('(10 - 4) // 2') == 3
          and safe_eval('7 - 2 - 1') == 4)
    rejected = False
    try:
        safe_eval('__import__("os").system("ls")')
    except Exception:
        rejected = True
    t1 = t1 and rejected
    print('[1] safe math evaluator (values ok, unsafe rejected) -> ' + ('PASS' if t1 else 'FAIL'))
    ok = ok and t1

    cur = Curriculum(seed=1)
    grader = Grader()

    # 2. math generation + grading
    pm = cur.gen('math', 4)
    good = grader.grade(pm, 'the answer is ' + str(pm.expected))
    bad = grader.grade(pm, str(int(pm.expected) + 1))
    t2 = good['passed'] and not bad['passed']
    print('[2] math grade (correct PASS, wrong FAIL) -> ' + ('PASS' if t2 else 'FAIL'))
    ok = ok and t2

    # 3. code generation + execution grading
    pc = cur.gen('code', 1)
    ref = _REF_BY_NAME[pc.func]
    good_c = grader.grade(pc, '```python\n' + ref + '\n```')
    buggy = ref.replace('a + b', 'a - b').replace('s[::-1]', 's')
    bad_c = grader.grade(pc, '```python\n' + buggy + '\n```')
    t3 = good_c['passed'] and not bad_c['passed']
    print('[3] code grade by execution (ref PASS, buggy FAIL) -> ' + ('PASS' if t3 else 'FAIL'))
    ok = ok and t3

    # 4. adaptive difficulty up + down
    up = Difficulty(start=3, window=4)
    for _ in range(8):
        up.record(True)
    down = Difficulty(start=6, window=4)
    for _ in range(8):
        down.record(False)
    t4 = up.d > 3 and down.d < 6
    print('[4] difficulty adapts (up=' + str(up.d) + ', down=' + str(down.d) + ') -> ' + ('PASS' if t4 else 'FAIL'))
    ok = ok and t4

    # capable student: math exact, code reference
    def student(p: Problem) -> str:
        if p.domain == 'math':
            return 'After computing, the answer is ' + str(p.expected) + '.'
        return '```python\n' + _REF_BY_NAME[p.func] + '\n```'

    collected: List[dict] = []
    eng = ArenaEngine(student, emit_sink=collected.append,
                      cfg=ArenaConfig(rounds=16, difficulty_start=2, window=4, seed=3))
    rep = eng.run()
    t5 = (rep['rows_emitted'] > 0 and rep['solve_rate'] > 0.5
          and rep['model_elo'] > 1000.0 and rep['final_difficulty'] >= 2
          and all(r.get('verified') is True for r in collected))
    print('[5] arena run: solve_rate=' + str(rep['solve_rate']) + ' rows=' + str(rep['rows_emitted']) +
          ' elo=' + str(rep['model_elo']) + ' diff=' + str(rep['final_difficulty']) +
          ' -> ' + ('PASS' if t5 else 'FAIL'))
    ok = ok and t5

    # 6. every emitted row is unique + verified (no gamed/dup rows)
    sigs = [r['question'] + '|' + r['answer'] for r in collected]
    t6 = len(sigs) == len(set(sigs)) and rep['rows_emitted'] == len(collected)
    print('[6] emitted rows unique + verified (' + str(len(collected)) + ' rows) -> ' + ('PASS' if t6 else 'FAIL'))
    ok = ok and t6

    print('\n== RESULT: ' + ('ALL PASS \u2705' if ok else 'FAIL \u274c') + ' ==')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(_selftest())
