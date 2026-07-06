
import numpy as np
#from qutip import tensor, qeye, sigmax, sigmay, sigmaz
#from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PolyCollection
from itertools import combinations, product
import time



### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def _tag_dtype(max_value):
    ''' np.int64 while tags fit in it (fast), else python-int object arrays (unbounded size,
        so codes with more than 31 qubits can be studied)
    '''
    return np.int64 if max_value <= (1 << 62) else object


def get_tags(base, size, wmax=-1, wmin=0):
    '''
    returns a list of "tags" (eg. [000, 001, 002, 010, 011, ...]) written as int of base b \n
    base = the base b of each digit (0, 1, 2, ... b-1)                                     \n
    size = the size of the tag string                                                      \n
    wmax = the maximum weight / highest number of non-0 digit in a tag                     \n
    wmin = the minimum weight / lowest number of non-0 digit in a tag
    '''
    return _get_tags(base, size, wmax % (size+1), max(wmin, 0), _tag_dtype(base**size))


def _get_tags(base, size, wmax, wmin, dt):
    if wmin > wmax: return np.array([], dtype=dt)
    if wmax == 0:  return np.array([0], dtype=dt)
    if size == 1: return np.array(range(wmin, base), dtype=dt)
    tags_0 = _get_tags(base, size-1, min(size-1,wmax), wmin, dt)
    tags_1 = _get_tags(base, size-1, wmax-1, wmin-1, dt)
    parts = [tags_0*base] + [tags_1*base + b for b in range(1, base)]
    return np.concatenate(parts)


@np.vectorize(signature="(),()->()")
def letterify(tag, size):
    ''' translate a base 4 tag into a string with characters (I,X,Y,Z)
    '''
    string = ""
    for _ in range(size):
        match tag & 3:
            case 0: string = "I" + string
            case 1: string = "X" + string
            case 2: string = "Z" + string
            case 3: string = "Y" + string
        tag >>= 2
    return string


@np.vectorize(signature="(),()->()")
def binrepr(tag, size):
    ''' translate a binary tag into a string with characters (0,1)
    '''
    string = ""
    for _ in range(size):
        match tag & 1:
            case 0: string = "0" + string
            case 1: string = "1" + string
        tag >>= 1
    return string



### - - - - - - - - - - - - - - - - - - - - - - -



@np.vectorize(excluded=("g","gc"))
def get_IXYZ_error_proba(t, g=1, gc=1):
    '''
    gives single Pauli error probabilities given 
    the rate parameters of the Lindbladian equation \n
    -> (pI, pX, pY, pZ)
    '''
    pX = (1-np.exp(-2*g*t))/4
    pY = pX
    pZ = (1-np.exp(-(g+gc)*t))/2 - pX
    pI = 1-pX-pY-pZ
    return pI, pX, pY, pZ


def entropy_form(p):
    ''' p -> - p ln p
    '''
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    mask = (p > 0) & (p < 1)          # only these contribute; avoids log(0) warnings
    out[mask] = -p[mask] * np.log(p[mask])
    return out


@np.vectorize(signature="(),()->()")
def get_weight(err, coden):
    ''' compute the weight of a tag of base 4
    '''
    w = coden
    for _ in range(coden):
        w -= (err & 3) == 0
        err >>= 2
    return w



### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def symp(a, b, n, maskx=None):
    '''
    symplectic product of two n-qubit Paulis: 
    0 if they commute, 1 if they anticommute
    '''
    a, b = int(a), int(b)
    if maskx is None: maskx = int("01"*n, 2)
    ax, az = a & maskx, (a >> 1) & maskx
    bx, bz = b & maskx, (b >> 1) & maskx
    return ((ax & bz) ^ (az & bx)).bit_count() & 1



### - - - - - - - - - - - - - - - - - - - - - - -



class QEC_code:
    def __init__(self, name, code_string):
        self.name = name
        self.code_string = code_string
        self.n = len(code_string[0])
        self.m = len(code_string)
        self._set_code() # -> self.generators
        self._compute_logical_operators() # -> self.X, self.Z


    def _set_code(self):
        self.generators = []
        for sg in self.code_string:
            stab_string = ""
            for P in sg:
                match P:
                    case "I" | "0": stab_string += "0"
                    case "X" | "1": stab_string += "1"
                    case "Y" | "3": stab_string += "3"
                    case "Z" | "2": stab_string += "2"
            self.generators.append(int(stab_string,4)) 
            # stabilisers generators are stored as base 4 int representing its associated Pauli sequence


    def _compute_logical_operators(self):
        maskx = int("01"*self.n, 2)
        
        rows = [((s & maskx) << 1) | ((s >> 1) & maskx) for s in self.generators]

        pivot_cols, r = [], 0
        for col in range(2*self.n):
            sel = next((i for i in range(r, len(rows)) if (rows[i] >> col) & 1), None)
            if sel is None: continue
            rows[r], rows[sel] = rows[sel], rows[r]
            for i in range(len(rows)):
                if i != r and (rows[i] >> col) & 1: rows[i] ^= rows[r]
            pivot_cols.append(col)
            r += 1
            if r == len(rows): break

        free_cols = [c for c in range(2*self.n) if c not in set(pivot_cols)]
        kernel = []
        for fc in free_cols:
            v = 1 << fc
            for ri, pc in enumerate(pivot_cols):
                if (rows[ri] >> fc) & 1: v |= 1 << pc
            kernel.append(v)

        while kernel:
            a = kernel.pop()
            j = next((i for i, b in enumerate(kernel) if symp(a, b, self.n, maskx)), None)
            if j is None: continue
            b = kernel.pop(j)
            score = lambda p: (p & maskx).bit_count() - ((p >> 1) & maskx).bit_count()
            self.X, self.Z = (a, b) if score(a) >= score(b) else (b, a)
            break
    


### - - - - - - - - - - - - - - - - - - - - - - -



def logical(E, code):
    ''' 
    logical class of error E:
    I->0, X->1, Z->2, Y->3
    '''
    return (symp(E, code.X, code.n) << 1) | symp(E, code.Z, code.n)


def is_stabilizer(E, code):
    ''' True iff E is in the stabilizer group (0-syndrome AND trivial logical action)
    '''
    return code.logical(E) == 0 and all(symp(E, s, code.n) == 0 for s in code.generators)



### - - - - - - - - - - - - - - - - - - - - - - -



NQSC_code = QEC_code("NQSC", [
    "ZZIIIIIII",    
    "IZZIIIIII",
    "IIIZZIIII",
    "IIIIZZIII",
    "IIIIIIZZI",
    "IIIIIIIZZ",
    "XXXXXXIII",
    "IIIXXXXXX"
])

SQSC_code = QEC_code("SQSC", [
    "IIIXXXX",
    "IXXIIXX",
    "XIXIXIX",
    "IIIZZZZ",
    "IZZIIZZ",
    "ZIZIZIZ"
])

LMPZ_code = QEC_code("LMPZ", [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ"
])

X_code = QEC_code("X", [
    "IZZ",
    "ZZI"
])

Z_code = QEC_code("Z", [
    "IXX",
    "XXI"
])


def rotated_surface_code(d):
    if d % 2 == 0: raise ValueError
    code = []
    for y in range(d-1): # bulk 4qubits
        for x in range(d-1): 
            stab = ["I"]*d**2
            op = "Z" if (x+y) % 2 == 0 else "X"
            stab[x  +d*y  ] = op
            stab[x+1+d*y  ] = op
            stab[x  +d*y+d] = op
            stab[x+1+d*y+d] = op
            code.append("".join(stab))
    for i in range(0,d-1,2): # outer 2qubits # up
        stab = ["I"]*d**2
        stab[i] = "X"
        stab[i+1] = "X"
        code.append("".join(stab))
    for i in range(0,d-1,2): # right
        stab = ["I"]*d**2
        stab[(i+1)*d-1] = "Z"
        stab[(i+2)*d-1] = "Z"
        code.append("".join(stab))
    for i in range(0,d-1,2): # down
        stab = ["I"]*d**2
        stab[-1-i] = "X"
        stab[-2-i] = "X"
        code.append("".join(stab))
    for i in range(0,d-1,2): # left
        stab = ["I"]*d**2
        stab[-(i+1)*d] = "Z"
        stab[-(i+2)*d] = "Z"
        code.append("".join(stab))
    return QEC_code(f"{d}RSC", code)



### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def get_proba(err, p_unit, coden):
    ''' 
    compute the probability associated to each errors given single error probabilities \n
    err = error tags or array of error tags                                            \n
    p_unit = (pI, pX, pY, pZ)
    '''
    err = np.asarray(err)
    p_unit = np.asarray(p_unit, dtype=float)[[0, 1, 3, 2]]  # (I,X,Y,Z) -> digit order (I,X,Z,Y)
    proba = np.ones(err.shape, dtype=float)
    for k in range(coden):
        proba *= p_unit[((err >> (2*k)) & 3).astype(np.int64)]  # digit is 0..3, safe even for big-int tags
    return proba


def _popcount_parity(x): # parity (popcount & 1) of each element, fully vectorized
    x = np.asarray(x)
    if x.dtype == object: # python big ints (more than 31 qubits) -> unbounded bit_count
        return np.frompyfunc(lambda v: int(v).bit_count() & 1, 1, 1)(x)
    if hasattr(np, "bitwise_count"): # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int64) & 1
    x = x.astype(np.uint64) # SWAR popcount fallback
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64) & 1

def get_syndrome(error_tags, code):
    '''
    compute the syndroms of each errors     \n
    err = error tags or array of error tags
    '''
    big = code.n > 31                                      # tags need more than 62 bits -> python big ints
    error_tags = np.asarray(error_tags, dtype=object) if big else np.asarray(error_tags)
    scalar = error_tags.ndim == 0                          # 0-d object arrays misbehave -> work in 1-d
    error_tags = np.atleast_1d(error_tags)
    maskx = np.array(int("01"*code.n, 2), dtype=object) if big else int("01"*code.n, 2)
    ex = error_tags & maskx
    ez = (error_tags >> 1) & maskx
    syndrome = np.zeros(error_tags.shape, dtype=(object if big else np.int64))
    for S in code.generators:
        S = int(S)
        sx = S & maskx
        sz = (S >> 1) & maskx
        anti = (ex & sz) ^ (ez & sx)
        syndrome = (syndrome << 1) | _popcount_parity(anti)
    return syndrome[0] if scalar else syndrome



### - - - - - - - - - - - - - - - - - - - - - - -



def find_syndrome_errors(error_syndromes):
    ''' syndrome[error_ind] -> {syndrome:error_ind[]}
    '''
    syndrome_errors = {}
    for i,s in enumerate(error_syndromes):
        if s in syndrome_errors: syndrome_errors[s].append(i)
        else: syndrome_errors[s] = [i]
    return syndrome_errors


def get_syndrome_proba(syndrome_errors, error_proba):
    ''' -> proba[syndrome_ind]
    '''
    syndrome_proba = [0]*len(syndrome_errors)
    for i,L in enumerate(syndrome_errors):
        for j in L:
            syndrome_proba[i] += error_proba[j]
    return syndrome_proba


def sorted_syndrome_error_indices(syndrome_errors, error_proba, syndrome_proba=None):
    ''' sort syndromes by their probabilities
    '''
    if syndrome_proba is None: syndrome_proba = get_syndrome_proba(syndrome_errors, error_proba) 
    syndrome_ind = sorted(range(len(syndrome_errors)), reverse=True, key=lambda i: syndrome_proba[i])
    syndrome_error_ind = [sorted(syndrome_errors[si],  reverse=True, key=lambda i: error_proba[i]) for si in range(len(syndrome_errors))]
    return syndrome_ind, syndrome_error_ind


def apply_correction(syndrome_errors, error_proba, error_tags):
    ''' -> new_error[error_ind]
    '''
    corrected = [None]*len(error_tags)
    for si, errors in enumerate(syndrome_errors):
        mpe_proba = 0
        mpe_tag = None
        for ei in errors:
            if error_proba[ei] > mpe_proba:
                mpe_proba = error_proba[ei]
                mpe_tag = error_tags[ei]
        for ei in errors:
            corrected[ei] = error_tags[ei] ^ mpe_tag
    return corrected
            

def get_new_proba(error_proba, corrected_errors, corrected_error_tags):
    ''' proba[new_error_ind]
    '''
    new_proba = [0]*len(corrected_error_tags)
    for ei,nei in enumerate(corrected_errors):
        new_proba[nei] += error_proba[ei]
    return new_proba


def get_fidelity(error_proba, error_tags, code):
    ''' compute the corrected state fidelity (the success probability)
    '''
    return np.sum([p for p, t in zip(error_proba, error_tags) if code.logical(t) == 0])


def compute_threashold(p,p_prime):
    y = np.array(p_prime) - np.array(p)
    if y[0] > 0: return 0
    for i in range(1, len(y)):
        if y[i] > 0:
            return p[i-1] - y[i-1] * (p[i]-p[i-1]) / (y[i]-y[i-1])



### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def naive_corrections(code):
    N = 2**code.m
    corrections = {}
    for weight in range(code.n+1):
        print(f"weight {weight}")
        error_tags = get_tags(4, code.n, weight, weight)
        syndromes = get_syndrome(error_tags, code)
        for et,s in zip(error_tags, syndromes):
            if not s in corrections:
                corrections[s] = et
                N -= 1
                if N == 0: break
        if N == 0: break
    return corrections



### - - - - - - - - - - - - - - - - - - - - - - -



def _pure_error(syndrome, code):
    n, m = code.n, code.m
    maskx = int("01"*n, 2)
    rows = [((s & maskx) << 1) | ((s >> 1) & maskx) for s in code.generators]
    targets = [(int(syndrome) >> (m-1-i)) & 1 for i in range(m)]
    pivots, r = [], 0
    for col in range(2*n):
        sel = next((i for i in range(r, m) if (rows[i] >> col) & 1), None)
        if sel is None: continue
        rows[r], rows[sel] = rows[sel], rows[r]
        targets[r], targets[sel] = targets[sel], targets[r]
        for i in range(m):
            if i != r and (rows[i] >> col) & 1:
                rows[i] ^= rows[r]; targets[i] ^= targets[r]
        pivots.append((col, r)); r += 1
        if r == m: break
    if any(targets[i] for i in range(r, m)):
        raise ValueError(f"syndrome {int(syndrome)} is not reachable for code {code.name}")
    correction = 0
    for col, ri in pivots:
        if targets[ri]: correction |= 1 << col
    return correction


def stabilizer_group(code):
    group = np.array([0], dtype=_tag_dtype(1 << (2*code.n)))
    for g in code.generators:
        group = np.concatenate([group, group ^ int(g)])
    return group


class MLDecoder:
    def __init__(self, code, p_unit, max_weight=None):
        self.code = code
        self.p_unit = np.asarray(p_unit, dtype=float)
        W = code.n if max_weight is None else min(max(max_weight, 0), code.n)
        if max_weight is None and 4**code.n > 5_000_000:
            raise ValueError(f"exact ML on {code.name} enumerates 4**{code.n} errors; "
                             f"pass max_weight (eg 2-3) for an approximate ML decoder")
        self.cache = self._build_table(W)             # syndrome -> correction (most probable class)

    def _build_table(self, W):
        code, n = self.code, self.code.n
        errs = get_tags(4, n, W)                       # every error of weight <= W
        syn  = get_syndrome(errs, code)
        pr   = get_proba(errs, self.p_unit, n)
        maskx = int("01"*n, 2)                         # vectorised logical class of each error
        if errs.dtype == object: maskx = np.array(maskx, dtype=object)
        ex, ez = errs & maskx, (errs >> 1) & maskx
        sympv = lambda P: _popcount_parity((ex & ((P >> 1) & maskx)) ^ (ez & (P & maskx)))
        lg = (sympv(int(code.X)) << 1) | sympv(int(code.Z))
        proba, rep = {}, {}                            # syndrome -> [p per class] / [error per class]
        for e, s, p, l in zip(errs.tolist(), syn.tolist(), pr.tolist(), lg.tolist()):
            proba.setdefault(s, [0., 0., 0., 0.])[l] += p
            r = rep.setdefault(s, [None, None, None, None])
            if r[l] is None: r[l] = e
        return {s: rep[s][max((l for l in range(4) if rep[s][l] is not None),
                              key=lambda l: proba[s][l])] for s in proba}

    def __call__(self, syndrome):
        s = int(syndrome)
        if s not in self.cache:                        # error heavier than max_weight -> best effort
            self.cache[s] = _pure_error(s, self.code)
        return self.cache[s]



### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def random_tag(size, p_unit):
    err = 0
    random_range = [p_unit[0], p_unit[0]+p_unit[1], p_unit[0]+p_unit[1]+p_unit[2]]
    for i in range(size):
        rn = np.random.random()
        if   rn < random_range[0]: e = 0
        elif rn < random_range[1]: e = 1
        elif rn < random_range[2]: e = 3
        else:                      e = 2
        err = (err << 2) + e
    return err






### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



if __name__ == "__main__":
    ...